import queue
import traceback

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from secmlt.adv.backends import Backends
from secmlt.adv.evasion.pgd import PGD
from secmlt.metrics.classification import Accuracy

try:
    from secmlt.models.pytorch.base_pytorch_nn import BasePyTorchClassifier as _Wrapper
except ImportError:
    from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier as _Wrapper  # type: ignore[no-redef]

# mean / std used when "Apply default normalization" is ticked
NORMALIZE_PARAMS = {
    "mnist":   {"mean": (0.1307,),                    "std": (0.3081,)},
    "cifar10": {"mean": (0.4914, 0.4822, 0.4465),     "std": (0.2023, 0.1994, 0.2010)},
}

DATASETS = {
    "mnist":   {"class": datasets.MNIST,   "num_classes": 10},
    "cifar10": {"class": datasets.CIFAR10, "num_classes": 10},
}


def _get_fooled(pytorch_model, preprocessing, adv_loader, device) -> torch.Tensor:
    """Return a bool tensor (length = batch size): True where the model is fooled."""
    pytorch_model.eval()
    results = []
    with torch.no_grad():
        for adv_x, y in adv_loader:
            adv_x = adv_x.to(device)
            if preprocessing is not None:
                adv_x = preprocessing(adv_x)
            preds = pytorch_model(adv_x).argmax(dim=1).cpu()
            results.append(preds != y)
    return torch.cat(results)


def run_evaluation(config: dict, q: queue.Queue) -> None:
    """Blocking evaluation — intended to run in a daemon thread."""
    try:
        hub_repo = config["hub_repo"]
        model_name = config["model_name"]
        model_kwargs = config.get("model_kwargs", {})
        dataset_name = config["dataset"]
        num_samples = config["num_samples"]
        perturbation_model = config["perturbation_model"]
        epsilon_values = config["epsilon_values"]
        num_steps = config["num_steps"]
        step_size = config["step_size"]
        backend = config.get("backend", Backends.NATIVE)

        # --- Load model ---
        q.put({"type": "progress", "message": f"Loading '{model_name}' from '{hub_repo}'..."})
        pytorch_model = torch.hub.load(
            hub_repo, model_name, trust_repo=True, skip_validation=True, **model_kwargs
        )
        pytorch_model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pytorch_model = pytorch_model.to(device)

        preprocessing = None
        if config.get("normalize"):
            p = NORMALIZE_PARAMS[dataset_name]
            preprocessing = transforms.Normalize(p["mean"], p["std"])
        model = _Wrapper(pytorch_model, preprocessing=preprocessing)

        # --- Load dataset (always raw [0,1]; normalization is inside the model wrapper) ---
        q.put({"type": "progress", "message": f"Loading {dataset_name} test set..."})
        ds_cfg = DATASETS[dataset_name]
        test_dataset = ds_cfg["class"](
            root="./data", train=False, download=True,
            transform=transforms.ToTensor(),
        )
        dataset_indices = list(range(min(num_samples, len(test_dataset))))
        num_samples = len(dataset_indices)  # clamp to actual dataset size
        test_loader = DataLoader(
            Subset(test_dataset, dataset_indices), batch_size=num_samples, shuffle=False
        )

        # --- Clean accuracy ---
        q.put({"type": "progress", "message": "Computing clean accuracy..."})
        clean_acc = float(Accuracy()(model, test_loader))
        q.put({"type": "clean_accuracy", "accuracy": clean_acc})

        # --- Security evaluation curve: PGD at each epsilon ---
        # `fooled[j]` is True once sample j has been successfully attacked at any prior epsilon.
        # We only run the attack on samples not yet fooled, then accumulate the mask.
        fooled = torch.zeros(num_samples, dtype=torch.bool)
        n = len(epsilon_values)

        for i, eps in enumerate(epsilon_values):
            active_pos = (~fooled).nonzero(as_tuple=True)[0].tolist()  # positions in [0, num_samples)

            q.put({
                "type": "progress",
                "message": (
                    f"PGD  ε={eps:.5f}  [{i + 1}/{n}]"
                    f"  ({len(active_pos)} samples remaining)"
                ),
                "progress": i / n,
            })

            if eps == 0.0:
                # Zero perturbation budget — attack is a no-op; reuse clean accuracy.
                robust_acc = clean_acc
            elif not active_pos:
                # Every sample is already fooled — robust accuracy is 0 from here on.
                robust_acc = 0.0
            else:
                active_dataset_idx = [dataset_indices[j] for j in active_pos]
                active_loader = DataLoader(
                    Subset(test_dataset, active_dataset_idx),
                    batch_size=len(active_pos),
                    shuffle=False,
                )
                attack = PGD(
                    perturbation_model=perturbation_model,
                    epsilon=eps,
                    num_steps=num_steps,
                    step_size=step_size,
                    backend=backend,
                )
                adv_loader = attack(model, active_loader)

                # Find which of the active samples are newly fooled.
                newly_fooled = _get_fooled(pytorch_model, preprocessing, adv_loader, device)
                active_pos_tensor = torch.tensor(active_pos)
                fooled[active_pos_tensor[newly_fooled]] = True

                robust_acc = (~fooled).sum().item() / num_samples

            q.put({
                "type": "result_point",
                "epsilon": eps,
                "accuracy": robust_acc,
                "index": i,
                "total": n,
                "progress": (i + 1) / n,
            })

        q.put({"type": "done", "message": "Evaluation complete!"})

    except Exception as exc:
        q.put({"type": "error", "message": str(exc), "traceback": traceback.format_exc()})
