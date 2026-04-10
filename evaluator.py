import base64
import io
import queue
import traceback

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from secmlt.adv.backends import Backends
from secmlt.adv.evasion.pgd import PGD
from secmlt.metrics.classification import Accuracy

try:
    from secmlt.models.pytorch.base_pytorch_nn import BasePyTorchClassifier as _Wrapper
except ImportError:
    from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier as _Wrapper  # type: ignore[no-redef]

try:
    import robustbench  # noqa: F401
    HAS_ROBUSTBENCH = True
except ImportError:
    HAS_ROBUSTBENCH = False

NORMALIZE_PARAMS = {
    "mnist":   {"mean": (0.1307,),                    "std": (0.3081,)},
    "cifar10": {"mean": (0.4914, 0.4822, 0.4465),     "std": (0.2023, 0.1994, 0.2010)},
}

DATASETS = {
    "mnist":   {"class": datasets.MNIST,   "num_classes": 10},
    "cifar10": {"class": datasets.CIFAR10, "num_classes": 10},
}

DATASET_CLASSES = {
    "cifar10": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
    "mnist":   ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
}

PRECONFIGURED_MODELS = [
    # ── PyTorch Hub ──────────────────────────────────────────────────────────
    {
        "id": "cifar10_resnet20",
        "label": "CIFAR-10 · ResNet-20 (PyTorch Hub)",
        "source": "hub",
        "hub_repo": "chenyaofo/pytorch-cifar-models",
        "model_name": "cifar10_resnet20",
        "model_kwargs": {"pretrained": True},
        "dataset": "cifar10",
        "normalize": False,
    },
    {
        "id": "cifar10_resnet56",
        "label": "CIFAR-10 · ResNet-56 (PyTorch Hub)",
        "source": "hub",
        "hub_repo": "chenyaofo/pytorch-cifar-models",
        "model_name": "cifar10_resnet56",
        "model_kwargs": {"pretrained": True},
        "dataset": "cifar10",
        "normalize": False,
    },
    {
        "id": "cifar10_vgg11_bn",
        "label": "CIFAR-10 · VGG-11 BN (PyTorch Hub)",
        "source": "hub",
        "hub_repo": "chenyaofo/pytorch-cifar-models",
        "model_name": "cifar10_vgg11_bn",
        "model_kwargs": {"pretrained": True},
        "dataset": "cifar10",
        "normalize": False,
    },
    {
        "id": "cifar10_mobilenetv2",
        "label": "CIFAR-10 · MobileNetV2-x0.5 (PyTorch Hub)",
        "source": "hub",
        "hub_repo": "chenyaofo/pytorch-cifar-models",
        "model_name": "cifar10_mobilenetv2_x0_5",
        "model_kwargs": {"pretrained": True},
        "dataset": "cifar10",
        "normalize": False,
    },
    # ── RobustBench ──────────────────────────────────────────────────────────
{
        "id": "rb_cifar10_Wong2020",
        "label": "CIFAR-10 · ResNet-50 L\u221e \u2014 Wong 2020 (RobustBench)",
        "source": "robustbench",
        "model_name": "Wong2020Fast",
        "dataset": "cifar10",
        "threat_model": "Linf",
        "normalize": False,
    },
    {
        "id": "rb_cifar10_Rice2020",
        "label": "CIFAR-10 · ResNet-18 L\u221e \u2014 Rice 2020 (RobustBench)",
        "source": "robustbench",
        "model_name": "Rice2020Overfitting",
        "dataset": "cifar10",
        "threat_model": "Linf",
        "normalize": False,
    },
    {
        "id": "rb_cifar10_Hendrycks2019",
        "label": "CIFAR-10 · WRN-28-10 L\u221e \u2014 Hendrycks 2019 (RobustBench)",
        "source": "robustbench",
        "model_name": "Hendrycks2019Using",
        "dataset": "cifar10",
        "threat_model": "Linf",
        "normalize": False,
    },
]

MODEL_BY_ID: dict = {m["id"]: m for m in PRECONFIGURED_MODELS}


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_pytorch_model(model_config: dict):
    """Load a raw PyTorch model from config. Returns (model, device)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source = model_config["source"]

    if source == "hub":
        model = torch.hub.load(
            model_config["hub_repo"],
            model_config["model_name"],
            trust_repo=True,
            skip_validation=True,
            **model_config.get("model_kwargs", {}),
        )
    elif source == "robustbench":
        if not HAS_ROBUSTBENCH:
            raise ImportError(
                "robustbench is not installed. Run: pip install robustbench"
            )
        from robustbench.utils import load_model as _rb_load
        model = _rb_load(
            model_name=model_config["model_name"],
            dataset=model_config["dataset"],
            threat_model=model_config["threat_model"],
        )
    else:
        raise ValueError(f"Unknown model source: {source!r}")

    model.eval()
    return model.to(device), device


def _make_wrapper(pytorch_model, model_config: dict):
    """Wrap a PyTorch model with optional normalization. Returns (wrapper, preprocessing)."""
    preprocessing = None
    if model_config.get("normalize"):
        p = NORMALIZE_PARAMS[model_config["dataset"]]
        preprocessing = transforms.Normalize(p["mean"], p["std"])
    return _Wrapper(pytorch_model, preprocessing=preprocessing), preprocessing


def _tensor_to_b64(tensor: torch.Tensor) -> str:
    """Convert a (C, H, W) float tensor in [0,1] to a base64-encoded PNG string."""
    arr = (tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    if arr.shape[2] == 1:
        arr = arr.squeeze(2)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _get_fooled(pytorch_model, preprocessing, adv_loader, device) -> torch.Tensor:
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


def get_sample_images(dataset_name: str, count: int = 16, start_index: int = 0) -> list:
    """Return sample test images as base64 PNG thumbnails."""
    ds_cfg = DATASETS[dataset_name]
    test_dataset = ds_cfg["class"](
        root="./data", train=False, download=True,
        transform=transforms.ToTensor(),
    )
    classes = DATASET_CLASSES[dataset_name]
    result = []
    for i in range(start_index, min(start_index + count, len(test_dataset))):
        img_tensor, label = test_dataset[i]
        result.append({
            "index": i,
            "label": int(label),
            "label_name": classes[int(label)],
            "image_b64": _tensor_to_b64(img_tensor),
        })
    return result


# ── Evaluation (security curve) ──────────────────────────────────────────────

def run_evaluation(config: dict, q: queue.Queue) -> None:
    """Blocking evaluation — intended to run in a daemon thread."""
    try:
        model_config = MODEL_BY_ID[config["model_id"]]
        dataset_name = model_config["dataset"]
        num_samples = config["num_samples"]
        perturbation_model = config["perturbation_model"]
        epsilon_values = config["epsilon_values"]
        num_steps = config["num_steps"]
        step_size = config["step_size"]
        backend = config.get("backend", Backends.NATIVE)

        q.put({"type": "progress", "message": f"Loading '{model_config['label']}'..."})
        pytorch_model, device = load_pytorch_model(model_config)
        model, preprocessing = _make_wrapper(pytorch_model, model_config)

        q.put({"type": "progress", "message": f"Loading {dataset_name} test set..."})
        ds_cfg = DATASETS[dataset_name]
        test_dataset = ds_cfg["class"](
            root="./data", train=False, download=True,
            transform=transforms.ToTensor(),
        )
        dataset_indices = list(range(min(num_samples, len(test_dataset))))
        num_samples = len(dataset_indices)
        test_loader = DataLoader(
            Subset(test_dataset, dataset_indices), batch_size=num_samples, shuffle=False
        )

        q.put({"type": "progress", "message": "Computing clean accuracy..."})
        clean_acc = float(Accuracy()(model, test_loader))
        q.put({"type": "clean_accuracy", "accuracy": clean_acc})

        fooled = torch.zeros(num_samples, dtype=torch.bool)
        n = len(epsilon_values)

        for i, eps in enumerate(epsilon_values):
            active_pos = (~fooled).nonzero(as_tuple=True)[0].tolist()
            q.put({
                "type": "progress",
                "message": (
                    f"PGD  \u03b5={eps:.5f}  [{i + 1}/{n}]"
                    f"  ({len(active_pos)} samples remaining)"
                ),
                "progress": i / n,
            })

            if eps == 0.0:
                robust_acc = clean_acc
            elif not active_pos:
                robust_acc = 0.0
            else:
                active_dataset_idx = [dataset_indices[j] for j in active_pos]
                active_loader = DataLoader(
                    Subset(test_dataset, active_dataset_idx),
                    batch_size=len(active_pos), shuffle=False,
                )
                attack = PGD(
                    perturbation_model=perturbation_model,
                    epsilon=eps,
                    num_steps=num_steps,
                    step_size=step_size,
                    backend=backend,
                )
                adv_loader = attack(model, active_loader)
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


# ── Visualization (single-image perturbation viewer) ─────────────────────────

def run_visualization(config: dict, q: queue.Queue) -> None:
    """Run PGD at increasing epsilon values on one image; stream perturbed images."""
    try:
        model_config = MODEL_BY_ID[config["model_id"]]
        dataset_name = model_config["dataset"]
        image_index = config["image_index"]
        perturbation_model = config["perturbation_model"]
        epsilon_max = config["epsilon_max"]
        epsilon_steps = config["epsilon_steps"]
        num_steps = config["num_steps"]
        step_size = config["step_size"]
        backend = config.get("backend", Backends.NATIVE)

        classes = DATASET_CLASSES[dataset_name]

        q.put({"type": "progress", "message": f"Loading '{model_config['label']}'..."})
        pytorch_model, device = load_pytorch_model(model_config)
        model, preprocessing = _make_wrapper(pytorch_model, model_config)

        q.put({"type": "progress", "message": f"Loading image {image_index}..."})
        ds_cfg = DATASETS[dataset_name]
        test_dataset = ds_cfg["class"](
            root="./data", train=False, download=True,
            transform=transforms.ToTensor(),
        )
        img_tensor, true_label = test_dataset[image_index]
        true_label = int(true_label)

        # Clean prediction
        with torch.no_grad():
            x = img_tensor.unsqueeze(0).to(device)
            inp = preprocessing(x) if preprocessing is not None else x
            probs = torch.softmax(pytorch_model(inp), dim=1)[0].cpu().numpy()
        clean_pred = int(np.argmax(probs))

        q.put({
            "type": "image",
            "epsilon": 0.0,
            "image_b64": _tensor_to_b64(img_tensor),
            "predicted_class": classes[clean_pred],
            "predicted_idx": clean_pred,
            "true_class": classes[true_label],
            "true_idx": true_label,
            "confidence": float(probs[clean_pred]),
            "correct": clean_pred == true_label,
            "probs": [float(p) for p in probs],
            "class_names": classes,
        })

        single_loader = DataLoader(
            Subset(test_dataset, [image_index]), batch_size=1, shuffle=False
        )
        epsilon_values = list(np.linspace(0, epsilon_max, epsilon_steps + 1))[1:]

        for i, eps in enumerate(epsilon_values):
            q.put({
                "type": "progress",
                "message": f"Attacking \u03b5={eps:.4f} [{i + 1}/{len(epsilon_values)}]...",
                "progress": i / len(epsilon_values),
            })

            attack = PGD(
                perturbation_model=perturbation_model,
                epsilon=eps,
                num_steps=num_steps,
                step_size=step_size,
                backend=backend,
            )
            adv_loader = attack(model, single_loader)

            for adv_x, _ in adv_loader:
                adv_img = adv_x[0]
                with torch.no_grad():
                    x = adv_img.unsqueeze(0).to(device)
                    inp = preprocessing(x) if preprocessing is not None else x
                    probs = torch.softmax(pytorch_model(inp), dim=1)[0].cpu().numpy()
                adv_pred = int(np.argmax(probs))

                q.put({
                    "type": "image",
                    "epsilon": float(eps),
                    "image_b64": _tensor_to_b64(adv_img.cpu()),
                    "predicted_class": classes[adv_pred],
                    "predicted_idx": adv_pred,
                    "true_class": classes[true_label],
                    "true_idx": true_label,
                    "confidence": float(probs[adv_pred]),
                    "correct": adv_pred == true_label,
                    "probs": [float(p) for p in probs],
                    "class_names": classes,
                })

        q.put({"type": "done"})

    except Exception as exc:
        q.put({"type": "error", "message": str(exc), "traceback": traceback.format_exc()})
