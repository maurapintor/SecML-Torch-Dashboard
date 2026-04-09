# SecML-Torch Dashboard

A web-based dashboard for evaluating the adversarial robustness of PyTorch models. It runs PGD attacks across a range of perturbation magnitudes and plots the security evaluation curve (robust accuracy vs. ε) in real time.

> [!WARNING]
> Models are loaded via `torch.hub.load(..., trust_repo=True)`. Only point this tool at hub repos you trust.

## Requirements

- Python 3.9+
- PyTorch (CPU or CUDA)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the server

```bash
python main.py
```

Then open `http://localhost:8000` in your browser.

## Usage

### 1. Model

The dashboard loads models via `torch.hub`. Fill in:

| Field | Description | Example |
|---|---|---|
| **Torch Hub Repo** | GitHub repo in `owner/repo` format | `chenyaofo/pytorch-cifar-models` |
| **Model Name** | Entry point name registered in the hub | `cifar10_resnet20` |
| **Extra kwargs** | JSON object passed to `torch.hub.load` | `{"pretrained": true}` |

Any model loadable with `torch.hub.load(repo, name, **kwargs)` is supported.

### 2. Dataset

| Field | Description |
|---|---|
| **Dataset** | `CIFAR-10` or `MNIST`. Downloaded automatically on first use to `./data/`. |
| **Apply normalization** | Applies the standard mean/std normalization for the selected dataset inside the model wrapper, keeping inputs in [0, 1] for the attack. |
| **Test Samples** | Number of test-set samples to evaluate (starting from index 0). |

### 3. Attack — PGD

| Field | Description |
|---|---|
| **Backend** | `native` (secml-torch), `foolbox`, or `adv_lib`. |
| **Perturbation Norm** | L∞, L2, or L1 ball constraint. |
| **Epsilon min / max / steps** | Defines the ε grid as `numpy.linspace(min, max, steps)`. ε=0 is evaluated as clean accuracy without running the attack. |
| **PGD Steps** | Number of projected gradient descent iterations per attack. |
| **Step Size** | Per-step perturbation size (α). A common heuristic is `ε / steps * 2.5`. |

### 4. Running an evaluation

Click **Run Evaluation**. The progress bar updates as each ε is processed. Results appear in the chart and table as they arrive.

The evaluation is cumulative: once a sample is successfully attacked at ε_i, it is excluded from subsequent runs at larger ε values. This avoids redundant computation and ensures the curve is monotonically non-increasing.

### 5. Exporting results

Click **Download PDF** after an evaluation completes. The PDF includes the configuration, clean accuracy, the security evaluation curve, and a table of (ε, accuracy, drop) values.

## API

The server exposes a small REST + SSE API usable without the UI.

### `POST /api/evaluate`

Start an evaluation job. Returns a `job_id`.

```json
{
  "hub_repo": "chenyaofo/pytorch-cifar-models",
  "model_name": "cifar10_resnet20",
  "model_kwargs": {"pretrained": true},
  "dataset": "cifar10",
  "num_samples": 100,
  "perturbation_model": "linf",
  "epsilon_min": 0.0,
  "epsilon_max": 0.03,
  "epsilon_steps": 10,
  "num_steps": 20,
  "step_size": 0.003,
  "normalize": true,
  "backend": "native"
}
```

### `GET /api/stream/{job_id}`

Server-Sent Events stream. Each event is a JSON object with a `type` field:

| type | Fields | Description |
|---|---|---|
| `progress` | `message`, `progress` (0–1) | Status update |
| `clean_accuracy` | `accuracy` | Clean accuracy on the test subset |
| `result_point` | `epsilon`, `accuracy`, `index`, `total`, `progress` | Robust accuracy at one ε |
| `done` | `message` | Evaluation finished |
| `error` | `message`, `traceback` | Unhandled exception |

### `GET /api/datasets`

Returns the list of supported dataset names.
