# SecML-Torch Dashboard

A web-based dashboard for evaluating the adversarial robustness of PyTorch models. It runs PGD attacks across a range of perturbation magnitudes and plots the security evaluation curve (robust accuracy vs. ε) in real time. A second **Visualizer** page lets you inspect how a single image is perturbed as ε increases.

> [!WARNING]
> Models loaded via `torch.hub.load(..., trust_repo=True)`. Only point this tool at hub repos you trust.

## Requirements

- Python 3.9+
- PyTorch (CPU or CUDA)

Install dependencies:

```bash
pip install -r requirements.txt
```

RobustBench is installed directly from GitHub; a `models/` directory is created automatically to cache its weights (excluded from version control).

## Running the server

```bash
python main.py
```

Then open `http://localhost:8000` in your browser.

## Pages

| Page | URL | Description |
|---|---|---|
| **Evaluation** | `/` | Security evaluation curve (robust accuracy vs. ε) |
| **Visualizer** | `/visualize` | Per-image perturbation viewer |

## Usage

### 1. Model

Choose a pre-configured model from the dropdown. Available models:

| Label | Source | Dataset |
|---|---|---|
| CIFAR-10 · ResNet-20 | PyTorch Hub (`chenyaofo/pytorch-cifar-models`) | CIFAR-10 |
| CIFAR-10 · ResNet-56 | PyTorch Hub (`chenyaofo/pytorch-cifar-models`) | CIFAR-10 |
| CIFAR-10 · VGG-11 BN | PyTorch Hub (`chenyaofo/pytorch-cifar-models`) | CIFAR-10 |
| CIFAR-10 · MobileNetV2-x0.5 | PyTorch Hub (`chenyaofo/pytorch-cifar-models`) | CIFAR-10 |
| CIFAR-10 · PreActResNet-18 Linf — Wong 2020 | RobustBench | CIFAR-10 |
| CIFAR-10 · PreActResNet-18 Linf — Rice 2020 | RobustBench | CIFAR-10 |

RobustBench models are downloaded and cached in `./models/` on first use.

### 2. Dataset

Dataset and normalization are fixed per model. Samples are taken from the test split, downloaded automatically to `./data/` on first use.

| Field | Description |
|---|---|
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

### 6. Visualizer

Open `/visualize`. Select a model, an image from the gallery, and attack parameters, then click **Run**. The tool streams perturbed versions of the image at each ε value so you can see how the perturbation evolves and when the model is fooled.

## API

The server exposes a REST + SSE API usable without the UI.

### `GET /api/models`

Returns the list of pre-configured model descriptors.

### `GET /api/datasets`

Returns the list of supported dataset names.

### `GET /api/images/{dataset}`

Returns a page of test-set images as base64-encoded PNG thumbnails.

Query params: `count` (default 16), `start` (default 0).

### `POST /api/evaluate`

Start an evaluation job. Returns a `job_id`.

```json
{
  "model_id": "cifar10_resnet20",
  "num_samples": 100,
  "perturbation_model": "linf",
  "epsilon_min": 0.0,
  "epsilon_max": 0.03,
  "epsilon_steps": 10,
  "num_steps": 20,
  "step_size": 0.003,
  "backend": "native"
}
```

### `GET /api/stream/{job_id}`

Server-Sent Events stream for an evaluation job. Each event is a JSON object with a `type` field:

| type | Fields | Description |
|---|---|---|
| `progress` | `message`, `progress` (0–1) | Status update |
| `clean_accuracy` | `accuracy` | Clean accuracy on the test subset |
| `result_point` | `epsilon`, `accuracy`, `index`, `total`, `progress` | Robust accuracy at one ε |
| `done` | `message` | Evaluation finished |
| `error` | `message`, `traceback` | Unhandled exception |

### `POST /api/visualize`

Start a visualization job. Returns a `job_id`.

```json
{
  "model_id": "cifar10_resnet20",
  "image_index": 0,
  "perturbation_model": "linf",
  "epsilon_max": 0.03,
  "epsilon_steps": 10,
  "num_steps": 20,
  "step_size": 0.003,
  "backend": "native"
}
```

### `GET /api/visualize/stream/{job_id}`

Server-Sent Events stream for a visualization job. Each event is a JSON object with a `type` field:

| type | Fields | Description |
|---|---|---|
| `progress` | `message` | Status update |
| `image` | `epsilon`, `image_b64`, `predicted_class`, `predicted_idx`, `true_class`, `true_idx`, `fooled` | Perturbed image at one ε |
| `done` | `message` | Visualization finished |
| `error` | `message`, `traceback` | Unhandled exception |
