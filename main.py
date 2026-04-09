import asyncio
import json
import queue
import threading
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from evaluator import DATASETS, run_evaluation

app = FastAPI(title="SecML-Torch Dashboard")
app.mount("/static", StaticFiles(directory="static"), name="static")

# job_id -> {"queue": Queue, "status": str}
jobs: Dict[str, Dict] = {}


class EvalParams(BaseModel):
    hub_repo: str
    model_name: str
    model_kwargs: Optional[Dict[str, Any]] = {}
    dataset: str = "cifar10"
    num_samples: int = 20
    perturbation_model: str = "linf"
    epsilon_min: float = 0.0
    epsilon_max: float = 0.1
    epsilon_steps: int = 10
    num_steps: int = 20
    step_size: float = 0.01
    normalize: bool = False
    backend: str = "native"

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        if v not in ("native", "foolbox", "advlib"):
            raise ValueError("backend must be 'native', 'foolbox', or 'advlib'")
        return v

    @field_validator("dataset")
    @classmethod
    def validate_dataset(cls, v: str) -> str:
        if v not in DATASETS:
            raise ValueError(f"dataset must be one of {list(DATASETS.keys())}")
        return v

    @field_validator("perturbation_model")
    @classmethod
    def validate_norm(cls, v: str) -> str:
        if v not in ("l1", "l2", "linf"):
            raise ValueError("perturbation_model must be 'l1', 'l2', or 'linf'")
        return v

    @field_validator("epsilon_steps")
    @classmethod
    def validate_steps(cls, v: int) -> int:
        if v < 2:
            raise ValueError("epsilon_steps must be >= 2")
        return v


@app.get("/", response_class=HTMLResponse)
async def serve_index() -> str:
    with open("static/index.html") as f:
        return f.read()


@app.get("/api/datasets")
async def list_datasets() -> List[str]:
    return list(DATASETS.keys())


@app.post("/api/evaluate")
async def start_evaluation(params: EvalParams) -> Dict[str, str]:
    epsilon_values: List[float] = list(
        np.linspace(params.epsilon_min, params.epsilon_max, params.epsilon_steps)
    )
    job_id = str(uuid.uuid4())
    q: queue.Queue = queue.Queue()
    jobs[job_id] = {"queue": q, "status": "running"}

    config = {
        "hub_repo": params.hub_repo,
        "model_name": params.model_name,
        "model_kwargs": params.model_kwargs or {},
        "dataset": params.dataset,
        "num_samples": params.num_samples,
        "perturbation_model": params.perturbation_model,
        "epsilon_values": epsilon_values,
        "num_steps": params.num_steps,
        "step_size": params.step_size,
        "normalize": params.normalize,
        "backend": params.backend,
    }
    thread = threading.Thread(target=run_evaluation, args=(config, q), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.get("/api/stream/{job_id}")
async def stream_results(job_id: str) -> StreamingResponse:
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    q = jobs[job_id]["queue"]
    loop = asyncio.get_event_loop()

    async def event_generator():
        while True:
            try:
                item = await loop.run_in_executor(None, lambda: q.get(timeout=60))
            except queue.Empty:
                yield ": keepalive\n\n"
                continue

            yield f"data: {json.dumps(item)}\n\n"
            if item["type"] in ("done", "error"):
                jobs[job_id]["status"] = item["type"]
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
