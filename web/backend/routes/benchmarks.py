"""Benchmark manager endpoints (wraps llama-bench)."""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class BenchmarkRequest(BaseModel):
    name: Optional[str] = None
    model_ref: str
    models_dir: Optional[str] = None
    hf_token: Optional[str] = None
    gpu_devices: str = ""
    repetitions: int = 5
    delay: int = 0
    n_prompt: str = "512"
    n_gen: str = "128"
    pg: str = ""
    n_depth: str = "0"
    batch_size: str = "2048"
    ubatch_size: str = "512"
    threads: str = "8"
    n_gpu_layers: str = "99"
    split_mode: str = "layer"
    main_gpu: str = "0"
    tensor_split: str = ""
    flash_attn: bool = True
    embeddings: bool = False
    no_kv_offload: bool = False
    no_warmup: bool = False
    extra_flags: str = ""


def _validate_request(payload: BenchmarkRequest) -> None:
    if not payload.model_ref.strip():
        raise HTTPException(status_code=400, detail="model_ref is required")
    if payload.repetitions <= 0:
        raise HTTPException(status_code=400, detail="repetitions must be positive")
    if payload.delay < 0:
        raise HTTPException(status_code=400, detail="delay must be zero or positive")


@router.get("")
async def list_benchmarks(request: Request) -> Dict[str, Any]:
    manager = request.app.state.manager
    return {"benchmarks": manager.list_benchmarks()}


@router.post("")
async def start_benchmark(payload: BenchmarkRequest, request: Request) -> Dict[str, Any]:
    _validate_request(payload)
    manager = request.app.state.manager
    record = await manager.start_benchmark(payload.model_dump())
    return {"benchmark": manager.get_benchmark(record.id)}


@router.get("/{benchmark_id}")
async def get_benchmark(benchmark_id: str, request: Request) -> Dict[str, Any]:
    manager = request.app.state.manager
    data = manager.get_benchmark(benchmark_id)
    if data is None:
        raise HTTPException(status_code=404, detail="benchmark not found")
    logs = manager.get_benchmark_logs(benchmark_id, tail=500)
    return {"benchmark": data, "logs": logs}


@router.post("/{benchmark_id}/stop")
async def stop_benchmark(benchmark_id: str, request: Request) -> Dict[str, Any]:
    manager = request.app.state.manager
    await manager.stop_benchmark(benchmark_id)
    data = manager.get_benchmark(benchmark_id)
    if data is None:
        raise HTTPException(status_code=404, detail="benchmark not found")
    return {"benchmark": data}
