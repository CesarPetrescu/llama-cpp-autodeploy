"""GPU + memory planning endpoints."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from fastapi import APIRouter

import memory_utils  # type: ignore

router = APIRouter()


def _gpu_to_dict(gpu) -> Dict[str, Any]:
    return {
        "index": gpu.index,
        "name": gpu.name,
        "total": gpu.total,
        "free": gpu.free,
        "total_h": memory_utils.format_bytes(gpu.total),
        "free_h": memory_utils.format_bytes(gpu.free),
    }


@router.get("/gpus")
async def list_gpus() -> Dict[str, Any]:
    gpus = memory_utils.detect_gpus()
    return {"gpus": [_gpu_to_dict(g) for g in gpus]}


@router.post("/plan")
async def plan_memory(state: Dict[str, Any]) -> Dict[str, Any]:
    profile = memory_utils.estimate_memory_profile(dict(state), refresh=True)
    gpus: List[Dict[str, Any]] = []
    for usage in profile.gpus:
        gpus.append({
            "info": _gpu_to_dict(usage.info),
            "weights": usage.weights,
            "weights_h": memory_utils.format_bytes(usage.weights),
            "kv": usage.kv,
            "kv_h": memory_utils.format_bytes(usage.kv),
        })
    return {
        "source_label": profile.source_label,
        "model_label": profile.model_label,
        "quant": profile.quant,
        "param_count": profile.param_count,
        "param_count_h": memory_utils.format_params(profile.param_count),
        "weights_total": profile.weights_total,
        "weights_total_h": memory_utils.format_bytes(profile.weights_total),
        "weights_gpu": profile.weights_gpu,
        "weights_gpu_h": memory_utils.format_bytes(profile.weights_gpu),
        "weights_cpu": profile.weights_cpu,
        "weights_cpu_h": memory_utils.format_bytes(profile.weights_cpu),
        "kv_total": profile.kv_total,
        "kv_total_h": memory_utils.format_bytes(profile.kv_total),
        "ctx_size": profile.ctx_size,
        "layers_est": profile.layers_est,
        "gpus": gpus,
        "cpu_total": profile.cpu_total,
        "cpu_total_h": memory_utils.format_bytes(profile.cpu_total),
        "cpu_available": profile.cpu_available,
        "cpu_available_h": memory_utils.format_bytes(profile.cpu_available),
        "cpu_weights": profile.cpu_weights,
        "cpu_weights_h": memory_utils.format_bytes(profile.cpu_weights),
        "cpu_kv": profile.cpu_kv,
        "cpu_kv_h": memory_utils.format_bytes(profile.cpu_kv),
        "warnings": profile.warnings,
        "summary": memory_utils.profile_summary_lines(profile),
    }


@router.post("/auto-split")
async def auto_split(payload: Dict[str, Any]) -> Dict[str, Any]:
    policy = str(payload.get("policy") or "vram")
    split = memory_utils.auto_tensor_split(policy=policy)
    return {"tensor_split": split, "policy": policy}
