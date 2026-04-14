"""GPU + memory planning endpoints."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from fastapi import APIRouter, Request

import memory_utils  # type: ignore

router = APIRouter()


def _field(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _process_to_dict(process: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pid": process.get("pid"),
        "process_name": process.get("process_name"),
        "raw_process_name": process.get("raw_process_name"),
        "label": process.get("label"),
        "kind": process.get("kind"),
        "status": process.get("status"),
        "detail": process.get("detail"),
        "used_memory": process.get("used_memory"),
        "used_memory_h": memory_utils.format_bytes(process.get("used_memory")),
        "memory_percent": process.get("memory_percent"),
    }


def _gpu_to_dict(gpu: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "index": _field(gpu, "index"),
        "name": _field(gpu, "name"),
        "uuid": _field(gpu, "uuid"),
        "total": _field(gpu, "total"),
        "free": _field(gpu, "free"),
        "used": _field(gpu, "used"),
        "total_h": memory_utils.format_bytes(_field(gpu, "total")),
        "free_h": memory_utils.format_bytes(_field(gpu, "free")),
        "used_h": memory_utils.format_bytes(_field(gpu, "used")),
        "utilization_gpu": _field(gpu, "utilization_gpu"),
        "utilization_memory": _field(gpu, "utilization_memory"),
        "memory_percent": _field(gpu, "memory_percent"),
        "processes": [_process_to_dict(process) for process in (_field(gpu, "processes") or [])],
    }


def _system_to_dict(system: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "cpu_percent": system.get("cpu_percent"),
        "cpu_count_logical": system.get("cpu_count_logical"),
        "cpu_count_physical": system.get("cpu_count_physical"),
        "load_1": system.get("load_1"),
        "load_5": system.get("load_5"),
        "load_15": system.get("load_15"),
        "memory_total": system.get("memory_total"),
        "memory_total_h": memory_utils.format_bytes(system.get("memory_total")),
        "memory_available": system.get("memory_available"),
        "memory_available_h": memory_utils.format_bytes(system.get("memory_available")),
        "memory_used": system.get("memory_used"),
        "memory_used_h": memory_utils.format_bytes(system.get("memory_used")),
        "memory_percent": system.get("memory_percent"),
        "cores": system.get("cores", []),
    }


def _managed_processes(request: Request) -> Dict[int, Dict[str, Any]]:
    manager = getattr(request.app.state, "manager", None)
    if manager is None:
        return {}

    processes: Dict[int, Dict[str, Any]] = {}
    for inst in manager.list_instances():
        pid = inst.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            continue
        endpoint = None
        if inst.get("host") and inst.get("port") is not None:
            endpoint = f'{inst["host"]}:{inst["port"]}'
        processes[pid] = {
            "label": inst.get("name") or inst.get("id") or f"instance {pid}",
            "kind": "instance",
            "status": inst.get("status"),
            "detail": endpoint,
        }

    for build in manager.list_builds():
        pid = build.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            continue
        ref = build.get("config", {}).get("ref") if isinstance(build.get("config"), dict) else None
        processes[pid] = {
            "label": f'build {build.get("id") or pid}',
            "kind": "build",
            "status": build.get("status"),
            "detail": str(ref or "latest"),
        }
    return processes


@router.get("/gpus")
async def list_gpus(request: Request) -> Dict[str, Any]:
    gpus = memory_utils.detect_gpu_runtime(managed_processes=_managed_processes(request))
    system = memory_utils.detect_system_usage()
    return {
        "gpus": [_gpu_to_dict(g) for g in gpus],
        "system": _system_to_dict(system),
    }


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
