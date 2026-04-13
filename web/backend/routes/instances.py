"""CRUD + lifecycle endpoints for managed llama-server instances."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()


class InstanceConfig(BaseModel):
    """Subset of LoadModelState fields relevant to launching llama-server."""

    mode: str = "llm"  # llm | embed
    model_ref: str = ""
    models_dir: Optional[str] = None
    hf_token: Optional[str] = None
    host: str = "127.0.0.1"
    port: int = 45540
    n_gpu_layers: Optional[int] = 999
    gpu_strategy: Optional[str] = "balanced"
    gpu_devices: Optional[str] = "all"
    auto_split_policy: Optional[str] = "vram"
    tensor_split: Optional[str] = None
    split_mode: Optional[str] = None
    ctx_size: Optional[int] = None
    n_cpu_moe: Optional[int] = None
    cpu_moe: bool = False
    mmproj: Optional[str] = None
    jinja: bool = False
    reasoning_format: Optional[str] = None
    no_context_shift: bool = False
    extra_flags: str = ""


class CreateInstanceRequest(BaseModel):
    name: str = Field(default="", max_length=120)
    config: InstanceConfig
    auto_start: bool = True


@router.get("")
async def list_instances(request: Request) -> Dict[str, Any]:
    manager = request.app.state.manager
    return {"instances": manager.list_instances()}


@router.post("")
async def create_instance(payload: CreateInstanceRequest, request: Request) -> Dict[str, Any]:
    manager = request.app.state.manager
    try:
        inst = await manager.create_instance(
            name=payload.name,
            config=payload.config.model_dump(),
            auto_start=payload.auto_start,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    data = manager.serialize_instance(inst.record.id)
    return {"instance": data}


@router.post("/recover")
async def recover_instances(request: Request) -> Dict[str, Any]:
    manager = request.app.state.manager
    recovered = await manager.recover_instances()
    recovered_payload = []
    for inst in recovered:
        data = manager.serialize_instance(inst.record.id)
        if data is not None:
            recovered_payload.append(data)
    return {
        "recovered": recovered_payload,
        "instances": manager.list_instances(),
    }


@router.get("/{instance_id}")
async def get_instance(instance_id: str, request: Request, tail: int = 500) -> Dict[str, Any]:
    manager = request.app.state.manager
    data = manager.serialize_instance(instance_id)
    if data is None:
        raise HTTPException(status_code=404, detail="instance not found")
    logs: List[str] = manager.get_instance_logs(instance_id, tail=tail)
    return {"instance": data, "logs": logs}


@router.post("/{instance_id}/start")
async def start_instance(instance_id: str, request: Request) -> Dict[str, Any]:
    manager = request.app.state.manager
    try:
        await manager.start_instance(instance_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="instance not found")
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"instance": manager.serialize_instance(instance_id)}


@router.post("/{instance_id}/stop")
async def stop_instance(instance_id: str, request: Request) -> Dict[str, Any]:
    manager = request.app.state.manager
    try:
        await manager.stop_instance(instance_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="instance not found")
    return {"instance": manager.serialize_instance(instance_id)}


@router.post("/{instance_id}/restart")
async def restart_instance(instance_id: str, request: Request) -> Dict[str, Any]:
    manager = request.app.state.manager
    try:
        await manager.restart_instance(instance_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="instance not found")
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"instance": manager.serialize_instance(instance_id)}


@router.delete("/{instance_id}")
async def delete_instance(instance_id: str, request: Request) -> Dict[str, Any]:
    manager = request.app.state.manager
    ok = await manager.delete_instance(instance_id)
    if not ok:
        raise HTTPException(status_code=404, detail="instance not found")
    return {"deleted": instance_id}
