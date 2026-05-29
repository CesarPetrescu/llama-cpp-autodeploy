"""Model library endpoints: scan local GGUFs, fetch from HF, rename, delete."""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

import loadmodel  # type: ignore
import memory_utils  # type: ignore

router = APIRouter()


_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._\-][A-Za-z0-9._\- /]*\.gguf$")


def _resolve_models_dir(request: Request) -> Path:
    cfg = request.app.state.cfg
    models_dir = Path(cfg.models_dir).expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def _resolve_model_path(models_dir: Path, name: str) -> Path:
    """Resolve ``name`` inside ``models_dir``, refusing traversal attempts."""
    if not name or ".." in name.split("/"):
        raise HTTPException(status_code=400, detail="Invalid model name.")
    candidate = (models_dir / name).resolve()
    try:
        candidate.relative_to(models_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Path escapes models_dir.") from exc
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"Model not found: {name}")
    if candidate.suffix.lower() != ".gguf":
        raise HTTPException(status_code=400, detail="Not a .gguf file.")
    return candidate


def _assert_not_in_use(manager, model_path: Path) -> None:
    """Prevent mutating a model file while a managed instance is actively using it."""
    target = model_path.resolve()
    for inst in manager.list_instances():
        if not inst.get("alive"):
            continue
        if inst.get("status") not in ("running", "stopping"):
            continue
        cmdline = inst.get("cmdline") or []
        for tok in cmdline:
            try:
                resolved = Path(tok).resolve()
            except OSError:
                continue
            if resolved == target:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Model is in use by running instance "
                        f"'{inst.get('name')}'. Stop it first."
                    ),
                )


def _gguf_metadata(path: Path) -> Dict[str, Any]:
    try:
        size = path.stat().st_size
    except OSError:
        size = None
    params = memory_utils._infer_params_from_name(path.name)  # noqa: SLF001 (reuse inference)
    quant_bits = memory_utils._infer_quant_bits(path.name)   # noqa: SLF001
    quant_tag = None
    if quant_bits is not None:
        for pattern, bits in memory_utils._QUANT_PATTERNS:  # noqa: SLF001
            if bits == quant_bits and pattern in path.name.upper():
                quant_tag = pattern
                break
    return {
        "name": path.name,
        "path": str(path),
        "size": size,
        "size_h": memory_utils.format_bytes(size),
        "params": params,
        "params_h": memory_utils.format_params(params),
        "quant": quant_tag,
    }


@router.get("/local")
async def list_local(request: Request) -> Dict[str, Any]:
    models_dir = _resolve_models_dir(request)
    out: List[Dict[str, Any]] = []
    for p in sorted(models_dir.rglob("*.gguf")):
        rel = p.relative_to(models_dir)
        meta = _gguf_metadata(p)
        meta["rel"] = str(rel)
        out.append(meta)
    return {"models_dir": str(models_dir), "models": out}


class RenameRequest(BaseModel):
    new_name: str


@router.post("/local/{name:path}/rename")
async def rename_model(name: str, payload: RenameRequest, request: Request) -> Dict[str, Any]:
    models_dir = _resolve_models_dir(request)
    src = _resolve_model_path(models_dir, name)

    new_name = payload.new_name.strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="new_name is required.")
    if not new_name.lower().endswith(".gguf"):
        new_name += ".gguf"
    if not _SAFE_NAME_RE.match(new_name):
        raise HTTPException(
            status_code=400,
            detail="new_name must contain only letters, digits, dot, dash, underscore, slash, space, and end in .gguf",
        )

    dest = (models_dir / new_name).resolve()
    try:
        dest.relative_to(models_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail="new_name escapes models_dir.")
    if dest.exists():
        raise HTTPException(status_code=409, detail=f"{new_name} already exists.")

    manager = request.app.state.manager
    _assert_not_in_use(manager, src)

    dest.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dest)
    return {"model": _gguf_metadata(dest), "old_name": name, "new_name": new_name}


@router.delete("/local/{name:path}")
async def delete_model(name: str, request: Request) -> Dict[str, Any]:
    models_dir = _resolve_models_dir(request)
    target = _resolve_model_path(models_dir, name)
    manager = request.app.state.manager
    _assert_not_in_use(manager, target)
    try:
        target.unlink()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {exc}")
    return {"deleted": name, "path": str(target)}


@router.get("/binary-caps")
async def binary_caps() -> Dict[str, Any]:
    primary, secondary = loadmodel.detect_ubatch_flag()
    help_out = ""
    try:
        from web.backend.config import REPO_ROOT  # local import to avoid cycles
        llama_server = REPO_ROOT / "bin" / "llama-server"
        if llama_server.exists():
            help_out = loadmodel.run_capture([str(llama_server), "--help"])
    except Exception:
        help_out = ""
    return {
        "ubatch_primary": primary,
        "ubatch_secondary": secondary,
        "has_cpu_moe": "--cpu-moe" in help_out,
        "has_n_cpu_moe": "--n-cpu-moe" in help_out,
        "has_flash_attn": "--flash-attn" in help_out,
    }


class DownloadRequest(BaseModel):
    spec: str
    hf_token: Optional[str] = None


@router.post("/download")
async def download_model(payload: DownloadRequest, request: Request) -> Dict[str, Any]:
    cfg = request.app.state.cfg
    models_dir = Path(cfg.models_dir).expanduser()
    loop = asyncio.get_running_loop()
    try:
        path: Path = await loop.run_in_executor(
            None, loadmodel.resolve_gguf, payload.spec, models_dir, payload.hf_token or None
        )
    except SystemExit as exc:
        raise HTTPException(status_code=400, detail=f"Download failed: {exc}") from exc
    except Exception as exc:  # pragma: no cover - network errors
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"model": _gguf_metadata(path)}
