"""Public health check endpoint."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request

from ..config import REPO_ROOT

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict:
    cfg = request.app.state.cfg
    llama_server = REPO_ROOT / "bin" / "llama-server"
    return {
        "status": "ok",
        "service": "llama-cpp-autodeploy",
        "version": "0.1.0",
        "auth_required": True,
        "llama_server_present": llama_server.exists(),
        "models_dir": cfg.models_dir,
    }
