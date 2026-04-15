"""FastAPI application factory."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .auth import make_http_dependency
from .config import FRONTEND_DIST, STATE_PATH, WebConfig, load_config
from .process_manager import ProcessManager
from .routes import benchmarks, builds, events, health, instances, logs, memory, models
from .state import StateStore


def create_app(cfg: Optional[WebConfig] = None) -> FastAPI:
    cfg = cfg or load_config()
    store = StateStore(STATE_PATH)
    manager = ProcessManager(cfg, store)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await manager.startup()
        try:
            yield
        finally:
            await manager.shutdown()

    app = FastAPI(
        title="llama-cpp-autodeploy web backend",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.cfg = cfg
    app.state.store = store
    app.state.manager = manager

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    auth_dep = make_http_dependency(cfg)

    # Public routes
    app.include_router(health.router, prefix="/api")

    # Authenticated routes
    protected = [Depends(auth_dep)]
    app.include_router(memory.router, prefix="/api/memory", dependencies=protected)
    app.include_router(models.router, prefix="/api/models", dependencies=protected)
    app.include_router(instances.router, prefix="/api/instances", dependencies=protected)
    app.include_router(builds.router, prefix="/api/builds", dependencies=protected)
    app.include_router(benchmarks.router, prefix="/api/benchmarks", dependencies=protected)

    # WebSocket endpoints (auth handled inside the handler via query token)
    app.include_router(logs.router, prefix="/api")
    app.include_router(events.router, prefix="/api")

    # Static frontend (production build). /api takes precedence because it was
    # registered first.
    if FRONTEND_DIST.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
    else:
        @app.get("/")
        async def _placeholder() -> dict:
            return {
                "message": "llama-cpp-autodeploy backend is running.",
                "hint": "Build the frontend (cd web/frontend && npm install && npm run build) or run `npm run dev`.",
                "docs": "/docs",
            }

    return app
