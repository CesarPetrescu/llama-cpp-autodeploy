"""WebSocket endpoints for streaming instance / build / benchmark logs."""
from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status

from ..auth import verify_ws_token

router = APIRouter()


async def _stream_buffer(ws: WebSocket, buffer, include_history: bool) -> None:
    queue = buffer.subscribe()
    try:
        if include_history:
            for line in buffer.snapshot():
                await ws.send_text(line.rstrip("\n"))
        while True:
            line = await queue.get()
            await ws.send_text(line.rstrip("\n"))
    except WebSocketDisconnect:
        return
    except asyncio.CancelledError:
        raise
    except Exception:
        return
    finally:
        buffer.unsubscribe(queue)


@router.websocket("/instances/{instance_id}/logs")
async def instance_logs(
    websocket: WebSocket,
    instance_id: str,
    token: Optional[str] = Query(default=None),
    history: int = Query(default=1),
) -> None:
    cfg = websocket.app.state.cfg
    if not verify_ws_token(token, cfg):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    manager = websocket.app.state.manager
    buffer = manager.get_live_instance_buffer(instance_id)
    if buffer is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    await websocket.accept()
    await _stream_buffer(websocket, buffer, include_history=bool(history))


@router.websocket("/builds/{build_id}/logs")
async def build_logs(
    websocket: WebSocket,
    build_id: str,
    token: Optional[str] = Query(default=None),
    history: int = Query(default=1),
) -> None:
    cfg = websocket.app.state.cfg
    if not verify_ws_token(token, cfg):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    manager = websocket.app.state.manager
    buffer = manager.get_live_build_buffer(build_id)
    if buffer is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    await websocket.accept()
    await _stream_buffer(websocket, buffer, include_history=bool(history))


@router.websocket("/benchmarks/{benchmark_id}/logs")
async def benchmark_logs(
    websocket: WebSocket,
    benchmark_id: str,
    token: Optional[str] = Query(default=None),
    history: int = Query(default=1),
) -> None:
    cfg = websocket.app.state.cfg
    if not verify_ws_token(token, cfg):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    manager = websocket.app.state.manager
    buffer = manager.get_live_benchmark_buffer(benchmark_id)
    if buffer is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    await websocket.accept()
    await _stream_buffer(websocket, buffer, include_history=bool(history))
