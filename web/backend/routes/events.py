"""WebSocket pub-sub endpoints for live instance / build / benchmark updates."""
from __future__ import annotations

import asyncio
import json
from typing import Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status

from ..auth import verify_ws_token

router = APIRouter()


async def _pump(ws: WebSocket, broadcaster, initial_snapshot: dict) -> None:
    queue = broadcaster.subscribe()
    try:
        await ws.send_text(json.dumps(initial_snapshot))
        while True:
            payload = await queue.get()
            await ws.send_text(json.dumps(payload))
    except WebSocketDisconnect:
        return
    except asyncio.CancelledError:
        raise
    except Exception:
        return
    finally:
        broadcaster.unsubscribe(queue)


@router.websocket("/instances/events")
async def instance_events(
    websocket: WebSocket,
    token: Optional[str] = Query(default=None),
) -> None:
    cfg = websocket.app.state.cfg
    if not verify_ws_token(token, cfg):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    manager = websocket.app.state.manager
    await websocket.accept()
    await _pump(websocket, manager.instance_events, manager.instance_snapshot())


@router.websocket("/builds/events")
async def build_events(
    websocket: WebSocket,
    token: Optional[str] = Query(default=None),
) -> None:
    cfg = websocket.app.state.cfg
    if not verify_ws_token(token, cfg):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    manager = websocket.app.state.manager
    await websocket.accept()
    await _pump(websocket, manager.build_events, manager.build_snapshot())


@router.websocket("/benchmarks/events")
async def benchmark_events(
    websocket: WebSocket,
    token: Optional[str] = Query(default=None),
) -> None:
    cfg = websocket.app.state.cfg
    if not verify_ws_token(token, cfg):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    manager = websocket.app.state.manager
    await websocket.accept()
    await _pump(websocket, manager.benchmark_events, manager.benchmark_snapshot())
