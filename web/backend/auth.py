"""Bearer-token auth for the web backend."""
from __future__ import annotations

import hmac
from typing import Optional

from fastapi import Header, HTTPException, Query, Request, status

from .config import WebConfig


def _check_token(provided: Optional[str], expected: str) -> bool:
    if not provided or not expected:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def make_http_dependency(cfg: WebConfig):
    """Create a FastAPI dependency that validates Authorization: Bearer <token>."""

    async def dependency(
        request: Request,
        authorization: Optional[str] = Header(default=None),
        token_query: Optional[str] = Query(default=None, alias="token"),
    ) -> None:
        token: Optional[str] = None
        if authorization:
            parts = authorization.split(None, 1)
            if len(parts) == 2 and parts[0].lower() == "bearer":
                token = parts[1].strip()
        if token is None and token_query:
            token = token_query
        if not _check_token(token, cfg.token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing bearer token.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return dependency


def verify_ws_token(provided: Optional[str], cfg: WebConfig) -> bool:
    """Constant-time comparison used by WebSocket handlers (query-string token)."""
    return _check_token(provided, cfg.token)
