"""uvicorn entry point: ``python -m web.backend``."""
from __future__ import annotations

import argparse
import sys

import uvicorn

from .app import create_app
from .config import init_config, load_config


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser("web.backend")
    p.add_argument("--init", action="store_true", help="write a fresh .web_config.json and exit")
    p.add_argument("--host", default=None, help="override bind host")
    p.add_argument("--port", type=int, default=None, help="override bind port")
    p.add_argument("--reload", action="store_true", help="enable uvicorn auto-reload (dev only)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.init:
        cfg = init_config(force=True)
        print(f"Wrote .web_config.json (token: {cfg.token})")
        return 0

    cfg = load_config()
    host = args.host or cfg.host
    port = args.port or cfg.port
    print(f"[web] starting backend on http://{host}:{port}")
    print(f"[web] token: {cfg.token}")
    uvicorn.run(
        "web.backend.app:create_app",
        host=host,
        port=port,
        factory=True,
        reload=args.reload,
        ws_ping_interval=45.0,
        ws_ping_timeout=45.0,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
