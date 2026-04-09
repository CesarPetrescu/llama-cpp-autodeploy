#!/usr/bin/env python3
"""Launcher for the web backend.

Usage:
    python web_cli.py            # start the backend on host/port from config
    python web_cli.py --init     # write a fresh .web_config.json and print the token
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make sibling modules (loadmodel, memory_utils, autodevops) importable.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from web.backend.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
