#!/usr/bin/env python3
"""Convenience wrapper to launch llama.cpp rpc-server with custom settings."""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BIN_DIR = SCRIPT_DIR / "bin"
RPC_SERVER = BIN_DIR / "rpc-server"


def ensure_binary() -> None:
    if not RPC_SERVER.exists():
        raise FileNotFoundError(
            f"{RPC_SERVER} not found.\n"
            "Build llama.cpp with the distributed RPC backend enabled (use autodevops_cli.py "
            "and toggle 'Enable distributed RPC backend')."
        )


def shell_join(args: list[str]) -> str:
    return " ".join(shlex.quote(str(a)) for a in args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch llama.cpp rpc-server with configurable host/port/cache.",
        epilog=(
            "Example:\n"
            "  python rpc_server_cli.py --host 0.0.0.0 --port 5515 --cache /tmp/llama-cache "
            "--devices 0,1\n"
            "Environment variables are passed through; CUDA_VISIBLE_DEVICES can be provided either "
            "via --devices or the environment."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default="0.0.0.0", help="Interface to bind (default: 0.0.0.0)")
    parser.add_argument(
        "--port",
        default="5515",
        help="TCP port for rpc-server (default: 5515)",
    )
    parser.add_argument(
        "--cache",
        default="",
        help="Tensor cache directory passed to rpc-server (-c).",
    )
    parser.add_argument(
        "--devices",
        default="",
        help="Set CUDA_VISIBLE_DEVICES before launching (e.g. 0 or 0,1). Leave blank to keep current value.",
    )
    parser.add_argument(
        "--extra",
        default="",
        help="Additional rpc-server flags appended verbatim (e.g. '--tensor-parallel 2').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without executing it.",
    )
    args = parser.parse_args(argv)

    ensure_binary()

    cmd: list[str] = [str(RPC_SERVER), "-p", args.port, "--host", args.host]
    if args.cache:
        cmd += ["-c", args.cache]
    if args.extra:
        cmd.extend(shlex.split(args.extra))

    env = os.environ.copy()
    if args.devices:
        env["CUDA_VISIBLE_DEVICES"] = args.devices

    print("Command:", shell_join(cmd))
    if args.devices:
        print(f"CUDA_VISIBLE_DEVICES={args.devices}")

    if args.dry_run:
        return 0

    try:
        return subprocess.call(cmd, env=env)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
