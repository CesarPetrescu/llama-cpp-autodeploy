"""Build manager endpoints (wraps autodevops.py)."""
from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..config import REPO_ROOT

router = APIRouter()

AUTODEVOPS_SCRIPT = REPO_ROOT / "autodevops.py"

# Cached parse of `python autodevops.py --help` so repeat calls are cheap.
_flag_cache: Optional[Dict[str, Any]] = None


async def _probe_supported_flags() -> Dict[str, Any]:
    """Run ``autodevops.py --help`` once and parse the argparser output.

    Returns a dict with ``bool_flags`` (set), ``choice_flags`` (dict of
    flag -> list of choices), ``value_flags`` (flags with a free-form value),
    ``options`` (parsed metadata for the UI), and ``raw_help`` for debugging.
    Unknown flags are treated as unsupported.
    """
    global _flag_cache
    if _flag_cache is not None:
        return _flag_cache

    if not AUTODEVOPS_SCRIPT.exists():
        _flag_cache = {
            "bool_flags": [],
            "choice_flags": {},
            "value_flags": [],
            "options": [],
            "usage": "",
            "summary": "",
            "raw_help": "",
        }
        return _flag_cache

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(AUTODEVOPS_SCRIPT), "--help",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(REPO_ROOT),
        )
        out_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)
        help_text = out_bytes.decode("utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover - defensive
        _flag_cache = {
            "bool_flags": [],
            "choice_flags": {},
            "value_flags": [],
            "options": [],
            "usage": "",
            "summary": "",
            "raw_help": f"(probe failed: {exc})",
        }
        return _flag_cache

    bool_flags: List[str] = []
    choice_flags: Dict[str, List[str]] = {}
    value_flags: List[str] = []
    usage = ""
    summary = ""
    options: List[Dict[str, Any]] = []

    lines = help_text.splitlines()
    usage = next((line.strip() for line in lines if line.strip().startswith("usage:")), "")
    preface: List[str] = []
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "options:":
            break
        preface.append(stripped)
    if preface:
        summary = " ".join(preface)

    current: Optional[Dict[str, Any]] = None
    for line in lines:
        option_match = re.match(r"^\s{2,}((?:-\w,\s*)?--[A-Za-z0-9\-]+(?:\s+(?:\{[^}]+\}|[A-Z_]+))?)\s{2,}(.*)$", line)
        if option_match:
            if current is not None:
                options.append(current)
            syntax = option_match.group(1).strip()
            description = option_match.group(2).strip()
            flag_match = re.search(r"(--[A-Za-z0-9\-]+)", syntax)
            if flag_match is None:
                current = None
                continue
            flag = flag_match.group(1)
            choice_match = re.search(r"\{([^}]+)\}", syntax)
            metavar_match = re.search(rf"{re.escape(flag)}\s+([A-Z_]+)", syntax)
            current = {
                "flag": flag,
                "syntax": syntax,
                "description": description,
                "choices": [c.strip() for c in choice_match.group(1).split(",")] if choice_match else [],
                "metavar": metavar_match.group(1) if metavar_match else None,
            }
            continue
        if current is not None and re.match(r"^\s{20,}\S", line):
            current["description"] = f'{current["description"]} {line.strip()}'.strip()
    if current is not None:
        options.append(current)

    # Bool flags: argparse rendering "  --fast-math" (no metavar).
    for match in re.finditer(r"(?m)^\s+(--[A-Za-z0-9\-]+)(?:$|,|\s{2,})", help_text):
        flag = match.group(1)
        tail = help_text[match.end() : match.end() + 200]
        if tail.lstrip().startswith(("{", "REF", "CHOICES", "CPU", "FAST", "FORCE", "BLAS")):
            continue
        if flag not in bool_flags:
            bool_flags.append(flag)

    # Choice flags: argparse renders "--force-mmq {auto,on,off}".
    for match in re.finditer(r"(--[A-Za-z0-9\-]+)\s+\{([^}]+)\}", help_text):
        flag = match.group(1)
        choices = [c.strip() for c in match.group(2).split(",") if c.strip()]
        choice_flags[flag] = choices
        if flag in bool_flags:
            bool_flags.remove(flag)

    # Flags that take a value without choices (e.g. --ref REF) — still valid,
    # not bool.
    for match in re.finditer(r"(--[A-Za-z0-9\-]+)\s+[A-Z_]+", help_text):
        flag = match.group(1)
        if flag in bool_flags:
            bool_flags.remove(flag)
        if flag not in value_flags:
            value_flags.append(flag)

    for option in options:
        if option["flag"] == "--help":
            option["kind"] = "meta"
            continue
        if option["flag"] in choice_flags:
            option["kind"] = "choice"
        elif option["flag"] in value_flags:
            option["kind"] = "value"
        else:
            option["kind"] = "bool"

    _flag_cache = {
        "bool_flags": bool_flags,
        "choice_flags": choice_flags,
        "value_flags": [flag for flag in value_flags if flag != "--help"],
        "options": [option for option in options if option["flag"] != "--help"],
        "usage": usage,
        "summary": summary,
        "raw_help": help_text,
    }
    return _flag_cache


def _clear_flag_cache() -> None:
    """Test hook: force the next /supported-flags call to re-probe."""
    global _flag_cache
    _flag_cache = None


class BuildRequest(BaseModel):
    ref: str = "latest"
    now: bool = True
    fast_math: bool = False
    force_mmq: str = "auto"  # auto | on | off
    blas: str = "auto"  # auto | mkl | openblas | off
    distributed: bool = False
    cpu_only: bool = False


async def _validate_request(req: BuildRequest) -> None:
    spec = await _probe_supported_flags()
    choice_flags = spec["choice_flags"]
    bool_flags = set(spec["bool_flags"])

    def _require_choice(flag: str, value: str) -> None:
        choices = choice_flags.get(flag)
        if choices is None:
            return  # the flag probably just takes a free-form value
        if value not in choices:
            raise HTTPException(
                status_code=400,
                detail=f"{flag}={value!r} is not supported by autodevops.py. Allowed: {choices}",
            )

    def _require_bool(flag: str, value: bool) -> None:
        if not value:
            return
        if flag not in bool_flags and flag not in choice_flags:
            raise HTTPException(
                status_code=400,
                detail=f"{flag} is not supported by this autodevops.py. Remove it or update the script.",
            )

    _require_choice("--force-mmq", req.force_mmq)
    _require_choice("--blas", req.blas)
    _require_bool("--fast-math", req.fast_math)
    _require_bool("--distributed", req.distributed)
    _require_bool("--cpu-only", req.cpu_only)
    _require_bool("--now", req.now)


@router.get("")
async def list_builds(request: Request) -> Dict[str, Any]:
    manager = request.app.state.manager
    return {"builds": manager.list_builds()}


@router.get("/supported-flags")
async def supported_flags() -> Dict[str, Any]:
    spec = await _probe_supported_flags()
    return {
        "bool_flags": spec["bool_flags"],
        "choice_flags": spec["choice_flags"],
        "value_flags": spec["value_flags"],
        "options": spec["options"],
        "usage": spec["usage"],
        "summary": spec["summary"],
    }


@router.post("")
async def start_build(payload: BuildRequest, request: Request) -> Dict[str, Any]:
    await _validate_request(payload)
    manager = request.app.state.manager
    record = await manager.start_build(payload.model_dump())
    return {"build": manager.get_build(record.id)}


@router.get("/{build_id}")
async def get_build(build_id: str, request: Request) -> Dict[str, Any]:
    manager = request.app.state.manager
    data = manager.get_build(build_id)
    if data is None:
        raise HTTPException(status_code=404, detail="build not found")
    logs = manager.get_build_logs(build_id, tail=500)
    return {"build": data, "logs": logs}


@router.post("/{build_id}/stop")
async def stop_build(build_id: str, request: Request) -> Dict[str, Any]:
    manager = request.app.state.manager
    await manager.stop_build(build_id)
    data = manager.get_build(build_id)
    if data is None:
        raise HTTPException(status_code=404, detail="build not found")
    return {"build": data}
