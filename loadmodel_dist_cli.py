#!/usr/bin/env python3
"""Interactive CLI for launching llama.cpp distributed (RPC) inference."""
from __future__ import annotations

import curses
import ipaddress
import json
import os
import shlex
import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import socket
from concurrent.futures import ThreadPoolExecutor

from collections import deque
import tui_utils

SCRIPT_DIR = Path(__file__).resolve().parent
BIN_DIR = SCRIPT_DIR / "bin"
LLAMA_CLI = BIN_DIR / "llama-cli"
RPC_SERVER = BIN_DIR / "rpc-server"
MODELS_DIR = SCRIPT_DIR / "models"
DEFAULT_RPC_PORT = "5515"
SCAN_TIMEOUT = 0.25
CONFIG_PATH = SCRIPT_DIR / ".loadmodel_dist_cli.json"
LOG_HISTORY = 200
SPINNER_FRAMES = "|/-\\"

STRINGS = {
    "title": "llama.cpp Distributed Launcher",
    "instructions": "Arrows: navigate • Enter: edit/activate • PgUp/PgDn: scroll • Tab: cycle panes • ?: toggle help • q: quit",
    "logs_heading": "Logs",
    "help_heading": "Help",
    "no_space_warning": "Terminal too small for the launcher. Please enlarge.",
    "scan_start": "Scanning local subnets for rpc-server instances…",
    "scan_complete": "Network scan complete.",
    "no_options": "No options available.",
}


def append_log(ui_state: dict, message: str) -> None:
    if not message:
        return
    logs = ui_state.setdefault("logs", deque(maxlen=LOG_HISTORY))
    timestamp = time.strftime("%H:%M:%S")
    logs.append(f"[{timestamp}] {message}")
    ui_state["status_message"] = message
    ui_state.setdefault("logs_offset", 0)
    if ui_state.get("focus_area", 0) != 2:
        ui_state["logs_offset"] = 0


def load_saved_state() -> dict:
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, dict):
                return data
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    return {}


def save_state(options: List["OptionBase"], state: dict) -> None:
    payload = {}
    for opt in options:
        try:
            payload[opt.key] = opt.get_value(state)
        except Exception:
            continue
    try:
        CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def apply_saved_values(options: List["OptionBase"], state: dict, saved: dict) -> None:
    if not saved:
        return
    for opt in options:
        if opt.key in saved:
            try:
                opt.set_value(state, saved[opt.key])
            except Exception:
                continue


def draw_logs(
    stdscr: "curses._CursesWindow",
    ui_state: dict,
    start_row: int,
    width: int,
    height: int,
) -> int:
    logs = list(ui_state.get("logs", []))
    if not logs or start_row >= height:
        ui_state["logs_max_offset"] = 0
        ui_state["logs_visible_rows"] = 0
        return 0
    focus = ui_state.get("focus_area", 0) == 2
    available_rows = max(0, height - start_row)
    if available_rows <= 0:
        ui_state["logs_max_offset"] = 0
        ui_state["logs_visible_rows"] = 0
        return 0

    heading_attr = curses.A_BOLD | (curses.A_REVERSE if focus else curses.A_DIM)
    stdscr.addnstr(
        start_row,
        2,
        STRINGS["logs_heading"][: max(1, width - 4)],
        max(1, width - 4),
        heading_attr,
    )
    if available_rows == 1:
        ui_state["logs_max_offset"] = 0
        ui_state["logs_visible_rows"] = 0
        return 1

    body_rows = available_rows - 1
    ui_state.setdefault("logs_offset", 0)
    max_start = max(0, len(logs) - body_rows)
    offset = min(max(0, ui_state["logs_offset"]), max_start)
    first_idx = max_start - offset
    first_idx = max(0, first_idx)
    slice_logs = logs[first_idx : first_idx + body_rows]
    ui_state["logs_offset"] = offset
    ui_state["logs_max_offset"] = max_start
    ui_state["logs_visible_rows"] = len(slice_logs)

    for idx, line in enumerate(slice_logs, start=1):
        attr = curses.A_DIM | (curses.A_REVERSE if focus else 0)
        stdscr.addnstr(
            start_row + idx,
            2,
            line[: max(1, width - 4)],
            max(1, width - 4),
            attr,
        )
    return 1 + len(slice_logs)


def _option_detail_lines(opt: "OptionBase", state: dict) -> List[tuple[str, int]]:
    lines: List[tuple[str, int]] = []
    description = getattr(opt, "description", "")
    if description:
        lines.append((description, curses.A_NORMAL))
    if isinstance(opt, ChoiceOption):
        current = opt.current(state)
        lines.append(("", 0))
        lines.append((f"Current: {current.label}", curses.A_DIM))
        lines.append(("Left/Right: change selection", curses.A_DIM))
        if not current.enabled and current.reason:
            lines.append((f"Note: {current.reason}", curses.A_DIM))
    elif isinstance(opt, InputOption):
        value = str(state.get(opt.key, "")).strip()
        lines.append(("", 0))
        lines.append((f"Current: {value or '(blank)'}", curses.A_DIM))
        lines.append(("Enter: edit (Enter saves, Esc cancels)", curses.A_DIM))
    elif isinstance(opt, ToggleOption):
        enabled = bool(state.get(opt.key))
        lines.append(("", 0))
        lines.append((f"State: {'enabled' if enabled else 'disabled'}", curses.A_DIM))
        lines.append(("Space/Enter: toggle", curses.A_DIM))
    elif isinstance(opt, ActionOption):
        lines.append(("", 0))
        lines.append(("Enter: run this action", curses.A_DIM))
    if not lines:
        lines.append(("No details available.", curses.A_DIM))
    return lines


def prepare_help_panel(
    selected_opt: Optional["OptionBase"],
    state: dict,
    wrap_width: int,
    available_rows: int,
    ui_state: dict,
    height: int,
) -> tuple[str, List[tuple[str, int]], bool]:
    if selected_opt is None:
        ui_state["help_max_offset"] = 0
        ui_state["help_visible_lines"] = 0
        ui_state["help_total_lines"] = 0
        ui_state.setdefault("help_offset", 0)
        return ("Selected: (none)", [], False)

    opt_name = getattr(selected_opt, "name", None)
    if not opt_name:
        opt_name = selected_opt.__class__.__name__.replace("Option", "")
        opt_name = opt_name.replace("_", " ").title()
    header_text = f"Selected: {opt_name}"

    detail_pairs = _option_detail_lines(selected_opt, state)
    full_help_lines: List[tuple[str, int]] = []
    for raw, attr in detail_pairs:
        style = attr or curses.A_NORMAL
        if raw == "":
            full_help_lines.append(("", style))
            continue
        wrapped = textwrap.wrap(raw, wrap_width) or [""]
        for line in wrapped:
            full_help_lines.append((line, style))

    max_preview = max(5, min(10, height // 2))
    show_full_help = ui_state.get("show_full_help", False)
    help_source = full_help_lines if show_full_help else full_help_lines[:max_preview]

    ui_state["help_total_lines"] = len(help_source)
    ui_state.setdefault("help_offset", 0)
    available_rows = max(0, available_rows)
    max_offset = max(0, len(help_source) - available_rows)
    help_offset = min(max(0, ui_state.get("help_offset", 0)), max_offset)
    ui_state["help_offset"] = help_offset
    ui_state["help_max_offset"] = max_offset
    visible_help = help_source[help_offset : help_offset + available_rows]
    ui_state["help_visible_lines"] = len(visible_help)

    indicator = ""
    if help_offset > 0:
        indicator += "↑"
    if help_offset < max_offset:
        indicator += "↓"
    if not show_full_help and len(help_source) < len(full_help_lines):
        indicator += "+"
    if indicator:
        header_text = f"{header_text} ({indicator})"

    return (header_text, visible_help, True)


@dataclass
class ChoiceItem:
    value: str
    label: str
    enabled: bool = True
    reason: str | None = None


class OptionBase:
    def __init__(
        self,
        key: str,
        name: str,
        description: str,
        *,
        visible: Callable[[dict], bool] | None = None,
    ) -> None:
        self.key = key
        self.name = name
        self.description = description
        self._visible = visible
        self.icon = "[OPT]"
        self.default_value = None

    def is_visible(self, state: dict) -> bool:
        if self._visible is None:
            return True
        try:
            return bool(self._visible(state))
        except Exception:
            return True

    def render(
        self,
        win: "curses._CursesWindow",
        y: int,
        width: int,
        selected: bool,
        state: dict,
    ) -> int:
        raise NotImplementedError

    def handle_key(
        self,
        key: int,
        state: dict,
        stdscr: "curses._CursesWindow",
    ) -> Tuple[bool, Optional[str]]:
        return False, None

    def height(self, width: int, state: dict) -> int:
        raise NotImplementedError

    def get_value(self, state: dict):
        return state.get(self.key)

    def set_value(self, state: dict, value) -> None:
        state[self.key] = value

    def is_modified(self, state: dict) -> bool:
        return state.get(self.key) != self.default_value

    def get_summary(self, width: int) -> str:
        text = (self.description or "").strip()
        if not text:
            return ""
        summary = text.splitlines()[0].strip()
        if len(summary) > width:
            summary = summary[: max(0, width - 1)] + "…"
        return summary


class InputOption(OptionBase):
    def __init__(
        self,
        key: str,
        name: str,
        description: str,
        *,
        placeholder: str = "",
        on_change: Callable[[dict, str], None] | None = None,
        visible: Callable[[dict], bool] | None = None,
    ) -> None:
        super().__init__(key, name, description, visible=visible)
        self.placeholder = placeholder
        self._on_change = on_change
        self.icon = "[TXT]"
        self.default_value = None

    def _edit(self, stdscr: "curses._CursesWindow", state: dict) -> None:
        current = str(state.get(self.key, "")).strip()
        result = tui_utils.edit_line_dialog(
            stdscr,
            title=f"Edit {self.name}",
            initial=current,
            allow_empty=True,
        )
        if not result.accepted:
            return
        if result.value != current:
            state[self.key] = result.value
            if self._on_change is not None:
                self._on_change(state, result.value)

    def handle_key(
        self,
        key: int,
        state: dict,
        stdscr: "curses._CursesWindow",
    ) -> Tuple[bool, Optional[str]]:
        if key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
            self._edit(stdscr, state)
        return False, None

    def render(
        self,
        win: "curses._CursesWindow",
        y: int,
        width: int,
        selected: bool,
        state: dict,
    ) -> int:
        attr = curses.A_REVERSE if selected else curses.A_NORMAL
        if self.default_value is None:
            self.default_value = str(state.get(self.key, "") or "")
            state.setdefault(self.key, self.default_value)
        value = str(state.get(self.key, "")).strip()
        display = value or self.placeholder
        marker = "*" if self.is_modified(state) else " "
        label = f"{marker}{self.icon} {self.name}: {display}"
        win.addnstr(y, 2, label, max(10, width - 4), attr)
        summary = self.get_summary(max(0, width - len(label) - 6))
        if summary:
            win.addnstr(y, min(width - 2, 2 + len(label) + 1), f" · {summary}", max(10, width - len(label) - 4), curses.A_DIM)
        line_count = 1
        wrap_width = max(10, width - 6)
        for line in textwrap.wrap(self.description, wrap_width):
            win.addnstr(y + line_count, 6, line, max(10, width - 8), curses.A_DIM)
            line_count += 1
        return line_count

    def height(self, width: int, state: dict) -> int:
        wrap_width = max(10, width - 6)
        return 1 + len(textwrap.wrap(self.description, wrap_width))

    def get_value(self, state: dict):
        return state.get(self.key, "")

    def set_value(self, state: dict, value) -> None:
        state[self.key] = "" if value is None else str(value)

    def is_modified(self, state: dict) -> bool:
        baseline = "" if self.default_value is None else str(self.default_value)
        return str(state.get(self.key, "")) != baseline


class ToggleOption(OptionBase):
    def __init__(
        self,
        key: str,
        name: str,
        description: str,
        *,
        visible: Callable[[dict], bool] | None = None,
    ) -> None:
        super().__init__(key, name, description, visible=visible)
        self.icon = "[TGL]"
        self.default_value = None

    def render(
        self,
        win: "curses._CursesWindow",
        y: int,
        width: int,
        selected: bool,
        state: dict,
    ) -> int:
        attr = curses.A_REVERSE if selected else curses.A_NORMAL
        if self.default_value is None:
            self.default_value = bool(state.get(self.key, False))
            state.setdefault(self.key, self.default_value)
        mark = "✔" if state.get(self.key) else "✖"
        marker = "*" if self.is_modified(state) else " "
        label = f"{marker}{self.icon} [{mark}] {self.name}"
        win.addnstr(y, 2, label, max(10, width - 4), attr)
        summary = self.get_summary(max(0, width - len(label) - 6))
        if summary:
            win.addnstr(y, min(width - 2, 2 + len(label) + 1), f" · {summary}", max(10, width - len(label) - 4), curses.A_DIM)
        line_count = 1
        wrap_width = max(10, width - 6)
        for line in textwrap.wrap(self.description, wrap_width):
            win.addnstr(y + line_count, 6, line, max(10, width - 8), curses.A_DIM)
            line_count += 1
        return line_count

    def height(self, width: int, state: dict) -> int:
        wrap_width = max(10, width - 6)
        return 1 + len(textwrap.wrap(self.description, wrap_width))

    def handle_key(
        self,
        key: int,
        state: dict,
        stdscr: "curses._CursesWindow",
    ) -> Tuple[bool, Optional[str]]:
        if key in (curses.KEY_ENTER, ord("\n"), ord("\r"), ord(" "), ord("t")):
            state[self.key] = not bool(state.get(self.key))
        return False, None

    def get_value(self, state: dict):
        return bool(state.get(self.key))

    def set_value(self, state: dict, value) -> None:
        state[self.key] = bool(value)

    def is_modified(self, state: dict) -> bool:
        baseline = bool(self.default_value) if self.default_value is not None else False
        return bool(state.get(self.key)) != baseline


class ChoiceOption(OptionBase):
    def __init__(
        self,
        key: str,
        name: str,
        description: str,
        *,
        state: dict,
        choices: Iterable[ChoiceItem] | None = None,
        choices_fn: Callable[[dict], Iterable[ChoiceItem]] | None = None,
        on_change: Callable[[dict, ChoiceItem], None] | None = None,
        visible: Callable[[dict], bool] | None = None,
    ) -> None:
        super().__init__(key, name, description, visible=visible)
        if choices is None and choices_fn is None:
            raise ValueError("choices or choices_fn required")
        self._static_choices = list(choices or [])
        self._choices_fn = choices_fn
        self._on_change = on_change
        self._choices: List[ChoiceItem] = []
        self._index = 0
        self.icon = "[SEL]"
        self.default_value = None

    def _compute_choices(self, state: dict) -> None:
        if self._choices_fn is not None:
            items = list(self._choices_fn(state))
        else:
            items = list(self._static_choices)
        if not items:
            items = [ChoiceItem("", "(no options)", enabled=False, reason="No models discovered")]
        self._choices = items
        current_val = state.get(self.key)
        original_val = current_val
        found = False
        for idx, item in enumerate(self._choices):
            if item.value == current_val:
                self._index = idx
                found = True
                break
        if not found:
            for idx, item in enumerate(self._choices):
                if item.enabled:
                    self._index = idx
                    state[self.key] = item.value
                    if self._on_change is not None and item.value != original_val:
                        self._on_change(state, item)
                    found = True
                    break
        if not found:
            self._index = 0
            state[self.key] = self._choices[0].value

    def current(self, state: dict) -> ChoiceItem:
        self._compute_choices(state)
        return self._choices[self._index]

    def handle_key(
        self,
        key: int,
        state: dict,
        stdscr: "curses._CursesWindow",
    ) -> Tuple[bool, Optional[str]]:
        self._compute_choices(state)
        count = len(self._choices)
        if count == 0:
            return False, None
        if key in (curses.KEY_LEFT, ord("h")):
            for _ in range(count):
                self._index = (self._index - 1) % count
                if self._choices[self._index].enabled:
                    break
        elif key in (curses.KEY_RIGHT, ord("l"), ord(" ")):
            for _ in range(count):
                self._index = (self._index + 1) % count
                if self._choices[self._index].enabled:
                    break
        state[self.key] = self._choices[self._index].value
        if self._on_change is not None:
            self._on_change(state, self._choices[self._index])
        return False, None

    def render(
        self,
        win: "curses._CursesWindow",
        y: int,
        width: int,
        selected: bool,
        state: dict,
    ) -> int:
        self._compute_choices(state)
        if self.default_value is None:
            self.default_value = state.get(self.key, self._choices[self._index].value)
        attr = curses.A_REVERSE if selected else curses.A_NORMAL
        current = self._choices[self._index]
        marker = "*" if self.is_modified(state) else " "
        label = f"{marker}{self.icon} {self.name}: {current.label}"
        win.addnstr(y, 2, label, max(10, width - 4), attr)
        summary = self.get_summary(max(0, width - len(label) - 6))
        if summary:
            win.addnstr(y, min(width - 2, 2 + len(label) + 1), f" · {summary}", max(10, width - len(label) - 4), curses.A_DIM)
        line_count = 1
        wrap_width = max(10, width - 6)
        for line in textwrap.wrap(self.description, wrap_width):
            win.addnstr(y + line_count, 6, line, max(10, width - 8), curses.A_DIM)
            line_count += 1
        if not current.enabled and current.reason:
            win.addnstr(
                y + line_count,
                6,
                f"⚠ {current.reason}",
                max(10, width - 8),
                curses.color_pair(2) | curses.A_DIM,
            )
            line_count += 1
        return line_count

    def height(self, width: int, state: dict) -> int:
        self._compute_choices(state)
        wrap_width = max(10, width - 6)
        base = 1 + len(textwrap.wrap(self.description, wrap_width))
        current = self._choices[self._index]
        if not current.enabled and current.reason:
            base += 1
        return base

    def get_value(self, state: dict):
        self._compute_choices(state)
        return state.get(self.key, self._choices[self._index].value)

    def set_value(self, state: dict, value) -> None:
        self._compute_choices(state)
        for idx, item in enumerate(self._choices):
            if item.value == value:
                self._index = idx
                state[self.key] = item.value
                return
        if self._choices:
            state[self.key] = self._choices[self._index].value

    def is_modified(self, state: dict) -> bool:
        if self.default_value is None:
            return False
        return state.get(self.key) != self.default_value


class ActionOption(OptionBase):
    def __init__(
        self,
        name: str,
        description: str,
        action: Callable[[dict], Tuple[bool, Optional[str]]],
        *,
        visible: Callable[[dict], bool] | None = None,
    ) -> None:
        super().__init__(f"action:{name}", name, description, visible=visible)
        self._action = action
        self.icon = "[ACT]"

    def render(
        self,
        win: "curses._CursesWindow",
        y: int,
        width: int,
        selected: bool,
        state: dict,
    ) -> int:
        attr = curses.A_REVERSE if selected else curses.A_BOLD
        label = f"{self.icon} {self.name}"
        win.addnstr(y, 2, label, max(10, width - 4), attr)
        line_count = 1
        wrap_width = max(10, width - 6)
        for line in textwrap.wrap(self.description, wrap_width):
            win.addnstr(y + line_count, 6, line, max(10, width - 8), curses.A_DIM)
            line_count += 1
        return line_count

    def height(self, width: int, state: dict) -> int:
        wrap_width = max(10, width - 6)
        return 1 + len(textwrap.wrap(self.description, wrap_width))

    def handle_key(
        self,
        key: int,
        state: dict,
        stdscr: "curses._CursesWindow",
    ) -> Tuple[bool, Optional[str]]:
        if key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
            return self._action(state)
        return False, None

    def get_value(self, state: dict):
        return None

    def set_value(self, state: dict, value) -> None:
        return

    def is_modified(self, state: dict) -> bool:
        return False


def shell_join(parts: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in parts)


def list_local_gguf(models_dir: Path) -> List[str]:
    if not models_dir.exists() or not models_dir.is_dir():
        return []
    files = []
    for path in models_dir.rglob("*.gguf"):
        try:
            rel = path.relative_to(models_dir)
        except ValueError:
            rel = path.name
        files.append(str(rel))
    return sorted(files)


def refresh_local_models(state: dict) -> Tuple[bool, Optional[str]]:
    base = Path(state.get("models_dir") or str(MODELS_DIR)).expanduser()
    try:
        files = list_local_gguf(base)
    except Exception as exc:
        state["local_models"] = []
        return False, f"Failed to scan models: {exc}"
    state["local_models"] = files
    if not files:
        return False, "No GGUF models found in the selected directory."
    return False, f"Found {len(files)} GGUF model(s)."


def build_local_choices(state: dict) -> List[ChoiceItem]:
    models = state.get("local_models")
    if models is None:
        refresh_local_models(state)
        models = state.get("local_models") or []
    if not models:
        return [ChoiceItem("", "No GGUF files discovered", enabled=False)]
    return [ChoiceItem(m, m) for m in models]


def on_models_dir_change(state: dict, new_value: str) -> None:
    state["models_dir"] = new_value or str(MODELS_DIR)
    _, message = refresh_local_models(state)
    state["status"] = message or ""


def set_model_from_choice(state: dict, choice: ChoiceItem) -> None:
    if not choice.value:
        return
    base = Path(state.get("models_dir") or str(MODELS_DIR)).expanduser()
    candidate = (base / choice.value).expanduser()
    state["model_path"] = str(candidate)
    state["selected_local_model"] = choice.value


def ensure_binary(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build llama.cpp with RPC support using autodevops (Enable distributed RPC backend)."
        )


def parse_hosts(hosts: str) -> str:
    entries = [h.strip() for h in hosts.split(",") if h.strip()]
    if not entries:
        raise ValueError("At least one RPC host must be provided (format: host:port).")
    return ",".join(entries)


def build_llama_command(state: dict) -> List[str]:
    ensure_binary(LLAMA_CLI)
    model = state.get("model_path", "").strip()
    if not model:
        raise ValueError("Model path is required.")
    host_list = parse_hosts(state.get("rpc_hosts", ""))

    cmd: List[str] = [str(LLAMA_CLI), "-m", model, "--rpc", host_list]

    ctx = state.get("ctx_size", "").strip()
    if ctx:
        cmd += ["--ctx-size", ctx]
    ngl = state.get("n_gpu_layers", "").strip()
    if ngl:
        cmd += ["-ngl", ngl]
    batch = state.get("batch_size", "").strip()
    if batch:
        cmd += ["-b", batch]
    cache_dir = state.get("tensor_cache", "").strip()
    if cache_dir:
        cmd += ["-c", cache_dir]
    extra = state.get("extra_flags", "").strip()
    if extra:
        cmd.extend(shlex.split(extra))
    prompt_file = state.get("prompt_file", "").strip()
    prompt_text = state.get("prompt_text", "").strip()
    if prompt_file:
        cmd += ["-f", prompt_file]
    elif prompt_text:
        cmd += ["-p", prompt_text]
    return cmd


def build_rpc_command(state: dict) -> List[str]:
    ensure_binary(RPC_SERVER)
    port = state.get("worker_port", DEFAULT_RPC_PORT).strip() or DEFAULT_RPC_PORT
    host = state.get("worker_host", "0.0.0.0").strip() or "0.0.0.0"
    cmd: List[str] = [str(RPC_SERVER), "-p", port, "--host", host]
    cache = state.get("worker_cache", "").strip()
    if cache:
        cmd += ["-c", cache]
    return cmd


def start_worker_process(state: dict) -> subprocess.Popen:
    cmd = build_rpc_command(state)
    env = os.environ.copy()
    device = state.get("worker_devices", "").strip()
    if device:
        env["CUDA_VISIBLE_DEVICES"] = device
    try:
        proc = subprocess.Popen(cmd, env=env)
    except FileNotFoundError as exc:
        raise FileNotFoundError(str(exc)) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to start rpc-server: {exc}") from exc
    return proc


def merge_hosts(state: dict, hosts: Iterable[str]) -> None:
    new_hosts = {h.strip() for h in hosts if h.strip()}
    if not new_hosts:
        return
    existing = {h.strip() for h in state.get("rpc_hosts", "").split(",") if h.strip()}
    combined = sorted(existing | new_hosts)
    state["rpc_hosts"] = ",".join(combined)
    discovered = {h.strip() for h in state.get("discovered_hosts", []) if h.strip()}
    state["discovered_hosts"] = sorted(discovered | new_hosts)


def _list_private_networks() -> Tuple[List[ipaddress.IPv4Network], List[str]]:
    networks: List[ipaddress.IPv4Network] = []
    local_ips: List[str] = []
    try:
        output = subprocess.check_output(
            ["ip", "-o", "-f", "inet", "addr", "show"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return networks, local_ips
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        cidr = parts[3]
        try:
            iface = ipaddress.ip_interface(cidr)
        except ValueError:
            continue
        ip = iface.ip
        if ip.is_loopback or not ip.is_private:
            continue
        network = iface.network
        if network.num_addresses > 256:
            network = ipaddress.ip_network(f"{ip}/24", strict=False)
        networks.append(network)
        local_ips.append(str(ip))
    # dedupe networks
    uniq = []
    seen = set()
    for net in networks:
        if net.network_address not in seen:
            uniq.append(net)
            seen.add(net.network_address)
    return uniq, local_ips


def discover_network_workers(port: str, timeout: float = SCAN_TIMEOUT) -> List[str]:
    try:
        port_int = int(port)
    except ValueError:
        return []
    networks, local_ips = _list_private_networks()
    if not networks:
        return []
    local_set = set(local_ips)
    discovered: List[str] = []
    lock = threading.Lock()

    def probe(ip: ipaddress.IPv4Address) -> None:
        addr = str(ip)
        if addr in local_set:
            return
        try:
            with socket.create_connection((addr, port_int), timeout=timeout):
                with lock:
                    discovered.append(f"{addr}:{port}")
        except (socket.timeout, OSError):
            pass

    with ThreadPoolExecutor(max_workers=128) as executor:
        for network in networks:
            for host in network.hosts():
                executor.submit(probe, host)

    return sorted(set(discovered))


def action_launch(state: dict) -> Tuple[bool, Optional[str]]:
    messages: List[str] = []
    if state.get("local_worker"):
        proc = state.get("worker_process")
        if not proc or proc.poll() is not None:
            try:
                proc = start_worker_process(state)
                state["worker_process"] = proc
                messages.append(f"Started local rpc-server (pid {proc.pid}).")
            except (FileNotFoundError, RuntimeError) as exc:
                return False, str(exc)
        port = state.get("worker_port", DEFAULT_RPC_PORT).strip() or DEFAULT_RPC_PORT
        host = state.get("worker_host", "0.0.0.0").strip() or "0.0.0.0"
        if host in ("0.0.0.0", "::"):
            merge_hosts(state, [f"127.0.0.1:{port}"])
        else:
            merge_hosts(state, [f"{host}:{port}"])
    try:
        cmd = build_llama_command(state)
    except (FileNotFoundError, ValueError) as exc:
        return False, str(exc)

    curses.def_prog_mode()
    curses.endwin()
    if messages:
        for msg in messages:
            print(msg, flush=True)
    print("Launching distributed llama-cli:\n  " + shell_join(cmd), flush=True)
    try:
        exit_code = subprocess.call(cmd)
    except KeyboardInterrupt:
        exit_code = 130
    finally:
        curses.reset_prog_mode()
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        curses.doupdate()
    msg = "llama-cli exited" if exit_code == 0 else f"llama-cli exited with code {exit_code}"
    return False, msg


def action_start_worker(state: dict) -> Tuple[bool, Optional[str]]:
    if not state.get("local_worker", False):
        return False, "Enable 'Launch local rpc-server' toggle first."
    proc = state.get("worker_process")
    if proc and proc.poll() is None:
        return False, f"rpc-server already running (pid {proc.pid})."
    try:
        proc = start_worker_process(state)
    except (FileNotFoundError, RuntimeError) as exc:
        return False, str(exc)
    state["worker_process"] = proc
    port = state.get("worker_port", DEFAULT_RPC_PORT).strip() or DEFAULT_RPC_PORT
    host = state.get("worker_host", "0.0.0.0").strip() or "0.0.0.0"
    if host in ("0.0.0.0", "::"):
        merge_hosts(state, [f"127.0.0.1:{port}"])
    else:
        merge_hosts(state, [f"{host}:{port}"])
    return False, f"rpc-server started (pid {proc.pid})."


def action_refresh_models(state: dict) -> Tuple[bool, Optional[str]]:
    return refresh_local_models(state)


def action_scan_network(state: dict) -> Tuple[bool, Optional[str]]:
    port = state.get("worker_port", DEFAULT_RPC_PORT).strip() or DEFAULT_RPC_PORT
    hosts = discover_network_workers(port)
    if hosts:
        merge_hosts(state, hosts)
        return False, f"Discovered {len(hosts)} worker(s) on port {port}."
    return False, f"No rpc-server instances found on port {port}."


def build_options(state: dict) -> List[OptionBase]:
    return [
        InputOption(
            "models_dir",
            "Models directory",
            "Directory containing local GGUF files.",
            placeholder=str(MODELS_DIR),
            on_change=on_models_dir_change,
        ),
        ChoiceOption(
            "selected_local_model",
            "Local GGUF",
            "Select a GGUF to populate the model path automatically.",
            state=state,
            choices_fn=build_local_choices,
            on_change=set_model_from_choice,
        ),
        ActionOption(
            "Refresh local models",
            "Rescan the models directory for GGUF files.",
            action_refresh_models,
        ),
        ActionOption(
            "Scan network for rpc-server",
            "Probe private subnets for rpc-server instances listening on the configured port.",
            action_scan_network,
        ),
        InputOption(
            "model_path",
            "Model GGUF path",
            "Absolute path to the GGUF model to serve. Auto-filled when selecting from the list above.",
            placeholder=str((MODELS_DIR / "model.gguf").resolve()),
        ),
        InputOption(
            "rpc_hosts",
            "Worker hosts",
            "Comma-separated list of host:port entries for rpc-server workers.",
            placeholder=f"192.168.1.10:{DEFAULT_RPC_PORT},192.168.1.11:{DEFAULT_RPC_PORT}",
        ),
        InputOption(
            "ctx_size",
            "Context window",
            "Context window in tokens (--ctx-size). Leave blank to use model defaults.",
            placeholder="4096",
        ),
        InputOption(
            "n_gpu_layers",
            "GPU layers",
            "Number of layers to keep on GPU (-ngl). Use 99 to offload as much as possible.",
            placeholder="99",
        ),
        InputOption(
            "batch_size",
            "Batch size",
            "Batch size (-b) for distributed inference.",
            placeholder="512",
        ),
        InputOption(
            "tensor_cache",
            "Tensor cache dir",
            "Optional cache directory (-c) shared across runs.",
            placeholder="/tmp/llama-cache",
        ),
        InputOption(
            "prompt_file",
            "Prompt file",
            "Path to a prompt file (-f). Leave blank to provide a prompt string instead.",
            placeholder="",
        ),
        InputOption(
            "prompt_text",
            "Prompt text",
            "Inline prompt (-p) used when no prompt file is supplied.",
            placeholder="Hello from llama.cpp!",
        ),
        InputOption(
            "extra_flags",
            "Extra llama-cli flags",
            "Additional arguments appended verbatim (e.g. --temp 0.7 --top-p 0.9).",
            placeholder="",
        ),
        ToggleOption(
            "local_worker",
            "Launch local rpc-server",
            "Start a rpc-server locally using the parameters below before running llama-cli.",
        ),
        InputOption(
            "worker_host",
            "rpc-server host",
            "Interface for the local rpc-server to bind.",
            placeholder="0.0.0.0",
            visible=lambda st: st.get("local_worker"),
        ),
        InputOption(
            "worker_port",
            "rpc-server port",
            "Listening port for the local rpc-server.",
            placeholder=DEFAULT_RPC_PORT,
            visible=lambda st: st.get("local_worker"),
        ),
        InputOption(
            "worker_cache",
            "rpc-server cache",
            "Optional tensor cache directory passed to rpc-server (-c).",
            placeholder="/tmp/llama-cache",
            visible=lambda st: st.get("local_worker"),
        ),
        InputOption(
            "worker_devices",
            "CUDA_VISIBLE_DEVICES",
            "Restrict GPUs visible to the local rpc-server (e.g. 0 or 0,1). Leave blank for all.",
            placeholder="",
            visible=lambda st: st.get("local_worker"),
        ),
        ActionOption(
            "Start local rpc-server",
            "Launch rpc-server locally in the background using the parameters above.",
            action_start_worker,
            visible=lambda st: st.get("local_worker"),
        ),
        ActionOption(
            "Launch distributed llama-cli",
            "Start llama-cli with the configured RPC workers.",
            action_launch,
        ),
    ]


def _draw_wide_layout(
    stdscr: "curses._CursesWindow",
    options: List[OptionBase],
    option_info: List[tuple[int, OptionBase, int]],
    selected_idx: int,
    scroll: int,
    state: dict,
    ui_state: dict,
    status: str,
    height: int,
    width: int,
) -> Tuple[Optional[int], Optional[int]]:
    body_top = 2
    status_y = max(0, height - 2)
    instructions_y = max(0, height - 1)
    body_height = max(0, status_y - body_top)
    left_margin = 2
    column_gap = 2
    min_left_width = 26
    min_right_width = 32

    left_width = max(min_left_width, int(width * 0.55))
    max_left = width - (left_margin + column_gap + min_right_width + 2)
    if left_width > max_left:
        left_width = max(min_left_width, max_left)
    right_start = left_margin + left_width + column_gap
    right_width = width - right_start - 2
    if right_width < min_right_width:
        right_width = max(10, min_right_width)
        left_width = max(min_left_width, width - (left_margin + column_gap + right_width + 2))
        right_start = left_margin + left_width + column_gap

    if body_height <= 0 or left_width <= 0 or right_width <= 0:
        warning = "Not enough space to render the launcher. Enlarge the window."
        stdscr.addnstr(body_top, left_margin, warning[: max(1, width - 4)], max(1, width - 4), curses.A_BOLD | curses.color_pair(2))
        return None, None

    total_visible = len(option_info)
    start = max(0, min(scroll, max(0, total_visible - 1)))
    y = body_top
    first_rendered: Optional[int] = None
    last_rendered: Optional[int] = None
    first_render_row: Optional[int] = None
    last_render_row: Optional[int] = None

    for visible_idx in range(start, total_visible):
        opt_index, opt, _ = option_info[visible_idx]
        opt_height = max(1, opt.height(left_width, state))
        if y >= status_y:
            break
        if opt_height > body_height and first_rendered is not None:
            break
        render_start = y
        if y + opt_height > status_y:
            if first_rendered is None:
                extra = opt.render(stdscr, y, left_width, opt_index == selected_idx, state)
                first_render_row = render_start
                y += extra
                first_rendered = visible_idx
                last_rendered = visible_idx
                last_render_row = y - 1
            break
        extra = opt.render(stdscr, y, left_width, opt_index == selected_idx, state)
        if first_render_row is None:
            first_render_row = render_start
        y += extra
        if first_rendered is None:
            first_rendered = visible_idx
        last_rendered = visible_idx
        last_render_row = y - 1
        if y >= status_y:
            break
        y += 1

    has_more_above = start > 0
    has_more_below = last_rendered is not None and last_rendered < total_visible - 1
    arrow_attr = curses.A_DIM | curses.A_BOLD
    if has_more_above and first_render_row is not None:
        arrow_row = max(body_top, min(first_render_row, height - 1))
        stdscr.addnstr(arrow_row, left_margin - 1, "↑", 1, arrow_attr)
    if has_more_below and last_render_row is not None:
        arrow_row = max(body_top, min(last_render_row, height - 1))
        stdscr.addnstr(arrow_row, left_margin - 1, "↓", 1, arrow_attr)

    separator_x = right_start - 1
    if 0 <= separator_x < width:
        stdscr.vline(body_top, separator_x, curses.ACS_VLINE, max(0, min(body_height, height - body_top)))

    wrap_width = max(10, right_width)
    right_y = body_top
    focus_area = ui_state.get("focus_area", 0)
    help_focus = focus_area == 1

    if status:
        for line in textwrap.wrap(status, wrap_width):
            if right_y >= status_y:
                break
            stdscr.addnstr(right_y, right_start, line, wrap_width, curses.A_BOLD | curses.color_pair(2))
            right_y += 1
        if right_y < status_y:
            right_y += 1

    selected_opt = options[selected_idx] if 0 <= selected_idx < len(options) else None
    if right_y < status_y:
        available_rows = status_y - (right_y + 1)
        header_text, visible_help, has_selection = prepare_help_panel(
            selected_opt,
            state,
            wrap_width,
            available_rows,
            ui_state,
            height,
        )
        header_attr = curses.A_BOLD | (curses.A_REVERSE if help_focus else 0)
        stdscr.addnstr(right_y, right_start, header_text[:wrap_width], wrap_width, header_attr)
        right_y += 1
        for line, attr in visible_help:
            if right_y >= status_y:
                break
            line_attr = (attr or curses.A_DIM) | (curses.A_REVERSE if help_focus else 0)
            stdscr.addnstr(right_y, right_start, line[:wrap_width], wrap_width, line_attr)
            right_y += 1
        if not has_selection:
            ui_state["help_max_offset"] = 0
            ui_state["help_visible_lines"] = 0
    else:
        ui_state["help_max_offset"] = 0
        ui_state["help_visible_lines"] = 0
        ui_state.setdefault("help_total_lines", 0)

    if status_y >= 0:
        stdscr.addnstr(status_y, 2, status[: max(1, width - 4)], max(1, width - 4), curses.color_pair(2) | curses.A_BOLD)
    logs_start = max(body_top, status_y - 3)
    draw_logs(stdscr, ui_state, logs_start, width, height)
    if instructions_y >= 0:
        instructions = STRINGS["instructions"]
        stdscr.addnstr(instructions_y, 2, instructions[: max(1, width - 4)], max(1, width - 4), curses.A_DIM)

    return first_rendered, last_rendered


def _draw_tablet_layout(
    stdscr: "curses._CursesWindow",
    options: List[OptionBase],
    option_info: List[tuple[int, OptionBase, int]],
    selected_idx: int,
    scroll: int,
    state: dict,
    ui_state: dict,
    status: str,
    height: int,
    width: int,
) -> Tuple[Optional[int], Optional[int]]:
    body_top = 2
    status_y = max(0, height - 2)
    instructions_y = max(0, height - 1)
    body_height = max(0, status_y - body_top)
    if body_height <= 0:
        warning = "Not enough rows to render options. Increase terminal height."
        stdscr.addnstr(body_top, 2, warning[: max(1, width - 4)], max(1, width - 4), curses.A_BOLD | curses.color_pair(2))
        return None, None

    available_width = width - 4
    column_gap = 3
    column_count = 2 if available_width >= 40 else 1
    if column_count <= 1:
        column_gap = 0
    column_width = max(20, (available_width - column_gap * (column_count - 1)) // column_count)
    x_positions = [2 + (column_width + column_gap) * idx for idx in range(column_count)]
    y_positions = [body_top] * column_count

    total_visible = len(option_info)
    start = max(0, min(scroll, max(0, total_visible - 1)))
    first_rendered: Optional[int] = None
    last_rendered: Optional[int] = None
    first_render_row: Optional[int] = None
    last_render_row: Optional[int] = None

    idx = start
    current_col = 0
    while idx < total_visible:
        opt_index, opt, _ = option_info[idx]
        opt_height = max(1, opt.height(column_width, state))
        placed = False
        col = current_col
        while col < column_count:
            if y_positions[col] + opt_height <= body_top + body_height:
                placed = True
                break
            col += 1
        if not placed:
            break
        current_col = col
        draw_y = y_positions[col]
        draw_x = x_positions[col]
        extra = opt.render(stdscr, draw_y, column_width, opt_index == selected_idx, state)
        if first_rendered is None:
            first_rendered = idx
            first_render_row = draw_y
        last_rendered = idx
        last_render_row = max(last_render_row or draw_y, draw_y + extra - 1)
        y_positions[col] = draw_y + extra + 1
        idx += 1

    has_more_above = start > 0
    has_more_below = last_rendered is not None and last_rendered < total_visible - 1
    arrow_attr = curses.A_DIM | curses.A_BOLD
    if has_more_above and first_render_row is not None:
        stdscr.addnstr(max(body_top, first_render_row), 0, "↑", 1, arrow_attr)
    if has_more_below and last_render_row is not None:
        stdscr.addnstr(min(body_top + body_height - 1, last_render_row), 0, "↓", 1, arrow_attr)

    wrap_width = max(10, width - 4)
    focus_area = ui_state.get("focus_area", 0)
    help_focus = focus_area == 1
    help_start = body_top + body_height
    selected_opt = options[selected_idx] if 0 <= selected_idx < len(options) else None
    if help_start < status_y:
        available_rows = status_y - (help_start + 1)
        header_text, visible_help, has_selection = prepare_help_panel(
            selected_opt,
            state,
            wrap_width,
            available_rows,
            ui_state,
            height,
        )
        header_attr = curses.A_BOLD | (curses.A_REVERSE if help_focus else curses.A_DIM)
        stdscr.hline(help_start, 1, curses.ACS_HLINE, width - 2)
        stdscr.addnstr(help_start, 3, f" {STRINGS['help_heading']} • {header_text} "[: max(1, width - 6)], max(1, width - 6), header_attr)
        row = help_start + 1
        for line, attr in visible_help:
            if row >= status_y:
                break
            line_attr = (attr or curses.A_DIM) | (curses.A_REVERSE if help_focus else 0)
            stdscr.addnstr(row, 2, line[: max(1, width - 4)], max(1, width - 4), line_attr)
            row += 1
        if not has_selection:
            ui_state["help_max_offset"] = 0
            ui_state["help_visible_lines"] = 0
    else:
        ui_state["help_max_offset"] = 0
        ui_state["help_visible_lines"] = 0
        ui_state.setdefault("help_total_lines", 0)

    if status and status_y >= 0:
        stdscr.addnstr(status_y, 2, status[: max(1, width - 4)], max(1, width - 4), curses.color_pair(2) | curses.A_BOLD)
    logs_start = max(body_top, status_y - 3)
    draw_logs(stdscr, ui_state, logs_start, width, height)
    if instructions_y >= 0:
        instructions = STRINGS["instructions"]
        stdscr.addnstr(instructions_y, 2, instructions[: max(1, width - 4)], max(1, width - 4), curses.A_DIM)

    return first_rendered, last_rendered


def next_visible(options: List[OptionBase], state: dict, start: int, delta: int) -> int:
    idx = start
    count = len(options)
    for _ in range(count):
        idx = (idx + delta) % count
        if options[idx].is_visible(state):
            return idx
    return start


def first_visible(options: List[OptionBase], state: dict) -> int:
    for idx, opt in enumerate(options):
        if opt.is_visible(state):
            return idx
    return 0


def build_visible_layout(
    options: List[OptionBase],
    width: int,
    state: dict,
) -> List[tuple[int, OptionBase, int]]:
    layout: List[tuple[int, OptionBase, int]] = []
    for idx, opt in enumerate(options):
        if not opt.is_visible(state):
            continue
        h = opt.height(width - 4, state)
        if h > 0:
            layout.append((idx, opt, h))
    return layout


def draw_screen(
    stdscr: "curses._CursesWindow",
    options: List[OptionBase],
    option_info: List[tuple[int, OptionBase, int]],
    selected_idx: int,
    scroll: int,
    state: dict,
    ui_state: dict,
) -> Tuple[Optional[int], Optional[int]]:
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    if height < 6 or width < 30:
        warning = STRINGS["no_space_warning"] if "no_space_warning" in STRINGS else "Terminal too small for the launcher. Please enlarge."
        stdscr.addnstr(0, 0, warning[: max(1, width - 1)], max(1, width - 1), curses.A_BOLD)
        stdscr.refresh()
        return None, None

    title = STRINGS["title"]
    stdscr.addnstr(0, max(1, (width - len(title)) // 2), title[: width - 2], curses.A_BOLD)

    instructions = STRINGS["instructions"]
    stdscr.addnstr(height - 1, 2, instructions[: max(1, width - 4)], max(1, width - 4), curses.A_DIM)

    status = state.get("status", "")

    if width >= 140:
        first_rendered, last_rendered = _draw_wide_layout(
            stdscr,
            options,
            option_info,
            selected_idx,
            scroll,
            state,
            ui_state,
            status,
            height,
            width,
        )
        stdscr.refresh()
        return first_rendered, last_rendered

    if width >= 80:
        first_rendered, last_rendered = _draw_tablet_layout(
            stdscr,
            options,
            option_info,
            selected_idx,
            scroll,
            state,
            ui_state,
            status,
            height,
            width,
        )
        stdscr.refresh()
        return first_rendered, last_rendered

    body_top = 2
    status_y = max(0, height - 2)
    instructions_y = max(0, height - 1)
    body_bottom = status_y - 1
    body_height = max(0, body_bottom - body_top + 1)
    if body_height <= 0 or not option_info:
        stdscr.refresh()
        return None, None

    scroll = max(0, min(scroll, max(0, len(option_info) - 1)))
    y = body_top
    first_rendered: Optional[int] = None
    last_rendered: Optional[int] = None
    first_render_row: Optional[int] = None
    last_render_row: Optional[int] = None

    for visible_idx in range(scroll, len(option_info)):
        opt_index, opt, opt_height = option_info[visible_idx]
        if y > body_bottom:
            break
        render_start = y
        if opt_height > body_height and first_rendered is not None:
            break
        if y + opt_height - 1 > body_bottom:
            if first_rendered is None:
                extra = opt.render(stdscr, y, width - 4, opt_index == selected_idx, state)
                first_render_row = render_start
                y += extra
                first_rendered = visible_idx
                last_rendered = visible_idx
                last_render_row = y - 1
            break
        extra = opt.render(stdscr, y, width - 4, opt_index == selected_idx, state)
        if first_render_row is None:
            first_render_row = render_start
        y += extra
        if first_rendered is None:
            first_rendered = visible_idx
        last_rendered = visible_idx
        last_render_row = y - 1
        if y > body_bottom:
            break
        y += 1

    has_more_above = scroll > 0
    has_more_below = last_rendered is not None and last_rendered < len(option_info) - 1
    arrow_attr = curses.A_DIM | curses.A_BOLD
    if has_more_above and first_render_row is not None:
        arrow_row = max(body_top, min(first_render_row, body_bottom))
        stdscr.addnstr(arrow_row, 0, "↑", 1, arrow_attr)
    if has_more_below and last_render_row is not None:
        arrow_row = max(body_top, min(last_render_row, body_bottom))
        stdscr.addnstr(arrow_row, 0, "↓", 1, arrow_attr)

    wrap_width = max(10, width - 4)
    focus_area = ui_state.get("focus_area", 0)
    help_focus = focus_area == 1
    selected_opt = options[selected_idx] if 0 <= selected_idx < len(options) else None
    help_start = body_bottom + 1
    if help_start < status_y:
        available_rows = status_y - (help_start + 1)
        header_text, visible_help, has_selection = prepare_help_panel(
            selected_opt,
            state,
            wrap_width,
            available_rows,
            ui_state,
            height,
        )
        header_attr = curses.A_BOLD | (curses.A_REVERSE if help_focus else curses.A_DIM)
        stdscr.hline(help_start, 1, curses.ACS_HLINE, width - 2)
        stdscr.addnstr(help_start, 3, f" {STRINGS['help_heading']} • {header_text} "[: max(1, width - 6)], max(1, width - 6), header_attr)
        row = help_start + 1
        for line, attr in visible_help:
            if row >= status_y:
                break
            line_attr = (attr or curses.A_DIM) | (curses.A_REVERSE if help_focus else 0)
            stdscr.addnstr(row, 2, line[: max(1, width - 4)], max(1, width - 4), line_attr)
            row += 1
        if not has_selection:
            ui_state["help_max_offset"] = 0
            ui_state["help_visible_lines"] = 0
    else:
        ui_state["help_max_offset"] = 0
        ui_state["help_visible_lines"] = 0
        ui_state.setdefault("help_total_lines", 0)

    if status:
        stdscr.addnstr(status_y, 2, status[: max(1, width - 4)], max(1, width - 4), curses.color_pair(2) | curses.A_BOLD)
    logs_start = max(body_top, status_y - 3)
    draw_logs(stdscr, ui_state, logs_start, width, height)
    stdscr.refresh()
    return first_rendered, last_rendered


def run_tui(stdscr: "curses._CursesWindow") -> None:
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN, -1)
    curses.init_pair(2, curses.COLOR_RED, -1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    state: dict = {
        "models_dir": str(MODELS_DIR),
        "local_models": None,
        "selected_local_model": "",
        "model_path": "",
        "rpc_hosts": "",
        "ctx_size": "",
        "n_gpu_layers": "999",
        "batch_size": "",
        "tensor_cache": "",
        "prompt_file": "",
        "prompt_text": "",
        "extra_flags": "",
        "local_worker": False,
        "worker_host": "0.0.0.0",
        "worker_port": DEFAULT_RPC_PORT,
        "worker_cache": "",
        "worker_devices": "",
        "worker_process": None,
        "discovered_hosts": [],
        "status": "",
    }

    saved_values = load_saved_state()
    if saved_values:
        state.update(saved_values)

    ui_state = {
        "focus_area": 0,
        "show_full_help": False,
        "logs": deque(maxlen=LOG_HISTORY),
        "status_message": None,
        "help_offset": 0,
        "logs_offset": 0,
        "help_max_offset": 0,
        "logs_max_offset": 0,
        "help_visible_lines": 0,
        "logs_visible_rows": 0,
        "last_selected": None,
    }

    _, initial_msg = refresh_local_models(state)
    if initial_msg:
        append_log(ui_state, initial_msg)

    append_log(ui_state, STRINGS["scan_start"])
    scan_start = time.time()
    discovered = discover_network_workers(state.get("worker_port", DEFAULT_RPC_PORT) or DEFAULT_RPC_PORT)
    if discovered:
        merge_hosts(state, discovered)
        msg = f"Discovered {len(discovered)} worker(s) on port {state.get('worker_port', DEFAULT_RPC_PORT)}."
        append_log(ui_state, msg)
        state["status"] = msg
    else:
        msg = f"No rpc-server instances detected on port {state.get('worker_port', DEFAULT_RPC_PORT)}."
        append_log(ui_state, msg)
        state["status"] = msg
    elapsed = max(time.time() - scan_start, 0.01)
    append_log(ui_state, f"{STRINGS['scan_complete']} ({elapsed:.1f}s)")

    options = build_options(state)
    apply_saved_values(options, state, saved_values)
    selected_idx = first_visible(options, state)
    scroll = 0

    while True:
        height, width = stdscr.getmaxyx()
        option_info = build_visible_layout(options, width, state)
        if not option_info:
            stdscr.erase()
            stdscr.addnstr(0, 0, STRINGS["no_options"], max(1, width - 1), curses.color_pair(2) | curses.A_BOLD)
            stdscr.addnstr(1, 0, "Press 'q' to quit.", max(1, width - 1), curses.A_DIM)
            stdscr.refresh()
            key = stdscr.getch()
            if key in (ord("q"), ord("Q")):
                save_state(options, state)
                return
            continue

        if not options[selected_idx].is_visible(state):
            selected_idx = first_visible(options, state)

        visible_mapping = {idx: i for i, (idx, _, _) in enumerate(option_info)}
        selected_visible_idx = visible_mapping.get(selected_idx, 0)
        scroll = max(0, min(scroll, max(0, len(option_info) - 1)))
        if selected_visible_idx < scroll:
            scroll = selected_visible_idx

        if ui_state.get("last_selected") != selected_idx:
            ui_state["help_offset"] = 0
            ui_state["last_selected"] = selected_idx

        if ui_state.get("status_message") is not None:
            state["status"] = ui_state["status_message"]

        first_rendered, last_rendered = draw_screen(
            stdscr,
            options,
            option_info,
            selected_idx,
            scroll,
            state,
            ui_state,
        )
        ui_state["status_message"] = None

        if selected_visible_idx < scroll:
            scroll = selected_visible_idx
        if (
            last_rendered is not None
            and selected_visible_idx > last_rendered
            and first_rendered is not None
        ):
            span = max(1, last_rendered - first_rendered + 1)
            scroll = max(0, selected_visible_idx - span + 1)

        key = stdscr.getch()
        if key in (ord("q"), ord("Q")):
            save_state(options, state)
            return
        if key in (curses.KEY_RESIZE,):
            continue
        if key == 9:  # Tab
            ui_state["focus_area"] = (ui_state.get("focus_area", 0) + 1) % 3
            continue
        if key == ord("?"):
            ui_state["show_full_help"] = not ui_state.get("show_full_help", False)
            ui_state["help_offset"] = 0
            append_log(ui_state, "Expanded help" if ui_state["show_full_help"] else "Collapsed help")
            continue

        focus_area = ui_state.get("focus_area", 0)
        if focus_area == 0:
            if key in (curses.KEY_DOWN, ord("j")):
                selected_idx = next_visible(options, state, selected_idx, +1)
                continue
            if key in (curses.KEY_UP, ord("k")):
                selected_idx = next_visible(options, state, selected_idx, -1)
                continue
            if key in (curses.KEY_PPAGE,):
                page = max(1, len(option_info) // 3 or 1)
                scroll = max(0, scroll - page)
                continue
            if key in (curses.KEY_NPAGE,):
                page = max(1, len(option_info) // 3 or 1)
                scroll = min(len(option_info) - 1, scroll + page)
                continue
            exit_requested, message = options[selected_idx].handle_key(key, state, stdscr)
            if message:
                state["status"] = message
                append_log(ui_state, message)
            else:
                state["status"] = ""
            if exit_requested:
                save_state(options, state)
                return
            continue

        if focus_area == 1:
            if key in (curses.KEY_DOWN, ord("j")):
                ui_state["help_offset"] = min(
                    ui_state.get("help_max_offset", 0),
                    ui_state.get("help_offset", 0) + 1,
                )
                continue
            if key in (curses.KEY_UP, ord("k")):
                ui_state["help_offset"] = max(0, ui_state.get("help_offset", 0) - 1)
                continue
            if key in (curses.KEY_NPAGE,):
                step = max(1, ui_state.get("help_visible_lines", 1))
                ui_state["help_offset"] = min(
                    ui_state.get("help_max_offset", 0),
                    ui_state.get("help_offset", 0) + step,
                )
                continue
            if key in (curses.KEY_PPAGE,):
                step = max(1, ui_state.get("help_visible_lines", 1))
                ui_state["help_offset"] = max(0, ui_state.get("help_offset", 0) - step)
                continue
            if key in (curses.KEY_HOME,):
                ui_state["help_offset"] = 0
                continue
            if key in (curses.KEY_END,):
                ui_state["help_offset"] = ui_state.get("help_max_offset", 0)
                continue
            if key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
                exit_requested, message = options[selected_idx].handle_key(key, state, stdscr)
                if message:
                    state["status"] = message
                    append_log(ui_state, message)
                if exit_requested:
                    save_state(options, state)
                    return
            continue

        if focus_area == 2:
            if key in (curses.KEY_DOWN, ord("j")):
                ui_state["logs_offset"] = min(
                    ui_state.get("logs_max_offset", 0),
                    ui_state.get("logs_offset", 0) + 1,
                )
                continue
            if key in (curses.KEY_UP, ord("k")):
                ui_state["logs_offset"] = max(0, ui_state.get("logs_offset", 0) - 1)
                continue
            if key in (curses.KEY_NPAGE,):
                step = max(1, ui_state.get("logs_visible_rows", 1))
                ui_state["logs_offset"] = min(
                    ui_state.get("logs_max_offset", 0),
                    ui_state.get("logs_offset", 0) + step,
                )
                continue
            if key in (curses.KEY_PPAGE,):
                step = max(1, ui_state.get("logs_visible_rows", 1))
                ui_state["logs_offset"] = max(0, ui_state.get("logs_offset", 0) - step)
                continue
            if key in (curses.KEY_HOME,):
                ui_state["logs_offset"] = ui_state.get("logs_max_offset", 0)
                continue
            if key in (curses.KEY_END,):
                ui_state["logs_offset"] = 0
                continue
            if key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
                exit_requested, message = options[selected_idx].handle_key(key, state, stdscr)
                if message:
                    state["status"] = message
                    append_log(ui_state, message)
                if exit_requested:
                    save_state(options, state)
                    return
            continue


def main() -> None:
    curses.wrapper(run_tui)


if __name__ == "__main__":
    main()
