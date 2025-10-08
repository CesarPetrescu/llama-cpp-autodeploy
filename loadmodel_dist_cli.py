#!/usr/bin/env python3
"""Interactive CLI for launching llama.cpp distributed (RPC) inference."""
from __future__ import annotations

import curses
import curses.textpad
import os
import shlex
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
BIN_DIR = SCRIPT_DIR / "bin"
LLAMA_CLI = BIN_DIR / "llama-cli"
RPC_SERVER = BIN_DIR / "rpc-server"
MODELS_DIR = SCRIPT_DIR / "models"


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

    def _edit(self, stdscr: "curses._CursesWindow", state: dict) -> None:
        h, w = stdscr.getmaxyx()
        prompt = f"Enter {self.name}: "
        width = max(20, min(60, w - len(prompt) - 4))
        start_x = max(1, (w - (len(prompt) + width + 2)) // 2)
        start_y = max(1, h // 2 - 1)
        win = curses.newwin(3, len(prompt) + width + 2, start_y, start_x)
        win.border()
        win.addstr(1, 1, prompt)
        edit_win = win.derwin(1, width, 1, len(prompt) + 1)
        edit_win.erase()
        current = str(state.get(self.key, "")).strip()
        if current:
            edit_win.addnstr(0, 0, current, width - 1)
        curses.curs_set(1)
        textpad = curses.textpad.Textbox(edit_win)
        win.refresh()
        try:
            new_value = textpad.edit().strip()
        except Exception:
            new_value = current
        curses.curs_set(0)
        if new_value != current:
            state[self.key] = new_value
            if self._on_change is not None:
                self._on_change(state, new_value)

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
        value = str(state.get(self.key, "")).strip()
        display = value or self.placeholder
        win.addnstr(y, 2, f"{self.name}: {display}", max(10, width - 4), attr)
        line_count = 1
        wrap_width = max(10, width - 6)
        for line in textwrap.wrap(self.description, wrap_width):
            win.addnstr(y + line_count, 6, line, max(10, width - 8), curses.A_DIM)
            line_count += 1
        return line_count

    def height(self, width: int, state: dict) -> int:
        wrap_width = max(10, width - 6)
        return 1 + len(textwrap.wrap(self.description, wrap_width))


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

    def render(
        self,
        win: "curses._CursesWindow",
        y: int,
        width: int,
        selected: bool,
        state: dict,
    ) -> int:
        attr = curses.A_REVERSE if selected else curses.A_NORMAL
        mark = "✔" if state.get(self.key) else "✖"
        win.addnstr(y, 2, f"[{mark}] {self.name}", max(10, width - 4), attr)
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
        self._state = state
        self._static_choices = list(choices or [])
        self._choices_fn = choices_fn
        self._on_change = on_change
        self._choices: List[ChoiceItem] = []
        self._index = 0

    def _compute_choices(self) -> None:
        state = self._state
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

    def handle_key(
        self,
        key: int,
        state: dict,
        stdscr: "curses._CursesWindow",
    ) -> Tuple[bool, Optional[str]]:
        self._compute_choices()
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
        self._compute_choices()
        attr = curses.A_REVERSE if selected else curses.A_NORMAL
        current = self._choices[self._index]
        win.addnstr(y, 2, f"{self.name}: {current.label}", max(10, width - 4), attr)
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
        self._compute_choices()
        wrap_width = max(10, width - 6)
        base = 1 + len(textwrap.wrap(self.description, wrap_width))
        current = self._choices[self._index]
        if not current.enabled and current.reason:
            base += 1
        return base


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

    def render(
        self,
        win: "curses._CursesWindow",
        y: int,
        width: int,
        selected: bool,
        state: dict,
    ) -> int:
        attr = curses.A_REVERSE if selected else curses.A_BOLD
        label = f"▶ {self.name}"
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
    port = state.get("worker_port", "50052").strip() or "50052"
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
    try:
        cmd = build_llama_command(state)
    except (FileNotFoundError, ValueError) as exc:
        return False, str(exc)

    curses.endwin()
    if messages:
        for msg in messages:
            print(msg, flush=True)
    print("Launching distributed llama-cli:\n  " + shell_join(cmd), flush=True)
    try:
        subprocess.call(cmd)
    except KeyboardInterrupt:
        pass
    return True, None


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
    return False, f"rpc-server started (pid {proc.pid})."


def action_refresh_models(state: dict) -> Tuple[bool, Optional[str]]:
    return refresh_local_models(state)


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
            placeholder="192.168.1.10:50052,192.168.1.11:50052",
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
            placeholder="50052",
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
) -> Tuple[Optional[int], Optional[int]]:
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    if height < 6 or width < 30:
        warning = "Terminal too small for the launcher. Please enlarge."
        stdscr.addnstr(0, 0, warning[: max(1, width - 1)], max(1, width - 1), curses.A_BOLD)
        stdscr.refresh()
        return None, None

    title = "llama.cpp Distributed Launcher"
    stdscr.addnstr(0, max(1, (width - len(title)) // 2), title[: width - 2], curses.A_BOLD)

    instructions = "Arrows: navigate • Enter: edit/activate • PgUp/PgDn: scroll • q: quit"
    stdscr.addnstr(height - 1, 2, instructions[: max(1, width - 4)], max(1, width - 4), curses.A_DIM)

    body_top = 2
    body_bottom = height - 3
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

    status = state.get("status", "")
    if status:
        stdscr.addnstr(height - 2, 2, status[: max(1, width - 4)], max(1, width - 4), curses.color_pair(2) | curses.A_BOLD)

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
        "n_gpu_layers": "",
        "batch_size": "",
        "tensor_cache": "",
        "prompt_file": "",
        "prompt_text": "",
        "extra_flags": "",
        "local_worker": False,
        "worker_host": "0.0.0.0",
        "worker_port": "50052",
        "worker_cache": "",
        "worker_devices": "",
        "worker_process": None,
        "status": "",
    }

    _, initial_msg = refresh_local_models(state)
    if initial_msg:
        state["status"] = initial_msg

    options = build_options(state)
    selected_idx = first_visible(options, state)
    scroll = 0

    while True:
        height, width = stdscr.getmaxyx()
        option_info = build_visible_layout(options, width, state)
        if not options[selected_idx].is_visible(state):
            selected_idx = first_visible(options, state)
        if not option_info:
            stdscr.erase()
            stdscr.addnstr(0, 0, "No options available.", max(1, width - 1), curses.color_pair(2) | curses.A_BOLD)
            stdscr.refresh()
            stdscr.getch()
            break

        # map selection to visible index
        selected_visible_idx = next(
            (i for i, (idx, _, _) in enumerate(option_info) if idx == selected_idx),
            0,
        )
        if selected_visible_idx < scroll:
            scroll = selected_visible_idx

        first_rendered, last_rendered = draw_screen(
            stdscr,
            options,
            option_info,
            selected_idx,
            scroll,
            state,
        )
        key = stdscr.getch()
        if key in (ord("q"), ord("Q")):
            break
        if key in (curses.KEY_RESIZE,):
            continue

        if key in (curses.KEY_UP, ord("k")):
            selected_idx = next_visible(options, state, selected_idx, -1)
        elif key in (curses.KEY_DOWN, ord("j")):
            selected_idx = next_visible(options, state, selected_idx, +1)
        elif key in (curses.KEY_PPAGE,):
            page = max(1, len(option_info) // 3 or 1)
            scroll = max(0, scroll - page)
        elif key in (curses.KEY_NPAGE,):
            page = max(1, len(option_info) // 3 or 1)
            scroll = min(len(option_info) - 1, scroll + page)
        else:
            exit_requested, message = options[selected_idx].handle_key(key, state, stdscr)
            if message:
                state["status"] = message
            else:
                state["status"] = ""
            if exit_requested:
                break

        visible_mapping = {idx: i for i, (idx, _, _) in enumerate(option_info)}
        selected_visible_idx = visible_mapping.get(selected_idx)
        if selected_visible_idx is not None:
            if first_rendered is not None and selected_visible_idx < scroll:
                scroll = selected_visible_idx
            if last_rendered is not None and selected_visible_idx > last_rendered:
                scroll = selected_visible_idx


def main() -> None:
    curses.wrapper(run_tui)


if __name__ == "__main__":
    main()
