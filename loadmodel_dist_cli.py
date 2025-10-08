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
from typing import Callable, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
BIN_DIR = SCRIPT_DIR / "bin"
LLAMA_CLI = BIN_DIR / "llama-cli"
RPC_SERVER = BIN_DIR / "rpc-server"


def shell_join(parts: List[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in parts)


class OptionBase:
    key: str
    name: str
    description: str

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description

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
    ) -> None:
        super().__init__(name, description)
        self.key = key
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
            edit_win.addstr(0, 0, current)
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
    ) -> None:
        super().__init__(name, description)
        self.key = key

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


class ActionOption(OptionBase):
    def __init__(
        self,
        name: str,
        description: str,
        action: Callable[[dict], Tuple[bool, Optional[str]]],
    ) -> None:
        super().__init__(name, description)
        self.key = f"action:{name}"
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


def ensure_binary(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; build llama.cpp first using autodevops.")


def parse_hosts(hosts: str) -> str:
    entries = [h.strip() for h in hosts.split(",") if h.strip()]
    if not entries:
        raise ValueError("At least one RPC host must be provided.")
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


def action_launch(state: dict) -> Tuple[bool, Optional[str]]:
    try:
        cmd = build_llama_command(state)
    except (FileNotFoundError, ValueError) as exc:
        return False, str(exc)

    curses.endwin()
    print("Launching distributed llama-cli:\n  " + shell_join(cmd), flush=True)
    try:
        subprocess.call(cmd)
    except KeyboardInterrupt:
        pass
    return True, None


def action_start_worker(state: dict) -> Tuple[bool, Optional[str]]:
    if not state.get("local_worker", False):
        return False, "Enable 'Launch local rpc-server' toggle first."
    env = os.environ.copy()
    device = state.get("worker_devices", "").strip()
    if device:
        env["CUDA_VISIBLE_DEVICES"] = device
    try:
        cmd = build_rpc_command(state)
    except FileNotFoundError as exc:
        return False, str(exc)
    curses.endwin()
    print("Starting rpc-server locally:\n  " + shell_join(cmd), flush=True)
    try:
        subprocess.call(cmd, env=env)
    except KeyboardInterrupt:
        pass
    return False, "rpc-server exited"


def build_options(state: dict) -> List[OptionBase]:
    return [
        InputOption(
            "model_path",
            "Model GGUF path",
            "Absolute path to the GGUF model you want to serve.",
            placeholder=str((SCRIPT_DIR / "models").resolve() / "model.gguf"),
        ),
        InputOption(
            "rpc_hosts",
            "Worker hosts",
            "Comma-separated list of host:port entries (e.g. 192.168.1.10:50052,192.168.1.11:50052).",
            placeholder="192.168.1.10:50052,192.168.1.11:50052",
        ),
        InputOption(
            "ctx_size",
            "Context window",
            "Context window in tokens (--ctx-size). Leave blank to use model default.",
            placeholder="4096",
        ),
        InputOption(
            "n_gpu_layers",
            "GPU layers",
            "Number of layers to keep on GPU (-ngl). Use 99 to place all possible layers on workers.",
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
            "Optional cache directory (-c) shared across runs to reduce network transfers.",
            placeholder="/tmp/llama-cache",
        ),
        InputOption(
            "prompt_file",
            "Prompt file",
            "Path to a prompt file (-f). Leave blank to provide a prompt string.",
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
            "Start a local rpc-server instance using the options below.",
        ),
        InputOption(
            "worker_host",
            "rpc-server host",
            "Interface for the local rpc-server to bind.",
            placeholder="0.0.0.0",
        ),
        InputOption(
            "worker_port",
            "rpc-server port",
            "Listening port for the local rpc-server.",
            placeholder="50052",
        ),
        InputOption(
            "worker_cache",
            "rpc-server cache",
            "Optional tensor cache directory passed to rpc-server (-c).",
            placeholder="/tmp/llama-cache",
        ),
        InputOption(
            "worker_devices",
            "CUDA_VISIBLE_DEVICES",
            "Restrict GPUs visible to the local rpc-server (e.g. 0 or 0,1).",
            placeholder="",
        ),
        ActionOption(
            "Launch distributed llama-cli",
            "Start llama-cli with the configured RPC hosts.",
            action_launch,
        ),
        ActionOption(
            "Start local rpc-server",
            "Run rpc-server locally using the parameters above.",
            action_start_worker,
        ),
    ]


def compute_option_layout(
    options: List[OptionBase],
    width: int,
    state: dict,
) -> List[tuple[int, OptionBase, int]]:
    layout: List[tuple[int, OptionBase, int]] = []
    for idx, opt in enumerate(options):
        h = opt.height(width - 4, state)
        if h > 0:
            layout.append((idx, opt, h))
    return layout


def draw_screen(
    stdscr: "curses._CursesWindow",
    options: List[OptionBase],
    selected_idx: int,
    scroll: int,
    state: dict,
    option_info: List[tuple[int, OptionBase, int]],
) -> Tuple[int | None, int | None]:
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    if height < 6 or width < 30:
        msg = "Terminal too small for the launcher. Please enlarge."
        stdscr.addnstr(0, 0, msg[: max(1, width - 1)], max(1, width - 1), curses.A_BOLD)
        stdscr.refresh()
        return None, None

    title = "llama.cpp Distributed Launcher"
    stdscr.addnstr(0, max(1, (width - len(title)) // 2), title[: width - 2], curses.A_BOLD)

    instructions = "Arrows: navigate • Enter: edit/activate • PgUp/PgDn: scroll • q: quit"
    stdscr.addnstr(height - 1, 2, instructions[: max(1, width - 4)], max(1, width - 4), curses.A_DIM)

    body_top = 2
    body_bottom = height - 3
    body_height = max(0, body_bottom - body_top + 1)
    if body_height <= 0:
        stdscr.refresh()
        return None, None

    # Determine visible options with heights
    total = len(option_info)
    if total == 0:
        stdscr.refresh()
        return None, None

    scroll = max(0, min(scroll, max(0, total - 1)))
    y = body_top
    first_rendered: int | None = None
    last_rendered: int | None = None
    first_render_row: int | None = None
    last_render_row: int | None = None

    for visible_idx in range(scroll, total):
        opt_idx, opt, opt_height = option_info[visible_idx]
        if y > body_bottom:
            break
        if opt_height > body_height and first_rendered is not None:
            break
        render_start = y
        if y + opt_height - 1 > body_bottom:
            if first_rendered is None:
                extra = opt.render(stdscr, y, width - 4, opt_idx == selected_idx, state)
                first_render_row = render_start
                y += extra
                first_rendered = visible_idx
                last_rendered = visible_idx
                last_render_row = y - 1
            break
        extra = opt.render(stdscr, y, width - 4, opt_idx == selected_idx, state)
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
    has_more_below = last_rendered is not None and last_rendered < total - 1
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
    if first_rendered is None:
        return None, None
    return first_rendered, last_rendered


def run_tui(stdscr: "curses._CursesWindow") -> None:
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN, -1)
    curses.init_pair(2, curses.COLOR_RED, -1)

    state = {
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
        "status": "",
    }

    options = build_options(state)
    selected_idx = 0
    scroll = 0

    while True:
        height, width = stdscr.getmaxyx()
        option_info = compute_option_layout(options, width, state)
        if not option_info:
            state["status"] = "No options available."
            stdscr.erase()
            stdscr.addnstr(0, 0, state["status"], max(1, width - 1), curses.color_pair(2) | curses.A_BOLD)
            stdscr.refresh()
            stdscr.getch()
            break

        scroll = max(0, min(scroll, max(0, len(option_info) - 1)))
        first_last = draw_screen(stdscr, options, selected_idx, scroll, state, option_info)
        if first_last == (None, None):
            key = stdscr.getch()
            if key in (ord("q"), ord("Q")):
                break
            continue
        first_rendered, last_rendered = first_last
        key = stdscr.getch()
        if key in (ord("q"), ord("Q")):
            break
        if key in (curses.KEY_RESIZE,):
            continue
        handled = False
        if key in (curses.KEY_UP, ord("k")):
            selected_idx = (selected_idx - 1) % len(options)
            handled = True
        elif key in (curses.KEY_DOWN, ord("j")):
            selected_idx = (selected_idx + 1) % len(options)
            handled = True
        elif key in (curses.KEY_PPAGE,):
            scroll = max(0, scroll - max(1, len(option_info) // 3 or 1))
            handled = True
        elif key in (curses.KEY_NPAGE,):
            scroll = min(len(option_info) - 1, scroll + max(1, len(option_info) // 3 or 1))
            handled = True
        else:
            exit_requested, message = options[selected_idx].handle_key(key, state, stdscr)
            if message:
                state["status"] = message
            else:
                state["status"] = ""
            if exit_requested:
                break

        visible_idx = None
        for idx, (global_idx, _, _) in enumerate(option_info):
            if global_idx == selected_idx:
                visible_idx = idx
                break
        if visible_idx is not None:
            if visible_idx < scroll:
                scroll = visible_idx
            elif last_rendered is not None and visible_idx > last_rendered:
                scroll = visible_idx
        if handled:
            continue


def main() -> None:
    curses.wrapper(run_tui)


if __name__ == "__main__":
    main()
