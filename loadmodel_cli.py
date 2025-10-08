#!/usr/bin/env python3
"""Interactive CLI front-end for loadmodel.py."""
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
from typing import Callable, Iterable, List, Optional

import loadmodel


SCRIPT_DIR = Path(__file__).resolve().parent
LOADMODEL_SCRIPT = SCRIPT_DIR / "loadmodel.py"


@dataclass
class ChoiceItem:
    value: str
    label: str
    enabled: bool = True
    reason: str | None = None


class OptionBase:
    key: str
    name: str
    description: str

    def __init__(self, *, visible: Callable[[dict], bool] | None = None) -> None:
        self._visible = visible

    def visible(self, state: dict) -> bool:
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
    ) -> tuple[bool, str | None]:
        """Return (exit_requested, status_message)."""
        return False, None


class ChoiceOption(OptionBase):
    def __init__(
        self,
        key: str,
        name: str,
        description: str,
        state: dict,
        *,
        choices: Iterable[ChoiceItem] | None = None,
        choices_fn: Callable[[dict], Iterable[ChoiceItem]] | None = None,
        on_change: Callable[[dict, ChoiceItem], None] | None = None,
        help_text: str = "",
        visible: Callable[[dict], bool] | None = None,
    ) -> None:
        super().__init__(visible=visible)
        if choices is None and choices_fn is None:
            raise ValueError("choices or choices_fn required")
        self.key = key
        self.name = name
        self.description = help_text or description
        self._state = state
        self._static_choices = list(choices or [])
        self._choices_fn = choices_fn
        self._on_change = on_change
        self._choices: list[ChoiceItem] = []
        self._index = 0
        if key not in state and self._static_choices:
            state[key] = self._static_choices[0].value

    def _compute_choices(self) -> None:
        if self._choices_fn is not None:
            items = list(self._choices_fn(self._state))
        else:
            items = list(self._static_choices)
        if not items:
            items = [ChoiceItem(value="", label="(no options)", enabled=False, reason="No options found")]
        self._choices = items
        current_val = self._state.get(self.key)
        original_val = current_val
        found = False
        if current_val is not None:
            for idx, item in enumerate(self._choices):
                if item.value == current_val:
                    self._index = idx
                    found = True
                    break
        if not found:
            for idx, item in enumerate(self._choices):
                if item.enabled:
                    self._index = idx
                    self._state[self.key] = item.value
                    if self._on_change is not None and item.value != original_val:
                        self._on_change(self._state, item)
                    found = True
                    break
        if not found:
            self._index = 0
            self._state[self.key] = self._choices[0].value

    @property
    def current(self) -> ChoiceItem:
        self._compute_choices()
        return self._choices[self._index]

    def _advance(self, delta: int) -> None:
        self._compute_choices()
        count = len(self._choices)
        if count == 0:
            return
        start = self._index
        for _ in range(count):
            self._index = (self._index + delta) % count
            if self._choices[self._index].enabled:
                break
        if not self._choices[self._index].enabled:
            self._index = start
        self._state[self.key] = self._choices[self._index].value
        if self._on_change is not None:
            self._on_change(self._state, self._choices[self._index])

    def handle_key(
        self,
        key: int,
        state: dict,
        stdscr: "curses._CursesWindow",
    ) -> tuple[bool, str | None]:
        if key in (curses.KEY_LEFT, ord("h")):
            self._advance(-1)
        elif key in (curses.KEY_RIGHT, ord("l"), ord(" ")):
            self._advance(1)
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
        label = f"{self.name}: {current.label}"
        win.addnstr(y, 2, label, max(10, width - 4), attr)
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


class InputOption(OptionBase):
    def __init__(
        self,
        key: str,
        name: str,
        description: str,
        state: dict,
        *,
        placeholder: str = "",
        on_change: Callable[[dict, str], None] | None = None,
        visible: Callable[[dict], bool] | None = None,
    ) -> None:
        super().__init__(visible=visible)
        self.key = key
        self.name = name
        self.description = description
        self.placeholder = placeholder
        self._state = state
        self._on_change = on_change
        state.setdefault(key, "")

    def edit(self, stdscr: "curses._CursesWindow") -> None:
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
        current = str(self._state.get(self.key, ""))
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
            self._state[self.key] = new_value
            if self._on_change is not None:
                self._on_change(self._state, new_value)

    def handle_key(
        self,
        key: int,
        state: dict,
        stdscr: "curses._CursesWindow",
    ) -> tuple[bool, str | None]:
        if key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
            self.edit(stdscr)
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
        value = str(self._state.get(self.key, "")).strip()
        display = value or self.placeholder
        win.addnstr(y, 2, f"{self.name}: {display}", max(10, width - 4), attr)
        line_count = 1
        wrap_width = max(10, width - 6)
        for line in textwrap.wrap(self.description, wrap_width):
            win.addnstr(y + line_count, 6, line, max(10, width - 8), curses.A_DIM)
            line_count += 1
        return line_count


class ToggleOption(OptionBase):
    def __init__(
        self,
        key: str,
        name: str,
        description: str,
        state: dict,
        *,
        visible: Callable[[dict], bool] | None = None,
    ) -> None:
        super().__init__(visible=visible)
        self.key = key
        self.name = name
        self.description = description
        self._state = state
        state.setdefault(key, False)

    def toggle(self) -> None:
        self._state[self.key] = not bool(self._state.get(self.key))

    def handle_key(
        self,
        key: int,
        state: dict,
        stdscr: "curses._CursesWindow",
    ) -> tuple[bool, str | None]:
        if key in (curses.KEY_ENTER, ord(" "), ord("t")):
            self.toggle()
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
        mark = "✔" if self._state.get(self.key) else "✖"
        win.addnstr(y, 2, f"[{mark}] {self.name}", max(10, width - 4), attr)
        line_count = 1
        wrap_width = max(10, width - 6)
        for line in textwrap.wrap(self.description, wrap_width):
            win.addnstr(y + line_count, 6, line, max(10, width - 8), curses.A_DIM)
            line_count += 1
        return line_count


class ActionOption(OptionBase):
    def __init__(
        self,
        name: str,
        description: str,
        action: Callable[[dict], tuple[bool, str | None]],
        *,
        visible: Callable[[dict], bool] | None = None,
    ) -> None:
        super().__init__(visible=visible)
        self.key = f"action:{name}"
        self.name = name
        self.description = description
        self._action = action

    def handle_key(
        self,
        key: int,
        state: dict,
        stdscr: "curses._CursesWindow",
    ) -> tuple[bool, str | None]:
        if key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
            return self._action(state)
        return False, None

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


def list_local_gguf(models_dir: Path) -> List[str]:
    if not models_dir.exists() or not models_dir.is_dir():
        return []
    items = []
    try:
        for path in models_dir.rglob("*.gguf"):
            try:
                rel = path.relative_to(models_dir)
            except ValueError:
                rel = path.name
            items.append(str(rel))
    except Exception:
        return []
    return sorted(items)


def refresh_local_models(state: dict) -> tuple[bool, str | None]:
    base = Path(state.get("models_dir", str(loadmodel.MODELS_DIR))).expanduser()
    models = list_local_gguf(base)
    state["local_models"] = models
    if not models:
        return False, "No GGUF models found in the selected directory."
    return False, f"Found {len(models)} GGUF model(s)."


def set_model_from_choice(state: dict, choice: ChoiceItem) -> None:
    if not choice.value:
        return
    base = Path(state.get("models_dir", str(loadmodel.MODELS_DIR))).expanduser()
    candidate = (base / choice.value).expanduser()
    state["model_ref"] = str(candidate)
    state["selected_local_model"] = choice.value


def build_local_choices(state: dict) -> List[ChoiceItem]:
    models = state.get("local_models")
    if models is None:
        refresh_local_models(state)
        models = state.get("local_models") or []
    items = [
        ChoiceItem(value=m, label=m, enabled=True)
        for m in models
    ]
    if not items:
        items.append(ChoiceItem(value="", label="No GGUF files discovered", enabled=False))
    return items


def parse_int(value: str | None, *, default: Optional[int] = None, name: str = "value") -> int:
    if value is None or str(value).strip() == "":
        if default is None:
            raise ValueError(f"{name} must be provided")
        return default
    try:
        return int(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def split_extra_flags(value: str) -> List[str]:
    value = (value or "").strip()
    if not value:
        return []
    try:
        return shlex.split(value)
    except ValueError as exc:
        raise ValueError(f"Could not parse extra flags: {exc}") from exc


def build_command(state: dict) -> List[str]:
    mode = state.get("mode")
    model_ref = (state.get("model_ref") or "").strip()
    if mode not in {"llm", "embed", "rerank"}:
        raise ValueError("Select a mode (LLM, Embeddings, or Reranker).")
    if not model_ref:
        raise ValueError("Model reference/path is required.")

    cmd: List[str] = [sys.executable, str(LOADMODEL_SCRIPT)]
    if mode == "llm":
        cmd.append("--llm")
    elif mode == "embed":
        cmd.append("--embed")
    else:
        cmd.append("--rerank")

    host = (state.get("host") or "127.0.0.1").strip()
    port = parse_int(state.get("port"), default=45540, name="Port")
    cmd += ["--host", host, "--port", str(port)]

    models_path = (state.get("models_dir") or str(loadmodel.MODELS_DIR)).strip()
    cmd += ["--path", models_path]

    hf_token = (state.get("hf_token") or "").strip()
    if hf_token:
        cmd += ["--hf-token", hf_token]

    if mode in ("llm", "embed"):
        n_gpu = parse_int(state.get("n_gpu_layers"), default=999, name="--n-gpu-layers")
        cmd += ["--n-gpu-layers", str(n_gpu)]
        tensor_split = (state.get("tensor_split") or "").strip()
        if tensor_split:
            cmd += ["--tensor-split", tensor_split]
        ctx_size = (state.get("ctx_size") or "").strip()
        if ctx_size:
            ctx_int = parse_int(ctx_size, name="--ctx-size")
            cmd += ["--ctx-size", str(ctx_int)]
    else:
        device = (state.get("device") or "").strip()
        if device:
            cmd += ["--device", device]
        device_map = (state.get("device_map") or "auto").strip()
        if device_map:
            cmd += ["--device-map", device_map]
        dtype = (state.get("dtype") or "bf16").strip()
        if dtype:
            cmd += ["--dtype", dtype]
        quant = (state.get("quant") or "8bit").strip()
        if quant:
            cmd += ["--quant", quant]
        doc_batch = parse_int(state.get("doc_batch"), default=64, name="--doc-batch")
        cmd += ["--doc-batch", str(doc_batch)]
        max_len = parse_int(state.get("max_len"), default=4096, name="--max-len")
        cmd += ["--max-len", str(max_len)]
        max_memory = (state.get("max_memory") or "").strip()
        if max_memory:
            cmd += ["--max-memory", max_memory]
        if state.get("trust_remote_code"):
            cmd.append("--trust-remote-code")
        instruction = (state.get("instruction") or "").strip()
        if instruction:
            cmd += ["--instruction", instruction]

    cmd.append(model_ref)

    if mode in ("llm", "embed"):
        extra_flags = split_extra_flags(state.get("extra_flags", ""))
        if extra_flags:
            cmd.append("--extra")
            cmd.extend(extra_flags)

    return cmd


def shell_join(parts: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in parts)


def action_launch(state: dict) -> tuple[bool, str | None]:
    try:
        cmd = build_command(state)
    except ValueError as exc:
        return False, str(exc)

    curses.endwin()
    print("Launching loadmodel:\n  " + shell_join(cmd), flush=True)
    try:
        subprocess.call(cmd)
    except KeyboardInterrupt:
        pass
    return True, None


def action_refresh(state: dict) -> tuple[bool, str | None]:
    return refresh_local_models(state)


def ensure_status(state: dict) -> str:
    return str(state.get("status") or "")


def set_status(state: dict, message: str | None) -> None:
    state["status"] = message or ""


def build_options(state: dict) -> List[OptionBase]:
    options: List[OptionBase] = []

    options.append(
        ChoiceOption(
            key="mode",
            name="Mode",
            description="Choose which server to run (llama.cpp LLM, llama.cpp embeddings, or Transformers reranker).",
            state=state,
            choices=[
                ChoiceItem("llm", "LLM (llama.cpp server)"),
                ChoiceItem("embed", "Embeddings (llama.cpp server)"),
                ChoiceItem("rerank", "Reranker (Transformers)")
            ],
        )
    )

    options.append(
        InputOption(
            key="models_dir",
            name="Models directory",
            description="Directory that stores local GGUF files. Used for browsing as well as download destination.",
            state=state,
            placeholder=str(loadmodel.MODELS_DIR),
            on_change=lambda st, _new: refresh_local_models(st),
        )
    )

    options.append(
        ChoiceOption(
            key="selected_local_model",
            name="Local GGUF",
            description="Select a GGUF from the models directory to populate the model reference.",
            state=state,
            choices_fn=build_local_choices,
            on_change=set_model_from_choice,
            visible=lambda st: st.get("mode") in {"llm", "embed"},
        )
    )

    options.append(
        ActionOption(
            name="Refresh local models",
            description="Rescan the models directory for GGUF files.",
            action=action_refresh,
            visible=lambda st: st.get("mode") in {"llm", "embed"},
        )
    )

    options.append(
        InputOption(
            key="model_ref",
            name="Model reference",
            description="Path or Hugging Face reference (org/repo:quant or org/repo:file.gguf). Required before launching.",
            state=state,
            placeholder="Qwen/Qwen2-7B-Instruct:Q4_K_M or ./models/model.gguf",
        )
    )

    options.append(
        InputOption(
            key="hf_token",
            name="HF token",
            description="Optional Hugging Face token used for gated models.",
            state=state,
            placeholder="From HF_TOKEN env",
        )
    )

    options.append(
        InputOption(
            key="host",
            name="Host",
            description="Network interface for the server to bind.",
            state=state,
            placeholder="127.0.0.1",
        )
    )

    options.append(
        InputOption(
            key="port",
            name="Port",
            description="TCP port exposed by the server.",
            state=state,
            placeholder="45540",
        )
    )

    options.append(
        InputOption(
            key="n_gpu_layers",
            name="--n-gpu-layers",
            description="llama.cpp layers to keep on GPU. Leave blank for auto fallback.",
            state=state,
            placeholder="999",
            visible=lambda st: st.get("mode") in {"llm", "embed"},
        )
    )

    options.append(
        InputOption(
            key="tensor_split",
            name="--tensor-split",
            description="Comma-separated split for multiple GPUs (e.g. 50,50).",
            state=state,
            placeholder="",
            visible=lambda st: st.get("mode") in {"llm", "embed"},
        )
    )

    options.append(
        InputOption(
            key="ctx_size",
            name="--ctx-size",
            description="Context window for llama.cpp. Leave blank to use model default.",
            state=state,
            placeholder="4096",
            visible=lambda st: st.get("mode") in {"llm", "embed"},
        )
    )
    options.append(
        InputOption(
            key="extra_flags",
            name="Extra llama flags",
            description="Additional llama-server flags added after --extra (e.g. --rope-scaling linear --rope-freq-base 10000).",
            state=state,
            placeholder="",
            visible=lambda st: st.get("mode") in {"llm", "embed"},
        )
    )

    options.append(
        InputOption(
            key="device",
            name="--device",
            description="Preferred device for reranker weights (e.g. cuda or cpu).",
            state=state,
            placeholder="auto-detect",
            visible=lambda st: st.get("mode") == "rerank",
        )
    )

    options.append(
        InputOption(
            key="device_map",
            name="--device-map",
            description="Device placement strategy for reranker (auto, cuda, cpu).",
            state=state,
            placeholder="auto",
            visible=lambda st: st.get("mode") == "rerank",
        )
    )

    options.append(
        InputOption(
            key="dtype",
            name="--dtype",
            description="Model dtype for reranker (auto, bf16, fp16, fp32).",
            state=state,
            placeholder="bf16",
            visible=lambda st: st.get("mode") == "rerank",
        )
    )

    options.append(
        InputOption(
            key="quant",
            name="--quant",
            description="Quantization for reranker (none, 8bit, 4bit).",
            state=state,
            placeholder="8bit",
            visible=lambda st: st.get("mode") == "rerank",
        )
    )

    options.append(
        InputOption(
            key="doc_batch",
            name="--doc-batch",
            description="Batch size for reranker scoring (reduce if you hit OOM).",
            state=state,
            placeholder="64",
            visible=lambda st: st.get("mode") == "rerank",
        )
    )

    options.append(
        InputOption(
            key="max_len",
            name="--max-len",
            description="Maximum token length for reranker tokenizer.",
            state=state,
            placeholder="4096",
            visible=lambda st: st.get("mode") == "rerank",
        )
    )

    options.append(
        InputOption(
            key="max_memory",
            name="--max-memory",
            description="Optional max memory map for reranker (e.g. 0=22GiB,1=22GiB,cpu=128GiB).",
            state=state,
            placeholder="",
            visible=lambda st: st.get("mode") == "rerank",
        )
    )

    options.append(
        ToggleOption(
            key="trust_remote_code",
            name="--trust-remote-code",
            description="Allow execution of remote code when loading reranker from Hugging Face.",
            state=state,
            visible=lambda st: st.get("mode") == "rerank",
        )
    )

    options.append(
        InputOption(
            key="instruction",
            name="Instruction",
            description="Custom rerank instruction (defaults to web search prompt).",
            state=state,
            placeholder="",
            visible=lambda st: st.get("mode") == "rerank",
        )
    )

    options.append(
        ActionOption(
            name="Launch",
            description="Download the model if needed and start the selected server.",
            action=action_launch,
        )
    )

    return options


def draw_screen(
    stdscr: "curses._CursesWindow",
    options: List[OptionBase],
    state: dict,
    selected_index: int,
) -> None:
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    title = "loadmodel launcher"
    stdscr.addnstr(0, 2, title, w - 4, curses.A_BOLD)
    y = 2
    visible_options = [opt for opt in options if opt.visible(state)]
    for idx, opt in enumerate(visible_options):
        if y >= h - 3:
            break
        lines = opt.render(stdscr, y, w - 4, idx == selected_index, state)
        y += lines + 1
    status = ensure_status(state)
    if status:
        stdscr.addnstr(h - 2, 2, status, w - 4, curses.color_pair(2) | curses.A_BOLD)
    instructions = "Arrows: navigate • Enter: edit/activate • q: quit"
    stdscr.addnstr(h - 1, 2, instructions, w - 4, curses.A_DIM)
    stdscr.refresh()


def run_tui(stdscr: "curses._CursesWindow") -> None:
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN, -1)
    curses.init_pair(2, curses.COLOR_RED, -1)

    state: dict = {
        "mode": "llm",
        "models_dir": str(loadmodel.MODELS_DIR),
        "hf_token": os.environ.get("HF_TOKEN", ""),
        "host": "127.0.0.1",
        "port": "45540",
        "n_gpu_layers": "999",
        "tensor_split": "",
        "ctx_size": "",
        "extra_flags": "",
        "device": "",
        "device_map": "auto",
        "dtype": "bf16",
        "quant": "8bit",
        "doc_batch": "64",
        "max_len": "4096",
        "max_memory": "",
        "trust_remote_code": False,
        "instruction": "",
        "model_ref": "",
        "local_models": None,
        "selected_local_model": "",
        "status": "",
    }

    _exit, initial_msg = refresh_local_models(state)
    if initial_msg:
        set_status(state, initial_msg)

    options = build_options(state)
    selected_index = 0

    while True:
        visible = [opt for opt in options if opt.visible(state)]
        if not visible:
            set_status(state, "No options available")
            return
        selected_index = max(0, min(selected_index, len(visible) - 1))
        draw_screen(stdscr, options, state, selected_index)
        key = stdscr.getch()
        if key in (ord("q"), ord("Q")):
            break
        if key in (curses.KEY_UP, ord("k")):
            selected_index = (selected_index - 1) % len(visible)
            continue
        if key in (curses.KEY_DOWN, ord("j")):
            selected_index = (selected_index + 1) % len(visible)
            continue
        exit_requested, message = visible[selected_index].handle_key(key, state, stdscr)
        if message:
            set_status(state, message)
        else:
            set_status(state, "")
        if exit_requested:
            break


def main() -> None:
    curses.wrapper(run_tui)


if __name__ == "__main__":
    main()

