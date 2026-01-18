#!/usr/bin/env python3
"""Interactive CLI front-end for loadmodel.py."""
from __future__ import annotations

import curses
import json
import os
import shlex
import subprocess
import sys
import textwrap
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import loadmodel
import memory_utils
import tui_utils


SCRIPT_DIR = Path(__file__).resolve().parent
LOADMODEL_SCRIPT = SCRIPT_DIR / "loadmodel.py"

STRINGS = {
    "title": "loadmodel launcher",
    "instructions": "Arrows: navigate • Enter: edit/activate • PgUp/PgDn: scroll • Tab: cycle panes • ?: toggle help • c: compact • q: quit",
    "logs_heading": "Logs",
    "help_heading": "Help",
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


def save_state(options: List["OptionBase"]) -> None:
    payload = {}
    for opt in options:
        try:
            payload[opt.key] = opt.get_value()
        except Exception:
            continue
    try:
        CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def apply_saved_values(options: List["OptionBase"], saved: dict) -> None:
    if not saved:
        return
    for opt in options:
        if opt.key in saved:
            try:
                opt.set_value(saved[opt.key])
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
CONFIG_PATH = SCRIPT_DIR / ".loadmodel_cli.json"
LOG_HISTORY = 200
SPINNER_FRAMES = "|/-\\"


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
    default_value: object
    icon: str = "[OPT]"

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

    def height(self, width: int, state: dict) -> int:
        raise NotImplementedError

    def is_modified(self, state: dict) -> bool:
        return False

    def get_summary(self, width: int) -> str:
        text = (self.description or "").strip()
        if not text:
            return ""
        summary = text.splitlines()[0].strip()
        if len(summary) > width:
            summary = summary[: max(0, width - 1)] + "…"
        return summary

    def get_value(self):
        raise NotImplementedError

    def set_value(self, value) -> None:
        raise NotImplementedError


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
        self.icon = "[SEL]"
        if key not in state and self._static_choices:
            state[key] = self._static_choices[0].value
        self.default_value = state.get(key)

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
        compact = bool(state.get("_ui_compact", False))
        attr = curses.A_REVERSE if selected else curses.A_NORMAL
        current = self._choices[self._index]
        marker = "*" if self.is_modified(state) else " "
        label = f"{marker}{self.icon} {self.name}: {current.label}"
        if not current.enabled and current.reason:
            attr |= curses.color_pair(2)
            label = f"{label} ⚠"
        win.addnstr(y, 2, label, max(10, width - 4), attr)
        if compact:
            return 1
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
        self._compute_choices()
        if bool(state.get("_ui_compact", False)):
            return 1
        base = 1
        wrap_width = max(10, width - 6)
        base += len(textwrap.wrap(self.description, wrap_width))
        current = self._choices[self._index]
        if not current.enabled and current.reason:
            base += 1
        return base

    def is_modified(self, state: dict) -> bool:
        return self._state.get(self.key) != self.default_value

    def get_value(self):
        self._compute_choices()
        return self._choices[self._index].value

    def set_value(self, value) -> None:
        self._compute_choices()
        for idx, item in enumerate(self._choices):
            if item.value == value:
                self._index = idx
                self._state[self.key] = item.value
                break


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
        self.icon = "[TXT]"
        self.default_value = state.get(key, "")
        state.setdefault(key, self.default_value)

    def edit(self, stdscr: "curses._CursesWindow") -> None:
        current = str(self._state.get(self.key, ""))
        result = tui_utils.edit_line_dialog(
            stdscr,
            title=f"Edit {self.name}",
            initial=current,
            allow_empty=True,
        )
        if not result.accepted:
            return
        if result.value != current:
            self._state[self.key] = result.value
            if self._on_change is not None:
                self._on_change(self._state, result.value)

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
        compact = bool(state.get("_ui_compact", False))
        attr = curses.A_REVERSE if selected else curses.A_NORMAL
        value = str(self._state.get(self.key, "")).strip()
        display = value or self.placeholder
        marker = "*" if self.is_modified(state) else " "
        label = f"{marker}{self.icon} {self.name}: {display}"
        win.addnstr(y, 2, label, max(10, width - 4), attr)
        if compact:
            return 1
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
        if bool(state.get("_ui_compact", False)):
            return 1
        wrap_width = max(10, width - 6)
        return 1 + len(textwrap.wrap(self.description, wrap_width))

    def is_modified(self, state: dict) -> bool:
        return str(state.get(self.key, "")) != str(self.default_value)

    def get_value(self):
        return self._state.get(self.key, "")

    def set_value(self, value) -> None:
        if value is None:
            return
        self._state[self.key] = str(value)


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
        self.icon = "[TGL]"
        self.default_value = bool(state.get(key, False))
        state.setdefault(key, self.default_value)

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
        compact = bool(state.get("_ui_compact", False))
        attr = curses.A_REVERSE if selected else curses.A_NORMAL
        mark = "✔" if self._state.get(self.key) else "✖"
        marker = "*" if self.is_modified(state) else " "
        label = f"{marker}{self.icon} [{mark}] {self.name}"
        win.addnstr(y, 2, label, max(10, width - 4), attr)
        if compact:
            return 1
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
        if bool(state.get("_ui_compact", False)):
            return 1
        wrap_width = max(10, width - 6)
        return 1 + len(textwrap.wrap(self.description, wrap_width))

    def is_modified(self, state: dict) -> bool:
        return bool(state.get(self.key)) != self.default_value

    def get_value(self):
        return bool(self._state.get(self.key))

    def set_value(self, value) -> None:
        self._state[self.key] = bool(value)


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
        compact = bool(state.get("_ui_compact", False))
        attr = curses.A_REVERSE if selected else curses.A_BOLD
        label = f"▶ {self.name}"
        win.addnstr(y, 2, label, max(10, width - 4), attr)
        if compact:
            return 1
        line_count = 1
        wrap_width = max(10, width - 6)
        for line in textwrap.wrap(self.description, wrap_width):
            win.addnstr(y + line_count, 6, line, max(10, width - 8), curses.A_DIM)
            line_count += 1
        return line_count

    def height(self, width: int, state: dict) -> int:
        if bool(state.get("_ui_compact", False)):
            return 1
        wrap_width = max(10, width - 6)
        return 1 + len(textwrap.wrap(self.description, wrap_width))


class ModelSummaryOption(OptionBase):
    def __init__(self, state: dict, *, visible: Callable[[dict], bool] | None = None) -> None:
        super().__init__(visible=visible)
        self._state = state
        self.key = "_model_overview"
        self.name = "Model overview"
        self.description = "Overview of memory footprint for the current selection."

    def render(
        self,
        win: "curses._CursesWindow",
        y: int,
        width: int,
        selected: bool,
        state: dict,
    ) -> int:
        if bool(state.get("_ui_compact", False)) and not selected:
            attr = curses.A_REVERSE if selected else curses.A_BOLD
            win.addnstr(y, 2, "Model overview", max(10, width - 4), attr)
            return 1
        profile = memory_utils.estimate_memory_profile(self._state)
        attr = curses.A_REVERSE if selected else curses.A_BOLD
        win.addnstr(y, 2, "Model overview", max(10, width - 4), attr)
        line_count = 1
        wrap_width = max(10, width - 6)
        for line in memory_utils.profile_summary_lines(profile):
            for wrapped in textwrap.wrap(line, wrap_width):
                win.addnstr(y + line_count, 6, wrapped, max(10, width - 8), curses.A_DIM)
                line_count += 1
        return line_count

    def height(self, width: int, state: dict) -> int:
        if bool(state.get("_ui_compact", False)) and state.get("_ui_selected_key") != self.key:
            return 1
        profile = memory_utils.estimate_memory_profile(self._state)
        wrap_width = max(10, width - 6)
        lines = 1
        for line in memory_utils.profile_summary_lines(profile):
            lines += len(textwrap.wrap(line, wrap_width))
        return lines


class MemoryVisualizerOption(OptionBase):
    def __init__(self, state: dict, *, visible: Callable[[dict], bool] | None = None) -> None:
        super().__init__(visible=visible)
        self._state = state
        self.key = "_memory_planner"
        self.name = "Memory planner"
        self.description = "Visual breakdown of GPU and CPU usage."

    @staticmethod
    def _draw_bar(
        win: "curses._CursesWindow",
        y: int,
        x: int,
        width: int,
        weights: int,
        kv: int,
        total: Optional[int],
    ) -> None:
        bar_width = max(1, width)
        total_capacity = total if total and total > 0 else max(weights + kv, 1)
        weight_ratio = min(max(weights / total_capacity, 0.0), 1.0)
        kv_ratio = min(max(kv / total_capacity, 0.0), 1.0)
        ratio_sum = weight_ratio + kv_ratio
        if ratio_sum > 1.0:
            weight_ratio /= ratio_sum
            kv_ratio /= ratio_sum
        weight_chars = int(round(weight_ratio * bar_width))
        kv_chars = int(round(kv_ratio * bar_width))
        while weight_chars + kv_chars > bar_width:
            if kv_chars >= weight_chars and kv_chars > 0:
                kv_chars -= 1
            elif weight_chars > 0:
                weight_chars -= 1
            else:
                break
        unused_chars = max(bar_width - weight_chars - kv_chars, 0)
        pos = x
        if weight_chars:
            win.addnstr(y, pos, "█" * weight_chars, weight_chars, curses.color_pair(3) | curses.A_BOLD)
            pos += weight_chars
        if kv_chars:
            win.addnstr(y, pos, "▓" * kv_chars, kv_chars, curses.color_pair(4) | curses.A_BOLD)
            pos += kv_chars
        if unused_chars:
            win.addnstr(y, pos, "░" * unused_chars, unused_chars, curses.A_DIM)

    def handle_key(
        self,
        key: int,
        state: dict,
        stdscr: "curses._CursesWindow",
    ) -> tuple[bool, str | None]:
        if key in (ord("r"), ord("R")):
            memory_utils.clear_cached_profile(self._state)
            return False, "Memory snapshot refreshed"
        return False, None

    def render(
        self,
        win: "curses._CursesWindow",
        y: int,
        width: int,
        selected: bool,
        state: dict,
    ) -> int:
        if bool(state.get("_ui_compact", False)) and not selected:
            attr = curses.A_REVERSE if selected else curses.A_BOLD
            win.addnstr(y, 2, "Memory planner", max(10, width - 4), attr)
            return 1
        profile = memory_utils.estimate_memory_profile(self._state)
        attr = curses.A_REVERSE if selected else curses.A_BOLD
        win.addnstr(y, 2, "Memory planner", max(10, width - 4), attr)
        line_count = 1
        legend = "Legend: █ weights  ▓ KV cache  ░ free"
        win.addnstr(y + line_count, 6, legend, max(10, width - 8), curses.A_DIM)
        line_count += 1
        usable_width = max(10, width - 18)

        if profile.gpus:
            for gpu in profile.gpus:
                label = f"GPU{gpu.info.index} {gpu.info.name}"
                if gpu.info.total:
                    label += f" • {memory_utils.format_bytes(gpu.info.total)} total"
                win.addnstr(y + line_count, 4, label, max(10, width - 6), curses.A_NORMAL)
                line_count += 1
                bar_y = y + line_count
                bar_x = 6
                bar_width = min(40, usable_width)
                total_mem = gpu.info.total if gpu.info.total and gpu.info.total > 0 else None
                self._draw_bar(win, bar_y, bar_x, bar_width, gpu.weights, gpu.kv, total_mem)
                used = gpu.weights + gpu.kv
                desc = f"{memory_utils.format_bytes(used)} used (W {memory_utils.format_bytes(gpu.weights)} / KV {memory_utils.format_bytes(gpu.kv)})"
                if gpu.info.free is not None and gpu.info.total:
                    free_after = max(gpu.info.free - used, 0)
                    desc += f" • free ≈{memory_utils.format_bytes(free_after)}"
                win.addnstr(bar_y, bar_x + bar_width + 2, desc, max(10, width - (bar_x + bar_width + 4)), curses.A_DIM)
                line_count += 1
        else:
            win.addnstr(y + line_count, 4, "No CUDA GPUs detected. CPU-only planning shown below.", max(10, width - 6), curses.color_pair(2) | curses.A_DIM)
            line_count += 1

        cpu_label = "System RAM"
        if profile.cpu_total:
            cpu_label += f" • {memory_utils.format_bytes(profile.cpu_total)} total"
        win.addnstr(y + line_count, 4, cpu_label, max(10, width - 6), curses.A_NORMAL)
        line_count += 1
        bar_y = y + line_count
        bar_x = 6
        bar_width = min(40, usable_width)
        cpu_total = profile.cpu_total if profile.cpu_total and profile.cpu_total > 0 else None
        cpu_used = profile.cpu_weights + profile.cpu_kv
        self._draw_bar(win, bar_y, bar_x, bar_width, profile.cpu_weights, profile.cpu_kv, cpu_total)
        cpu_desc = f"{memory_utils.format_bytes(cpu_used)} used (W {memory_utils.format_bytes(profile.cpu_weights)} / KV {memory_utils.format_bytes(profile.cpu_kv)})"
        if profile.cpu_available:
            free_cpu = max(profile.cpu_available - cpu_used, 0)
            cpu_desc += f" • free ≈{memory_utils.format_bytes(free_cpu)}"
        win.addnstr(bar_y, bar_x + bar_width + 2, cpu_desc, max(10, width - (bar_x + bar_width + 4)), curses.A_DIM)
        line_count += 1

        for warn in memory_utils.warning_lines(profile):
            win.addnstr(y + line_count, 4, f"⚠ {warn}", max(10, width - 6), curses.color_pair(2) | curses.A_BOLD)
            line_count += 1

        return line_count

    def height(self, width: int, state: dict) -> int:
        if bool(state.get("_ui_compact", False)) and state.get("_ui_selected_key") != self.key:
            return 1
        profile = memory_utils.estimate_memory_profile(self._state)
        lines = 2  # title + legend
        if profile.gpus:
            lines += 2 * len(profile.gpus)
        else:
            lines += 1
        lines += 2  # CPU label + bar
        warnings = list(memory_utils.warning_lines(profile))
        lines += len(warnings)
        return lines


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
    mark_profile_dirty(state)
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
    state["model_source"] = "local"
    mark_profile_dirty(state)


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
            kind, _ratios, err = memory_utils.parse_tensor_split(tensor_split)
            if kind == "invalid":
                raise ValueError(err or "Invalid --tensor-split value.")
            if kind == "auto":
                resolved = memory_utils.auto_tensor_split()
                if resolved:
                    cmd += ["--tensor-split", resolved]
            else:
                cmd += ["--tensor-split", tensor_split]
        ctx_size = (state.get("ctx_size") or "").strip()
        if ctx_size:
            ctx_int = parse_int(ctx_size, name="--ctx-size")
            cmd += ["--ctx-size", str(ctx_int)]
        if state.get("cpu_moe"):
            cmd.append("--cpu-moe")
        else:
            n_cpu_moe = (state.get("n_cpu_moe") or "").strip()
            if n_cpu_moe:
                cmd += ["--n-cpu-moe", str(parse_int(n_cpu_moe, name="--n-cpu-moe"))]
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

    curses.def_prog_mode()
    curses.endwin()
    print("Launching loadmodel:\n  " + shell_join(cmd), flush=True)
    try:
        subprocess.call(cmd)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            curses.reset_prog_mode()
            curses.curs_set(0)
            curses.doupdate()
        except curses.error:
            pass
    return True, None


def action_refresh(state: dict) -> tuple[bool, str | None]:
    return refresh_local_models(state)


def ensure_status(state: dict) -> str:
    return str(state.get("status") or "")


def set_status(state: dict, message: str | None) -> None:
    state["status"] = message or ""


def mark_profile_dirty(state: dict) -> None:
    memory_utils.clear_cached_profile(state)


def on_models_dir_change(state: dict, _new: str) -> None:
    mark_profile_dirty(state)
    _, message = refresh_local_models(state)
    set_status(state, message)


def on_model_ref_change(state: dict, _new: str) -> None:
    mark_profile_dirty(state)


def on_numeric_change(state: dict, _new: str) -> None:
    mark_profile_dirty(state)


def on_tensor_split_change(state: dict, new: str) -> None:
    mark_profile_dirty(state)
    raw = str(new or "").strip()
    if not raw:
        return
    kind, ratios, err = memory_utils.parse_tensor_split(raw)
    if kind == "invalid":
        set_status(state, err)
        return
    if kind == "ratios" and ratios:
        gpus = memory_utils.detect_gpus()
        if gpus and len(gpus) > 1 and len(ratios) != len(gpus):
            set_status(state, f"Note: detected {len(gpus)} GPU(s) but tensor split has {len(ratios)} value(s).")


def on_model_source_change(state: dict, choice: ChoiceItem) -> None:
    mark_profile_dirty(state)
    if choice.value == "remote":
        state["selected_local_model"] = ""
        state["model_ref"] = str(state.get("model_ref") or "").strip()
        if state["model_ref"] and Path(state["model_ref"]).exists():
            state["model_ref"] = ""
    else:
        selected = (state.get("selected_local_model") or "").strip()
        if selected:
            set_model_from_choice(state, ChoiceItem(selected, selected))


def build_gpu_strategy_choices(state: dict) -> List[ChoiceItem]:
    gpus = memory_utils.detect_gpus()
    count = len(gpus)
    gpu_missing_reason = "CUDA GPU not detected"
    items = [
        ChoiceItem("balanced", "Even split across GPUs", enabled=count > 1, reason=None if count > 1 else "Requires >=2 GPUs"),
        ChoiceItem(
            "vram",
            "Split by detected VRAM",
            enabled=count > 1,
            reason=None if count > 1 else "Requires >=2 GPUs",
        ),
        ChoiceItem("priority", "Prioritise GPU 0", enabled=count > 1, reason=None if count > 1 else "Requires >=2 GPUs"),
        ChoiceItem("auto", "Auto split (VRAM)", enabled=count > 1, reason=None if count > 1 else "Requires >=2 GPUs"),
        ChoiceItem("single", "Single GPU", enabled=count >= 1, reason=None if count >= 1 else gpu_missing_reason),
        ChoiceItem("cpu", "Offload to system RAM", enabled=True),
    ]
    if count == 0:
        for item in items[:-1]:
            item.enabled = False
            item.reason = gpu_missing_reason
    return items


def apply_gpu_strategy(state: dict, choice: ChoiceItem) -> None:
    strategy = choice.value
    mark_profile_dirty(state)
    if strategy == "cpu":
        state["tensor_split"] = ""
        state["n_gpu_layers"] = "0"
        return
    gpus = memory_utils.detect_gpus()
    if not gpus:
        set_status(state, "No CUDA GPUs detected; falling back to system RAM.")
        state["gpu_strategy"] = "cpu"
        state["tensor_split"] = ""
        state["n_gpu_layers"] = "0"
        return
    count = len(gpus)
    if strategy == "single" or count == 1:
        state["tensor_split"] = ""
        state["n_gpu_layers"] = "999"
        return
    if strategy == "auto":
        state["tensor_split"] = "auto"
        state["n_gpu_layers"] = "999"
        return
    if strategy == "priority":
        primary = 60
        parts = [primary]
        remaining = 100 - primary
        others = max(count - 1, 1)
        base = remaining // others
        for idx in range(others):
            share = base
            if idx < remaining - base * others:
                share += 1
            parts.append(share)
    elif strategy == "vram":
        totals = [float(gpu.total) if gpu.total and gpu.total > 0 else 1.0 for gpu in gpus]
        denom = sum(totals)
        if denom <= 0:
            base = 100 // count
            parts = [base for _ in range(count)]
            for idx in range(100 - base * count):
                parts[idx % count] += 1
        else:
            raw = [(total / denom) * 100.0 for total in totals]
            parts = [int(x) for x in raw]
            remainder = 100 - sum(parts)
            order = sorted(((raw[i] - parts[i], i) for i in range(count)), reverse=True)
            for offset in range(remainder):
                parts[order[offset % count][1]] += 1
    else:  # balanced
        base = 100 // count
        parts = [base for _ in range(count)]
        for idx in range(100 - base * count):
            parts[idx % count] += 1
    total = sum(parts)
    if total != 100 and total > 0:
        parts[-1] += 100 - total
    state["tensor_split"] = ",".join(str(max(p, 0)) for p in parts)
    state["n_gpu_layers"] = "999"

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
            on_change=lambda st, _choice: mark_profile_dirty(st),
        )
    )

    options.append(
        InputOption(
            key="models_dir",
            name="Models directory",
            description="Directory that stores local GGUF files. Used for browsing as well as download destination.",
            state=state,
            placeholder=str(loadmodel.MODELS_DIR),
            on_change=on_models_dir_change,
        )
    )

    options.append(
        ChoiceOption(
            key="model_source",
            name="Model source",
            description="Choose whether to launch a local GGUF file or reference a remote Hugging Face model.",
            state=state,
            choices=[
                ChoiceItem("local", "Local GGUF"),
                ChoiceItem("remote", "Remote (Hugging Face)")
            ],
            on_change=on_model_source_change,
            visible=lambda st: st.get("mode") in {"llm", "embed"},
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
            visible=lambda st: st.get("mode") in {"llm", "embed"} and st.get("model_source", "local") == "local",
        )
    )

    options.append(
        ActionOption(
            name="Refresh local models",
            description="Rescan the models directory for GGUF files.",
            action=action_refresh,
            visible=lambda st: st.get("mode") in {"llm", "embed"} and st.get("model_source", "local") == "local",
        )
    )

    options.append(
        InputOption(
            key="model_ref",
            name="Local model path",
            description="Absolute or relative path to a GGUF file. Auto-filled when selecting from the local list.",
            state=state,
            placeholder="Auto from selection or enter ./models/model.gguf",
            on_change=on_model_ref_change,
            visible=lambda st: st.get("mode") in {"llm", "embed"} and st.get("model_source", "local") == "local",
        )
    )

    options.append(
        InputOption(
            key="model_ref",
            name="Remote reference",
            description="Path or Hugging Face reference (org/repo:quant or org/repo:file.gguf). Required before launching.",
            state=state,
            placeholder="Qwen/Qwen2-7B-Instruct:Q4_K_M",
            on_change=on_model_ref_change,
            visible=lambda st: st.get("mode") in {"llm", "embed"} and st.get("model_source", "local") == "remote",
        )
    )

    options.append(
        ModelSummaryOption(
            state=state,
            visible=lambda st: st.get("mode") in {"llm", "embed"},
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
            on_change=on_numeric_change,
            visible=lambda st: st.get("mode") in {"llm", "embed"},
        )
    )

    options.append(
        InputOption(
            key="tensor_split",
            name="--tensor-split",
            description="Comma-separated ratios for multiple GPUs (e.g. 0.6,0.4 or 60,40). Use 'auto' to split by detected VRAM.",
            state=state,
            placeholder="",
            on_change=on_tensor_split_change,
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
            on_change=on_numeric_change,
            visible=lambda st: st.get("mode") in {"llm", "embed"},
        )
    )

    options.append(
        ToggleOption(
            key="cpu_moe",
            name="--cpu-moe",
            description="Offload all Mixture-of-Experts layers to CPU (llama.cpp). Overrides --n-cpu-moe when enabled.",
            state=state,
            visible=lambda st: st.get("mode") in {"llm", "embed"},
        )
    )

    options.append(
        InputOption(
            key="n_cpu_moe",
            name="--n-cpu-moe",
            description="Number of initial MoE layers whose experts should be offloaded to CPU. Leave blank to keep experts on GPU.",
            state=state,
            placeholder="",
            on_change=on_numeric_change,
            visible=lambda st: st.get("mode") in {"llm", "embed"} and not st.get("cpu_moe"),
        )
    )

    options.append(
        ChoiceOption(
            key="gpu_strategy",
            name="GPU memory strategy",
            description="Controls how llama.cpp spreads model weights across GPUs or system RAM.",
            state=state,
            choices_fn=build_gpu_strategy_choices,
            on_change=apply_gpu_strategy,
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
        MemoryVisualizerOption(
            state=state,
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


def _option_detail_lines(opt: OptionBase, state: dict) -> List[tuple[str, int]]:
    lines: List[tuple[str, int]] = []
    description = getattr(opt, "description", "")
    if description:
        lines.append((description, curses.A_NORMAL))
    if isinstance(opt, ChoiceOption):
        current = opt.current
        lines.append(("", 0))
        lines.append((f"Current: {current.label}", curses.A_DIM))
        lines.append(("Left/Right: change selection", curses.A_DIM))
        if not current.enabled and current.reason:
            lines.append((f"Note: {current.reason}", curses.A_DIM))
    elif isinstance(opt, InputOption):
        value = str(state.get(opt.key, "")).strip()
        lines.append(("", 0))
        lines.append((f"Current: {value or '(blank)'}", curses.A_DIM))
        if opt.key == "tensor_split" and value:
            kind, ratios, err = memory_utils.parse_tensor_split(value)
            if kind == "ratios" and ratios:
                denom = sum(ratios) or 1.0
                percents = [(r / denom) * 100.0 for r in ratios]
                gpus = memory_utils.detect_gpus()
                if gpus and len(gpus) == len(percents):
                    lines.append(("Normalised split:", curses.A_DIM))
                    for idx, pct in enumerate(percents):
                        lines.append((f"GPU{idx}: {pct:.1f}% ({gpus[idx].name})", curses.A_DIM))
                else:
                    joined = ", ".join(f"{pct:.1f}%" for pct in percents)
                    lines.append((f"Normalised: {joined}", curses.A_DIM))
            elif kind == "auto":
                lines.append(("Mode: auto (estimated from VRAM)", curses.A_DIM))
            elif kind == "invalid" and err:
                lines.append((f"⚠ {err}", curses.A_DIM))
        lines.append(("Enter: edit (Enter saves, Esc cancels)", curses.A_DIM))
    elif isinstance(opt, ToggleOption):
        enabled = bool(state.get(opt.key))
        lines.append(("", 0))
        lines.append((f"State: {'enabled' if enabled else 'disabled'}", curses.A_DIM))
        lines.append(("Space/Enter: toggle", curses.A_DIM))
    elif isinstance(opt, ActionOption):
        lines.append(("", 0))
        lines.append(("Enter: run this action", curses.A_DIM))
    elif isinstance(opt, ModelSummaryOption):
        lines.append(("", 0))
        profile = memory_utils.estimate_memory_profile(state)
        for line in memory_utils.profile_summary_lines(profile):
            lines.append((line, curses.A_DIM))
    elif isinstance(opt, MemoryVisualizerOption):
        lines.append(("", 0))
        lines.append(("Press 'r' to refresh the memory snapshot.", curses.A_DIM))
        lines.append(("Bars show weights (█) and KV cache (▓) usage per device.", curses.A_DIM))
    if not lines:
        lines.append(("No details available.", curses.A_DIM))
    return lines


def prepare_help_panel(
    selected_opt: OptionBase | None,
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


def _draw_wide_layout(
    stdscr: "curses._CursesWindow",
    visible_options: List[tuple[OptionBase, int]],
    selected_index: int,
    scroll: int,
    state: dict,
    ui_state: dict,
    status: str,
    height: int,
    width: int,
) -> tuple[int | None, int | None]:
    title_offset = 2
    body_top = title_offset
    status_y = max(0, height - 2)
    instructions_y = max(0, height - 1)
    body_height = max(0, status_y - body_top)
    left_margin = 2
    column_gap = 2
    min_left_width = 24
    min_right_width = 28

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
        return (None, None)

    total_visible = len(visible_options)
    start = max(0, min(scroll, max(0, total_visible - 1)))
    y = body_top
    first_rendered: int | None = None
    last_rendered: int | None = None
    first_render_row: int | None = None
    last_render_row: int | None = None
    for visible_idx in range(start, total_visible):
        opt, _ = visible_options[visible_idx]
        opt_height = opt.height(left_width, state)
        if y >= status_y:
            break
        if opt_height > body_height and first_rendered is not None:
            break
        render_start = y
        if y + opt_height > status_y:
            if first_rendered is None:
                extra = opt.render(stdscr, y, left_width, visible_idx == selected_index, state)
                first_render_row = render_start
                y += extra
                first_rendered = visible_idx
                last_rendered = visible_idx
                last_render_row = y - 1
            break
        extra = opt.render(stdscr, y, left_width, visible_idx == selected_index, state)
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
    if status:
        for line in textwrap.wrap(status, wrap_width):
            if right_y >= status_y:
                break
            stdscr.addnstr(right_y, right_start, line, wrap_width, curses.A_BOLD | curses.color_pair(2))
            right_y += 1
        if right_y < status_y:
            right_y += 1

    selected_opt: OptionBase | None = None
    if 0 <= selected_index < len(visible_options):
        selected_opt = visible_options[selected_index][0]

    focus_area = ui_state.get("focus_area", 0)
    help_focus = focus_area == 1

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

    draw_logs(stdscr, ui_state, max(body_top, status_y - 3), width, height)

    return (first_rendered, last_rendered)


def _draw_tablet_layout(
    stdscr: "curses._CursesWindow",
    visible_options: List[tuple[OptionBase, int]],
    selected_index: int,
    scroll: int,
    state: dict,
    ui_state: dict,
    status: str,
    height: int,
    width: int,
) -> tuple[int | None, int | None]:
    title_offset = 2
    body_top = title_offset
    status_y = max(0, height - 2)
    instructions_y = max(0, height - 1)
    body_height = max(0, status_y - body_top)
    if body_height <= 0:
        warning = "Not enough rows to render options. Increase terminal height."
        stdscr.addnstr(body_top, 2, warning[: max(1, width - 4)], max(1, width - 4), curses.A_BOLD | curses.color_pair(2))
        return (None, None)

    available_width = width - 4
    column_gap = 3
    column_count = 2 if available_width >= 40 else 1
    if column_count <= 1:
        column_gap = 0
    column_width = max(20, (available_width - column_gap * (column_count - 1)) // column_count)
    x_positions = [2 + (column_width + column_gap) * idx for idx in range(column_count)]
    y_positions = [body_top] * column_count

    total_visible = len(visible_options)
    start = max(0, min(scroll, max(0, total_visible - 1)))
    first_rendered: int | None = None
    last_rendered: int | None = None
    first_render_row: int | None = None
    last_render_row: int | None = None

    idx = start
    current_col = 0
    while idx < total_visible:
        opt, _ = visible_options[idx]
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
        extra = opt.render(stdscr, draw_y, column_width, idx == selected_index, state)
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

    selected_opt: OptionBase | None = None
    if 0 <= selected_index < len(visible_options):
        selected_opt = visible_options[selected_index][0]

    focus_area = ui_state.get("focus_area", 0)
    help_focus = focus_area == 1

    help_start = body_top + body_height
    if help_start < status_y:
        available_rows = status_y - (help_start + 1)
        header_text, visible_help, has_selection = prepare_help_panel(
            selected_opt,
            state,
            max(10, width - 4),
            available_rows,
            ui_state,
            height,
        )
        header_attr = curses.A_BOLD | (curses.A_REVERSE if help_focus else curses.A_DIM)
        stdscr.hline(help_start, 1, curses.ACS_HLINE, width - 2)
        heading_text = f" {STRINGS['help_heading']}: {header_text} "
        stdscr.addnstr(help_start, 3, heading_text[: max(1, width - 6)], max(1, width - 6), header_attr)
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
    draw_logs(stdscr, ui_state, max(body_top, status_y - 3), width, height)
    if instructions_y >= 0:
        instructions = STRINGS["instructions"]
        stdscr.addnstr(instructions_y, 2, instructions[: max(1, width - 4)], max(1, width - 4), curses.A_DIM)

    return (first_rendered, last_rendered)


def draw_screen(
    stdscr: "curses._CursesWindow",
    visible_options: List[tuple[OptionBase, int]],
    selected_index: int,
    scroll: int,
    state: dict,
    ui_state: dict,
) -> tuple[int | None, int | None]:
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    if height < 5 or width < 20:
        warning = "Terminal too small for loadmodel launcher. Please enlarge."
        stdscr.addnstr(0, 0, warning[: max(1, width - 1)], max(1, width - 1), curses.A_BOLD)
        stdscr.refresh()
        return (None, None)

    title = STRINGS["title"]
    stdscr.addnstr(0, 2, title, max(10, width - 4), curses.A_BOLD)

    body_top = 2
    instructions = STRINGS["instructions"]
    status = ensure_status(state)
    instructions_y = max(0, height - 1)
    status_y = max(0, instructions_y - 1)
    if width >= 140:
        first_rendered, last_rendered = _draw_wide_layout(
            stdscr,
            visible_options,
            selected_index,
            scroll,
            state,
            ui_state,
            status,
            height,
            width,
        )
        if status and status_y >= 0:
            stdscr.addnstr(status_y, 2, status[: max(1, width - 4)], max(1, width - 4), curses.color_pair(2) | curses.A_BOLD)
        if instructions_y >= 0:
            stdscr.addnstr(instructions_y, 2, instructions[: max(1, width - 4)], max(1, width - 4), curses.A_DIM)
        stdscr.refresh()
        return (first_rendered, last_rendered)
    if width >= 80:
        first_rendered, last_rendered = _draw_tablet_layout(
            stdscr,
            visible_options,
            selected_index,
            scroll,
            state,
            ui_state,
            status,
            height,
            width,
        )
        stdscr.refresh()
        return (first_rendered, last_rendered)

    body_bottom_exclusive = status_y
    body_height = max(0, body_bottom_exclusive - body_top)

    y = body_top
    first_rendered: int | None = None
    last_rendered: int | None = None
    first_render_row: int | None = None
    last_render_row: int | None = None
    start = 0
    if body_height <= 0:
        warning = "Not enough rows to render options. Increase terminal height."
        stdscr.addnstr(y, 2, warning[: max(1, width - 4)], max(1, width - 4), curses.A_BOLD | curses.color_pair(2))
    else:
        total_visible = len(visible_options)
        start = max(0, min(scroll, max(0, total_visible - 1)))
        for idx in range(start, total_visible):
            opt, opt_height = visible_options[idx]
            if opt_height <= 0:
                continue
            if y >= body_bottom_exclusive:
                break
            if opt_height > body_height and first_rendered is not None:
                break
            render_start = y
            if y + opt_height > body_bottom_exclusive:
                if first_rendered is None:
                    lines = opt.render(stdscr, y, width - 4, idx == selected_index, state)
                    y += lines
                    first_rendered = last_rendered = idx
                    first_render_row = render_start
                    last_render_row = y - 1
                break
            lines = opt.render(stdscr, y, width - 4, idx == selected_index, state)
            if first_render_row is None:
                first_render_row = render_start
            y += lines
            if first_rendered is None:
                first_rendered = idx
            last_rendered = idx
            last_render_row = y - 1
            if y >= body_bottom_exclusive:
                break
            y += 1
        if first_rendered is None:
            last_rendered = None

    has_more_above = body_height > 0 and start > 0
    has_more_below = (
        body_height > 0
        and last_rendered is not None
        and last_rendered < len(visible_options) - 1
    )
    arrow_attr = curses.A_DIM | curses.A_BOLD
    if has_more_above and first_render_row is not None:
        arrow_row = max(body_top, min(first_render_row, height - 1))
        stdscr.addnstr(arrow_row, 0, "↑", 1, arrow_attr)
    if has_more_below and last_render_row is not None:
        arrow_row = max(body_top, min(last_render_row, height - 1))
        stdscr.addnstr(arrow_row, 0, "↓", 1, arrow_attr)
    selected_opt: OptionBase | None = None
    if 0 <= selected_index < len(visible_options):
        selected_opt = visible_options[selected_index][0]

    focus_area = ui_state.get("focus_area", 0)
    help_focus = focus_area == 1
    help_start = max(y, body_top)
    if help_start < status_y:
        available_rows = status_y - (help_start + 1)
        header_text, visible_help, has_selection = prepare_help_panel(
            selected_opt,
            state,
            max(10, width - 4),
            available_rows,
            ui_state,
            height,
        )
        header_attr = curses.A_BOLD | (curses.A_REVERSE if help_focus else curses.A_DIM)
        stdscr.hline(help_start, 1, curses.ACS_HLINE, width - 2)
        heading_text = f" {STRINGS['help_heading']}: {header_text} "
        stdscr.addnstr(help_start, 3, heading_text[: max(1, width - 6)], max(1, width - 6), header_attr)
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
    if instructions_y >= 0:
        stdscr.addnstr(instructions_y, 2, instructions[: max(1, width - 4)], max(1, width - 4), curses.A_DIM)
    draw_logs(stdscr, ui_state, max(body_top, status_y - 3), width, height)
    stdscr.refresh()
    return (first_rendered, last_rendered)


def run_tui(stdscr: "curses._CursesWindow") -> None:
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN, -1)
    curses.init_pair(2, curses.COLOR_RED, -1)
    curses.init_pair(3, curses.COLOR_WHITE, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_BLUE, curses.COLOR_BLACK)

    state: dict = {
        "mode": "llm",
        "models_dir": str(loadmodel.MODELS_DIR),
        "model_source": "local",
        "hf_token": os.environ.get("HF_TOKEN", ""),
        "host": "127.0.0.1",
        "port": "45540",
        "n_gpu_layers": "999",
        "tensor_split": "",
        "ctx_size": "",
        "extra_flags": "",
        "gpu_strategy": "balanced",
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

    saved_values = load_saved_state()
    if saved_values:
        state.update(saved_values)

    ui_state = {
        "focus_area": 0,
        "show_full_help": False,
        "compact": True,
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

    gpu_infos = memory_utils.detect_gpus()
    saved_has_strategy = bool(saved_values) and "gpu_strategy" in saved_values
    if not gpu_infos:
        state["gpu_strategy"] = "cpu"
        state["n_gpu_layers"] = "0"
        append_log(ui_state, "CUDA GPU not detected; defaulting to CPU offload.")
    elif len(gpu_infos) == 1 and not saved_has_strategy:
        state["gpu_strategy"] = "single"
    elif len(gpu_infos) > 1 and not saved_has_strategy:
        state["gpu_strategy"] = "balanced"

    _exit, initial_msg = refresh_local_models(state)
    if initial_msg:
        set_status(state, initial_msg)
        append_log(ui_state, initial_msg)

    options = build_options(state)
    selected_index = 0
    scroll = 0

    while True:
        height, width = stdscr.getmaxyx()
        body_width = max(10, width - 4)

        visible_opts: List[OptionBase] = [opt for opt in options if opt.visible(state)]
        visible_entries: List[tuple[OptionBase, int]] = []

        if not visible_opts:
            stdscr.erase()
            stdscr.addnstr(0, 0, "No options available.", max(10, width - 4), curses.A_BOLD | curses.color_pair(2))
            stdscr.addnstr(1, 0, "Press 'q' to quit.", max(10, width - 4), curses.A_DIM)
            stdscr.refresh()
            key = stdscr.getch()
            if key in (ord("q"), ord("Q")):
                save_state(options)
                break
            if key == 9:
                ui_state["focus_area"] = (ui_state.get("focus_area", 0) + 1) % 3
            continue

        selected_index = max(0, min(selected_index, len(visible_opts) - 1))
        scroll = max(0, min(scroll, max(0, len(visible_opts) - 1)))

        state["_ui_compact"] = bool(ui_state.get("compact", True))
        state["_ui_selected_key"] = getattr(visible_opts[selected_index], "key", "")

        for opt in visible_opts:
            try:
                opt_height = max(1, opt.height(body_width, state))
            except Exception:
                opt_height = 1
            visible_entries.append((opt, opt_height))

        if ui_state.get("last_selected") != selected_index:
            ui_state["help_offset"] = 0
            ui_state["last_selected"] = selected_index

        if ui_state.get("status_message") is not None:
            set_status(state, ui_state["status_message"])

        first_rendered, last_rendered = draw_screen(
            stdscr,
            visible_entries,
            selected_index,
            scroll,
            state,
            ui_state,
        )
        ui_state["status_message"] = None

        if first_rendered is not None and selected_index < first_rendered:
            scroll = selected_index
            continue
        if last_rendered is not None and selected_index > last_rendered:
            span = last_rendered - first_rendered if first_rendered is not None else 0
            if span < 0:
                span = 0
            scroll = max(0, selected_index - span)
            continue

        key = stdscr.getch()
        if key in (ord("q"), ord("Q")):
            save_state(options)
            break
        if key in (curses.KEY_RESIZE,):
            continue
        if key == 9:
            ui_state["focus_area"] = (ui_state.get("focus_area", 0) + 1) % 3
            continue
        if key == ord("?"):
            ui_state["show_full_help"] = not ui_state.get("show_full_help", False)
            ui_state["help_offset"] = 0
            append_log(ui_state, "Expanded help" if ui_state["show_full_help"] else "Collapsed help")
            continue
        if key in (ord("c"), ord("C")):
            ui_state["compact"] = not ui_state.get("compact", True)
            ui_state["help_offset"] = 0
            append_log(ui_state, "Compact list enabled" if ui_state["compact"] else "Compact list disabled")
            continue

        focus_area = ui_state.get("focus_area", 0)

        if focus_area == 0:
            if key in (curses.KEY_DOWN, ord("j")):
                selected_index = (selected_index + 1) % len(visible_opts)
                continue
            if key in (curses.KEY_UP, ord("k")):
                selected_index = (selected_index - 1) % len(visible_opts)
                continue
            if key in (curses.KEY_PPAGE,):
                scroll = max(0, scroll - max(1, len(visible_opts) // 2))
                continue
            if key in (curses.KEY_NPAGE,):
                scroll = min(len(visible_opts) - 1, scroll + max(1, len(visible_opts) // 2))
                continue

            exit_requested, message = visible_opts[selected_index].handle_key(key, state, stdscr)
            if message:
                set_status(state, message)
                append_log(ui_state, message)
            else:
                set_status(state, "")
            if exit_requested:
                save_state(options)
                break
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
                exit_requested, message = visible_opts[selected_index].handle_key(key, state, stdscr)
                if message:
                    set_status(state, message)
                    append_log(ui_state, message)
                if exit_requested:
                    save_state(options)
                    break
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
                exit_requested, message = visible_opts[selected_index].handle_key(key, state, stdscr)
                if message:
                    set_status(state, message)
                    append_log(ui_state, message)
                if exit_requested:
                    save_state(options)
                    break
            continue

    save_state(options)


def main() -> None:
    try:
        curses.wrapper(run_tui)
    except curses.error as exc:
        if "endwin" not in str(exc):
            raise


if __name__ == "__main__":
    main()
