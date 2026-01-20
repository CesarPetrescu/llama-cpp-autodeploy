from __future__ import annotations

import curses
import json
import time
import textwrap
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from constants import LAYOUT, UI
from state_utils import StrictStateMixin


class ErrorSeverity(Enum):
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


@dataclass
class AppError:
    message: str
    severity: ErrorSeverity = ErrorSeverity.ERROR
    recoverable: bool = True


@dataclass
class UIState(StrictStateMixin):
    focus_area: int = 0
    show_full_help: bool = False
    compact: bool = True
    logs: deque = field(default_factory=lambda: deque(maxlen=UI.LOG_HISTORY))
    status_message: str | None = None
    help_offset: int = 0
    logs_offset: int = 0
    help_max_offset: int = 0
    logs_max_offset: int = 0
    help_visible_lines: int = 0
    logs_visible_rows: int = 0
    help_total_lines: int = 0
    last_selected: int | None = None
    scanning: bool = False
    scan_message: str = ""
    scan_start: float | None = None


def handle_error(error: AppError, ui_state: UIState | None = None) -> str:
    label = error.severity.value.upper()
    message = f"[{label}] {error.message}"
    if ui_state is not None:
        append_log(ui_state, message)
    if error.severity == ErrorSeverity.FATAL and not error.recoverable:
        raise SystemExit(1)
    return message


def append_log(ui_state: UIState, message: str) -> None:
    if not message:
        return
    logs = ui_state.logs
    timestamp = time.strftime("%H:%M:%S")
    logs.append(f"[{timestamp}] {message}")
    ui_state.status_message = message
    if ui_state.focus_area != 2:
        ui_state.logs_offset = 0


def load_saved_state(config_path: Path, ui_state: UIState | None = None) -> dict:
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, dict):
                return data
    except FileNotFoundError:
        return {}
    except Exception as exc:
        handle_error(AppError(f"Failed to load saved state: {exc}", ErrorSeverity.WARNING), ui_state)
        return {}
    return {}


def collect_option_values(options: Iterable, state: object | None = None) -> dict:
    payload: dict = {}
    for opt in options:
        try:
            if state is None:
                payload[opt.key] = opt.get_value()
            else:
                payload[opt.key] = opt.get_value(state)
        except Exception:
            continue
    return payload


def save_state(config_path: Path, options: Iterable, state: object | None = None, ui_state: UIState | None = None) -> None:
    payload = collect_option_values(options, state)
    try:
        config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        handle_error(AppError(f"Failed to save state: {exc}", ErrorSeverity.WARNING), ui_state)


def apply_saved_values(options: Iterable, saved: dict, state: object | None = None) -> None:
    if not saved:
        return
    for opt in options:
        if opt.key in saved:
            try:
                if state is None:
                    opt.set_value(saved[opt.key])
                else:
                    opt.set_value(state, saved[opt.key])
            except Exception:
                continue


def layout_mode(width: int) -> str:
    if width >= LAYOUT.WIDE_MIN_WIDTH:
        return "wide"
    if width >= LAYOUT.TABLET_MIN_WIDTH:
        return "tablet"
    return "narrow"


def format_scroll_indicator(first_index: int, total: int, visible_rows: int) -> str:
    if total <= 0 or visible_rows <= 0:
        return ""
    if total <= visible_rows:
        return ""
    current = max(1, min(total, first_index + 1))
    return f"[{current}/{total}]"


def draw_scrollbar(
    win: "curses._CursesWindow",
    *,
    top: int,
    height: int,
    x: int,
    first_index: int,
    total: int,
    visible_rows: int,
    attr: int,
) -> None:
    if total <= visible_rows or height <= 0:
        return
    max_scroll = max(1, total - visible_rows)
    track_height = max(1, height)
    thumb_pos = int((first_index / max_scroll) * (track_height - 1))
    for row in range(track_height):
        ch = "o" if row == thumb_pos else "|"
        try:
            win.addch(top + row, x, ch, attr)
        except curses.error:
            break


def draw_logs(
    stdscr: "curses._CursesWindow",
    ui_state: UIState,
    start_row: int,
    width: int,
    height: int,
    heading: str,
) -> int:
    logs = list(ui_state.logs)
    if not logs or start_row >= height:
        ui_state.logs_max_offset = 0
        ui_state.logs_visible_rows = 0
        return 0
    focus = ui_state.focus_area == 2
    available_rows = max(0, height - start_row)
    if available_rows <= 0:
        ui_state.logs_max_offset = 0
        ui_state.logs_visible_rows = 0
        return 0

    heading_attr = curses.A_BOLD | (curses.A_REVERSE if focus else curses.A_DIM)
    stdscr.addnstr(
        start_row,
        2,
        heading[: max(1, width - 4)],
        max(1, width - 4),
        heading_attr,
    )
    if available_rows == 1:
        ui_state.logs_max_offset = 0
        ui_state.logs_visible_rows = 0
        return 1

    body_rows = available_rows - 1
    max_start = max(0, len(logs) - body_rows)
    offset = min(max(0, ui_state.logs_offset), max_start)
    first_idx = max(0, max_start - offset)
    slice_logs = logs[first_idx : first_idx + body_rows]

    ui_state.logs_offset = offset
    ui_state.logs_max_offset = max_start
    ui_state.logs_visible_rows = len(slice_logs)

    indicator = format_scroll_indicator(first_idx, len(logs), body_rows)
    if indicator and width > len(indicator) + 4:
        stdscr.addnstr(
            start_row,
            max(2, width - len(indicator) - 2),
            indicator,
            len(indicator),
            curses.A_DIM,
        )

    scroll_x = width - 2
    if scroll_x >= 0:
        draw_scrollbar(
            stdscr,
            top=start_row + 1,
            height=body_rows,
            x=scroll_x,
            first_index=first_idx,
            total=len(logs),
            visible_rows=body_rows,
            attr=curses.A_DIM,
        )

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


def wrap_help_lines(detail_pairs: Sequence[Tuple[str, int]], wrap_width: int) -> List[Tuple[str, int]]:
    full_help_lines: List[Tuple[str, int]] = []
    for raw, attr in detail_pairs:
        style = attr or curses.A_NORMAL
        if raw == "":
            full_help_lines.append(("", style))
            continue
        wrapped = textwrap.wrap(raw, wrap_width) or [""]
        for line in wrapped:
            full_help_lines.append((line, style))
    return full_help_lines


def prepare_help_panel(
    header_text: str,
    detail_pairs: Sequence[Tuple[str, int]],
    wrap_width: int,
    available_rows: int,
    ui_state: UIState,
    height: int,
    *,
    enable_preview: bool = True,
) -> tuple[str, List[Tuple[str, int]], bool]:
    if not detail_pairs:
        ui_state.help_max_offset = 0
        ui_state.help_visible_lines = 0
        ui_state.help_total_lines = 0
        ui_state.help_offset = 0
        return (header_text, [], False)

    full_help_lines = wrap_help_lines(detail_pairs, wrap_width)
    max_preview = max(UI.HELP_PREVIEW_MIN, min(UI.HELP_PREVIEW_MAX, height // 2))
    show_full_help = ui_state.show_full_help if enable_preview else True
    help_source = full_help_lines if show_full_help else full_help_lines[:max_preview]

    ui_state.help_total_lines = len(help_source)
    max_offset = max(0, len(help_source) - max(0, available_rows))
    help_offset = min(max(0, ui_state.help_offset), max_offset)
    ui_state.help_offset = help_offset
    ui_state.help_max_offset = max_offset
    visible_help = help_source[help_offset : help_offset + max(0, available_rows)]
    ui_state.help_visible_lines = len(visible_help)

    indicator = ""
    if help_offset > 0:
        indicator += "^"
    if help_offset < max_offset:
        indicator += "v"
    if enable_preview and not show_full_help and len(help_source) < len(full_help_lines):
        indicator += "+"
    if indicator:
        header_text = f"{header_text} ({indicator})"

    return (header_text, visible_help, True)
