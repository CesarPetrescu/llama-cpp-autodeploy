#!/usr/bin/env python3
"""Small curses helpers shared by the TUIs in this repo."""
from __future__ import annotations

import curses
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class LineEditResult:
    value: str
    accepted: bool


def _safe_curs_set(visibility: int) -> None:
    try:
        curses.curs_set(visibility)
    except Exception:
        pass


def edit_line_dialog(
    stdscr: "curses._CursesWindow",
    *,
    title: str,
    initial: str = "",
    instructions: str | None = None,
    max_width: int = 90,
    allow_empty: bool = True,
) -> LineEditResult:
    """Edit a single line of text.

    - Enter: accept
    - Esc: cancel
    - Ctrl+U: clear
    """
    dialog_height = 5
    field_x = 2
    field_y = 2
    title_y = 1
    instructions_y = 3

    def layout() -> tuple["curses._CursesWindow", "curses._CursesWindow", int, int]:
        h, w = stdscr.getmaxyx()
        if h < dialog_height or w < 20:
            raise RuntimeError("terminal too small")
        dialog_width = min(max_width, w)
        dialog_width = max(20, dialog_width)
        dialog_width = min(dialog_width, w)

        start_y = max(0, (h - dialog_height) // 2)
        start_x = max(0, (w - dialog_width) // 2)
        win = curses.newwin(dialog_height, dialog_width, start_y, start_x)
        win.keypad(True)
        win.border()

        clean_title = (title or "").strip() or "Enter value"
        win.addnstr(title_y, 2, clean_title, max(1, dialog_width - 4), curses.A_BOLD)

        help_line = instructions
        if help_line is None:
            help_line = "Enter: save  Esc: cancel  Ctrl+U: clear"
        win.addnstr(instructions_y, 2, help_line, max(1, dialog_width - 4), curses.A_DIM)

        field_width = max(1, dialog_width - 4)
        field = win.derwin(1, field_width, field_y, field_x)
        field.keypad(True)
        return win, field, dialog_width, field_width

    try:
        win, field, dialog_width, field_width = layout()
    except Exception:
        return LineEditResult(value=initial, accepted=False)

    buffer = list(str(initial or ""))
    cursor = len(buffer)
    scroll = 0

    _safe_curs_set(1)
    try:
        while True:
            if cursor < scroll:
                scroll = cursor
            visible_capacity = max(1, field_width - 1)
            if cursor > scroll + visible_capacity:
                scroll = cursor - visible_capacity
            if scroll < 0:
                scroll = 0

            visible = "".join(buffer[scroll : scroll + field_width])
            field.erase()
            field.addnstr(0, 0, visible.ljust(field_width), field_width)
            field.move(0, max(0, min(field_width - 1, cursor - scroll)))
            win.refresh()

            key = win.getch()
            if key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
                value = "".join(buffer)
                value = value.strip()
                if value or allow_empty:
                    return LineEditResult(value=value, accepted=True)
                return LineEditResult(value=initial, accepted=False)
            if key == 27:  # Esc
                return LineEditResult(value=initial, accepted=False)
            if key in (curses.KEY_RESIZE,):
                try:
                    win, field, dialog_width, field_width = layout()
                    scroll = min(scroll, max(0, len(buffer) - 1))
                except Exception:
                    return LineEditResult(value=initial, accepted=False)
                continue

            if key in (curses.KEY_BACKSPACE, 127, 8):
                if cursor > 0:
                    buffer.pop(cursor - 1)
                    cursor -= 1
                continue
            if key in (curses.KEY_DC,):
                if cursor < len(buffer):
                    buffer.pop(cursor)
                continue
            if key in (curses.KEY_LEFT,):
                cursor = max(0, cursor - 1)
                continue
            if key in (curses.KEY_RIGHT,):
                cursor = min(len(buffer), cursor + 1)
                continue
            if key in (curses.KEY_HOME,):
                cursor = 0
                continue
            if key in (curses.KEY_END,):
                cursor = len(buffer)
                continue
            if key == 21:  # Ctrl+U
                buffer.clear()
                cursor = 0
                scroll = 0
                continue

            if 0 <= key <= 255:
                ch = chr(key)
                if ch.isprintable():
                    buffer.insert(cursor, ch)
                    cursor += 1
                continue
    finally:
        _safe_curs_set(0)

    return LineEditResult(value=initial, accepted=False)
