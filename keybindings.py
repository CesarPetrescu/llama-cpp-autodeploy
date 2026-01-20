from __future__ import annotations

import curses
from dataclasses import dataclass


@dataclass(frozen=True)
class Keybindings:
    QUIT = (ord("q"), ord("Q"))
    TOGGLE_HELP = (ord("?"),)
    TOGGLE_COMPACT = (ord("c"), ord("C"))
    REFRESH = (ord("r"), ord("R"))
    CONFIRM = (curses.KEY_ENTER, ord("\n"), ord("\r"))
    CANCEL = (27,)  # Esc
    TAB = (9,)

    NAV_UP = (curses.KEY_UP, ord("k"))
    NAV_DOWN = (curses.KEY_DOWN, ord("j"))
    NAV_LEFT = (curses.KEY_LEFT, ord("h"))
    NAV_RIGHT = (curses.KEY_RIGHT, ord("l"), ord(" "))

    PAGE_UP = (curses.KEY_PPAGE,)
    PAGE_DOWN = (curses.KEY_NPAGE,)
    HOME = (curses.KEY_HOME,)
    END = (curses.KEY_END,)


KEYS = Keybindings()
