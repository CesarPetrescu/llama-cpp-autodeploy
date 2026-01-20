from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayoutBreakpoints:
    WIDE_MIN_WIDTH: int = 140
    TABLET_MIN_WIDTH: int = 80
    NARROW_MIN_WIDTH: int = 40


@dataclass(frozen=True)
class UIDefaults:
    LOG_HISTORY: int = 200
    SPINNER_FRAMES: str = "|/-\\"
    SCAN_TIMEOUT: float = 0.25
    HELP_PREVIEW_MIN: int = 5
    HELP_PREVIEW_MAX: int = 10


@dataclass(frozen=True)
class LayoutSizing:
    min_left_width: int
    min_right_width: int
    left_ratio: float
    left_margin: int = 2
    column_gap: int = 2


LAYOUT = LayoutBreakpoints()
UI = UIDefaults()

AUTODEVOPS_LAYOUT = LayoutSizing(min_left_width=24, min_right_width=24, left_ratio=0.58)
LOADMODEL_LAYOUT = LayoutSizing(min_left_width=24, min_right_width=28, left_ratio=0.55)
LOADMODEL_DIST_LAYOUT = LayoutSizing(min_left_width=26, min_right_width=32, left_ratio=0.55)
