#!/usr/bin/env python3
"""Interactive TUI front-end for autodevops.py builds."""
from __future__ import annotations

import curses
import json
import locale
import platform
import shutil
import subprocess
import sys
import textwrap
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence, Set

import autodevops
import tui_utils

SCRIPT_DIR = Path(__file__).resolve().parent
AUTO_SCRIPT = SCRIPT_DIR / "autodevops.py"
CONFIG_PATH = SCRIPT_DIR / ".autodevops_cli.json"
LOG_HISTORY = 200
SPINNER_FRAMES = "|/-\\"

locale.setlocale(locale.LC_ALL, "")

STRINGS = {
    "title": "llama.cpp AutodevOps Builder",
    "instructions": "Arrows: navigate • Space: toggle/cycle • Enter: edit/apply • PgUp/PgDn: scroll • Tab: cycle panes • ?: toggle help • c: compact • Q: quit",
    "logs_heading": "Logs",
    "help_heading": "Help",
    "no_space_warning": "Not enough space to render menu. Enlarge the window.",
    "suggest_install_cuda": "NVCC not detected. Install the CUDA toolkit or export CUDA_HOME to enable CUDA builds and fast math.",
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


@dataclass
class ChoiceValue:
    label: str
    value: str
    enabled: bool = True
    reason: str | None = None


@dataclass
class SystemInfo:
    cpu_vendor: str
    cpu_flags: Set[str]
    arch: str
    gpu_vendor: str
    cuda_home: Path | None
    has_mkl: bool
    has_openblas: bool
    has_blis: bool


SHOW_UNAVAILABLE = False
SHOW_HARDWARE_BADGES = True
COMPACT_LIST = True


def set_show_unavailable(value: bool) -> None:
    global SHOW_UNAVAILABLE
    SHOW_UNAVAILABLE = value


def show_unavailable_enabled() -> bool:
    return SHOW_UNAVAILABLE


def set_show_hardware_badges(value: bool) -> None:
    global SHOW_HARDWARE_BADGES
    SHOW_HARDWARE_BADGES = value


def hardware_badges_enabled() -> bool:
    return SHOW_HARDWARE_BADGES


def set_compact_list(value: bool) -> None:
    global COMPACT_LIST
    COMPACT_LIST = bool(value)


def compact_list_enabled() -> bool:
    return COMPACT_LIST


class OptionBase:
    key: str
    name: str
    description: str
    help_text: str
    disabled: bool
    reason: str | None
    default_value: object
    icon: str = "[OPT]"

    def render(self, win: "curses._CursesWindow", y: int, width: int, selected: bool) -> int:
        raise NotImplementedError

    def handle_key(self, key: int) -> None:
        pass

    def height(self, width: int) -> int:
        return 1

    def get_help(self) -> str:
        return self.help_text or self.description

    def is_modified(self) -> bool:
        return False

    def get_summary(self, width: int) -> str:
        help_text = self.get_help().strip().splitlines()
        if not help_text:
            return ""
        summary = help_text[0].strip()
        if len(summary) > width:
            summary = summary[: max(0, width - 1)] + "…"
        return summary

    def get_value(self):
        raise NotImplementedError

    def set_value(self, value) -> None:
        raise NotImplementedError


class ToggleOption(OptionBase):
    def __init__(
        self,
        key: str,
        name: str,
        description: str,
        value: bool = False,
        *,
        disabled: bool = False,
        reason: str | None = None,
        help_text: str = "",
        on_change: Callable[[bool], None] | None = None,
    ) -> None:
        self.key = key
        self.name = name
        self.description = description
        self.help_text = help_text or description
        self.value = value
        self.disabled = disabled
        self.reason = reason
        self._on_change = on_change
        self.default_value = value

    def toggle(self) -> None:
        if not self.disabled:
            self.value = not self.value
            if self._on_change is not None:
                self._on_change(self.value)

    def handle_key(self, key: int) -> None:
        if key in (curses.KEY_ENTER, ord(" "), ord("t")):
            self.toggle()

    def render(self, win: "curses._CursesWindow", y: int, width: int, selected: bool) -> int:
        attr = curses.A_REVERSE if selected else curses.A_NORMAL
        if self.disabled:
            attr |= curses.color_pair(2)
        marker = "*" if self.is_modified() else " "
        label = f"{marker}[TGL] {'[x]' if self.value else '[ ]'} {self.name}"
        win.addnstr(y, 2, label, max(10, width - 4), attr)
        if compact_list_enabled():
            return 1
        summary = self.get_summary(max(0, width - len(label) - 6))
        if summary:
            win.addnstr(y, min(width - 2, 2 + len(label) + 1), f" · {summary}", max(10, width - len(label) - 4), curses.A_DIM)
        line_count = 1
        desc_attr = curses.A_DIM
        if self.disabled:
            desc_attr |= curses.color_pair(2)
        wrap_width = max(10, width - 6)
        for line in textwrap.wrap(self.description, wrap_width):
            win.addnstr(y + line_count, 6, line, max(10, width - 8), desc_attr)
            line_count += 1
        if self.disabled and self.reason:
            reason = f"⚠ {self.reason}"
            win.addnstr(y + line_count, 6, reason, max(10, width - 8), curses.color_pair(2) | curses.A_DIM)
            line_count += 1
        return line_count

    def height(self, width: int) -> int:
        if compact_list_enabled():
            return 1
        wrap_width = max(10, width - 6)
        base = 1 + len(textwrap.wrap(self.description, wrap_width))
        if self.disabled and self.reason:
            base += 1
        return base

    def is_modified(self) -> bool:
        return self.value != self.default_value

    def get_value(self) -> bool:
        return bool(self.value)

    def set_value(self, value) -> None:
        self.value = bool(value)


class ChoiceOption(OptionBase):
    def __init__(
        self,
        key: str,
        name: str,
        description: str,
        choices: Sequence[ChoiceValue],
        *,
        initial: str | None = None,
        help_text: str = "",
        show_unavailable_fn: Callable[[], bool] | None = None,
    ) -> None:
        if not choices:
            raise ValueError("choices cannot be empty")
        self.key = key
        self.name = name
        self.description = description
        self.help_text = help_text or description
        self.choices = list(choices)
        self.disabled = all(not c.enabled for c in self.choices)
        self.reason = None
        if self.disabled:
            self.reason = "No enabled options"
        self._show_unavailable_fn = show_unavailable_fn
        self.index = 0
        self.default_value = None
        if initial is not None:
            for idx, c in enumerate(self.choices):
                if c.value == initial:
                    self.index = idx
                    break
        if not self.choices[self.index].enabled:
            self._select_next_enabled(1)
        self.default_value = self.choices[self.index].value

    def _select_next_enabled(self, delta: int) -> None:
        if self.disabled:
            return
        count = len(self.choices)
        for _ in range(count):
            self.index = (self.index + delta) % count
            if self.choices[self.index].enabled:
                return

    def handle_key(self, key: int) -> None:
        if key in (curses.KEY_LEFT, ord("h")):
            self._select_next_enabled(-1)
        elif key in (curses.KEY_RIGHT, ord("l"), ord(" ")):
            self._select_next_enabled(1)

    def render(self, win: "curses._CursesWindow", y: int, width: int, selected: bool) -> int:
        attr = curses.A_REVERSE if selected else curses.A_NORMAL
        current = self.choices[self.index]
        if self.disabled:
            attr |= curses.color_pair(2)
        marker = "*" if self.is_modified() else " "
        label = f"{marker}[SEL] {self.name}: {current.label}"
        win.addnstr(y, 2, label, max(10, width - 4), attr)
        if compact_list_enabled():
            return 1
        summary = self.get_summary(max(0, width - len(label) - 6))
        if summary:
            win.addnstr(y, min(width - 2, 2 + len(label) + 1), f" · {summary}", max(10, width - len(label) - 4), curses.A_DIM)
        line_count = 1
        desc_attr = curses.A_DIM
        if self.disabled:
            desc_attr |= curses.color_pair(2)
        wrap_width = max(10, width - 6)
        for line in textwrap.wrap(self.description, wrap_width):
            win.addnstr(y + line_count, 6, line, max(10, width - 8), desc_attr)
            line_count += 1
        if not current.enabled and current.reason:
            reason = f"⚠ {current.reason}"
            win.addnstr(y + line_count, 6, reason, max(10, width - 8), curses.color_pair(2) | curses.A_DIM)
            line_count += 1
        show_unavailable = self._show_unavailable_fn() if self._show_unavailable_fn else False
        if show_unavailable:
            unavailable = [c for c in self.choices if not c.enabled]
            if unavailable:
                win.addnstr(y + line_count, 6, "Unavailable:", max(10, width - 8), curses.A_DIM)
                line_count += 1
                for choice in unavailable:
                    text = choice.label
                    if choice.reason:
                        text += f" — {choice.reason}"
                    list_width = max(10, width - 10)
                    wrapped = textwrap.wrap(text, list_width) or [""]
                    for idx, chunk in enumerate(wrapped):
                        prefix = "• " if idx == 0 else "  "
                        win.addnstr(y + line_count, 8, prefix + chunk, max(10, list_width), curses.A_DIM)
                        line_count += 1
        return line_count

    def height(self, width: int) -> int:
        if compact_list_enabled():
            return 1
        wrap_width = max(10, width - 6)
        base = 1 + len(textwrap.wrap(self.description, wrap_width))
        current = self.choices[self.index]
        if not current.enabled and current.reason:
            base += 1
        show_unavailable = self._show_unavailable_fn() if self._show_unavailable_fn else False
        if show_unavailable:
            unavailable = [c for c in self.choices if not c.enabled]
            if unavailable:
                base += 1
                for choice in unavailable:
                    text = choice.label
                    if choice.reason:
                        text += f" — {choice.reason}"
                    list_width = max(10, width - 10)
                    wrapped = textwrap.wrap(text, list_width) or [""]
                    base += len(wrapped)
        return base

    def is_modified(self) -> bool:
        selected = self.choices[self.index].value
        return selected != self.default_value

    def get_value(self):
        return self.choices[self.index].value

    def set_value(self, value) -> None:
        for idx, choice in enumerate(self.choices):
            if choice.value == value:
                self.index = idx
                break

    @property
    def value(self) -> ChoiceValue:
        return self.choices[self.index]


class InputOption(OptionBase):
    def __init__(self, key: str, name: str, description: str, value: str, placeholder: str = "", help_text: str = "") -> None:
        self.key = key
        self.name = name
        self.description = description
        self.help_text = help_text or description
        self.value = value
        self.placeholder = placeholder
        self.disabled = False
        self.reason = None
        self.default_value = value

    def edit(self, stdscr: "curses._CursesWindow") -> None:
        result = tui_utils.edit_line_dialog(
            stdscr,
            title=f"Edit {self.name}",
            initial=self.value,
            allow_empty=True,
        )
        if result.accepted:
            self.value = result.value

    def handle_key(self, key: int, stdscr: "curses._CursesWindow" | None = None) -> None:
        if key in (curses.KEY_ENTER, ord("\n"), ord("\r")) and stdscr is not None:
            self.edit(stdscr)

    def render(self, win: "curses._CursesWindow", y: int, width: int, selected: bool) -> int:
        attr = curses.A_REVERSE if selected else curses.A_NORMAL
        marker = "*" if self.is_modified() else " "
        display = self.value or self.placeholder
        label = f"{marker}[TXT] {self.name}: {display}"
        win.addnstr(y, 2, label, max(10, width - 4), attr)
        if compact_list_enabled():
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

    def height(self, width: int) -> int:
        if compact_list_enabled():
            return 1
        wrap_width = max(10, width - 6)
        return 1 + len(textwrap.wrap(self.description, wrap_width))

    def is_modified(self) -> bool:
        return self.value != self.default_value

    def get_value(self) -> str:
        return self.value

    def set_value(self, value) -> None:
        if value is None:
            return
        self.value = str(value)


class InfoBadgeOption(OptionBase):
    def __init__(
        self,
        key: str,
        name: str,
        description: str,
        *,
        help_text: str = "",
        icon: str = "ℹ",
        visible_fn: Callable[[], bool] | None = None,
    ) -> None:
        self.key = key
        self.name = name
        self.description = description
        self.help_text = help_text or description
        self.icon = icon
        self._visible_fn = visible_fn
        self.disabled = False
        self.reason = None

    def _visible(self) -> bool:
        if self._visible_fn is None:
            return True
        return self._visible_fn()

    def render(self, win: "curses._CursesWindow", y: int, width: int, selected: bool) -> int:
        if not self._visible():
            return 0
        attr = curses.A_REVERSE if selected else curses.A_DIM
        label = f"{self.icon} {self.name}"
        win.addnstr(y, 2, label, max(10, width - 4), attr)
        if compact_list_enabled():
            return 1
        line_count = 1
        wrap_width = max(10, width - 6)
        for line in textwrap.wrap(self.description, wrap_width):
            win.addnstr(y + line_count, 6, line, max(10, width - 8), curses.A_DIM)
            line_count += 1
        return line_count

    def handle_key(self, key: int) -> None:
        return

    def height(self, width: int) -> int:
        if not self._visible():
            return 0
        if compact_list_enabled():
            return 1
        wrap_width = max(10, width - 6)
        return 1 + len(textwrap.wrap(self.description, wrap_width))

    def get_value(self):
        return None

    def set_value(self, value) -> None:
        return


def detect_cpu_vendor() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.lower().startswith("vendor_id"):
                vendor = line.split(":", 1)[1].strip().lower()
                if "intel" in vendor:
                    return "intel"
                if "amd" in vendor or "hygon" in vendor:
                    return "amd"
    return "unknown"


def detect_gpu_vendor() -> str:
    if shutil.which("nvidia-smi"):
        return "nvidia"
    try:
        out = subprocess.check_output(["lspci"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        out = ""
    out_lower = out.lower()
    if "nvidia" in out_lower:
        return "nvidia"
    if "amd" in out_lower or "advanced micro devices" in out_lower:
        return "amd"
    if "intel" in out_lower and "vga" in out_lower:
        return "intel"
    return "unknown"


def detect_cpu_flags() -> Set[str]:
    flags: Set[str] = set()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            low = line.lower()
            if low.startswith("flags") or low.startswith("features"):
                _, _, raw_flags = line.partition(":")
                flags.update(flag.strip().lower() for flag in raw_flags.split())
    return flags


def blis_present() -> bool:
    candidates = [
        "/usr/lib/x86_64-linux-gnu/libblis.so",
        "/usr/lib/x86_64-linux-gnu/libblis.so.*",
        "/opt/amd/aocl/lib/libblis.so",
        "/opt/amd/aocl/lib/libblis.so.*",
    ]
    try:
        return autodevops._lib_present(candidates)  # type: ignore[attr-defined]
    except AttributeError:
        return False


def collect_system_info() -> SystemInfo:
    cpu_vendor = detect_cpu_vendor()
    gpu_vendor = detect_gpu_vendor()
    flags = detect_cpu_flags()
    arch = platform.machine().lower()
    cuda_home = autodevops.pick_cuda_home()
    has_mkl = autodevops.mkl_present()
    has_openblas = autodevops.openblas_present()
    has_blis = blis_present()
    return SystemInfo(
        cpu_vendor=cpu_vendor,
        cpu_flags=flags,
        arch=arch,
        gpu_vendor=gpu_vendor,
        cuda_home=cuda_home,
        has_mkl=has_mkl,
        has_openblas=has_openblas,
        has_blis=has_blis,
    )


def _attach_detection_metadata(config: dict, info: SystemInfo) -> dict:
    enriched = dict(config)
    enriched["detected_gpu_vendor"] = info.gpu_vendor
    enriched["detected_cuda_home"] = str(info.cuda_home) if info.cuda_home else None
    return enriched


def build_options(system_info: SystemInfo | None = None) -> List[OptionBase]:
    """Build the interactive option list.

    Parameters
    ----------
    system_info:
        Optional :class:`SystemInfo` instance to seed the wizard. Providing a
        pre-computed value makes it easy to unit test the menu logic without
        touching real hardware probes.
    """

    info = system_info or collect_system_info()
    cpu_vendor = info.cpu_vendor
    gpu_vendor = info.gpu_vendor
    has_cuda = info.cuda_home is not None
    fast_math_disabled = not has_cuda
    fast_math_reason = None
    if fast_math_disabled:
        if gpu_vendor == "nvidia":
            fast_math_reason = "NVCC (CUDA Toolkit) not found — set CUDA_HOME or install cuda-toolkit"
        else:
            fast_math_reason = "NVCC not found on this system"

    blas_choices = [
        ChoiceValue("Auto", "auto"),
        ChoiceValue("Intel oneAPI MKL", "mkl", info.has_mkl, "Intel MKL libraries not detected"),
        ChoiceValue("OpenBLAS", "openblas", info.has_openblas, "OpenBLAS libraries not detected"),
        ChoiceValue("AMD BLIS", "blis", info.has_blis, "AMD BLIS libraries not detected"),
        ChoiceValue("Disabled", "off"),
    ]

    backend_choices = [
        ChoiceValue(
            "CUDA (NVIDIA)",
            "cuda",
            has_cuda,
            "CUDA Toolkit not found (set CUDA_HOME or install cuda-toolkit)" if not has_cuda else None,
        ),
        ChoiceValue("ROCm (AMD)", "rocm", gpu_vendor == "amd", "ROCm toolchain not detected"),
        ChoiceValue("oneAPI / SYCL (Intel)", "oneapi", gpu_vendor == "intel", "Intel oneAPI compilers not detected"),
        ChoiceValue("Vulkan (universal)", "vulkan", True),
        ChoiceValue("CPU only", "cpu", True),
    ]

    has_avx2 = "avx2" in info.cpu_flags
    has_avx512 = any(flag.startswith("avx512") for flag in info.cpu_flags)
    has_avx_vnni = any(flag in info.cpu_flags for flag in ("avxvnni", "avx_vnni", "avx512_vnni"))
    is_arm = info.arch in {"aarch64", "arm64"}

    cpu_choices = [
        ChoiceValue("Auto (detect)", "auto"),
        ChoiceValue(
            "Intel AVX2", "intel_avx2", enabled=cpu_vendor == "intel" and has_avx2,
            reason="Requires an Intel CPU with AVX2"
        ),
        ChoiceValue(
            "Intel AVX-512 + MKL", "intel_avx512",
            enabled=cpu_vendor == "intel" and has_avx512,
            reason="CPU does not report AVX-512 support"
        ),
        ChoiceValue(
            "AMD Zen 3/4 (OpenBLAS)", "amd_zen", enabled=cpu_vendor == "amd" and has_avx2,
            reason="Requires an AMD CPU with AVX2"
        ),
        ChoiceValue(
            "AMD Zen 4 + VNNI", "amd_zen4",
            enabled=cpu_vendor == "amd" and has_avx_vnni,
            reason="AVX-VNNI extensions not detected"
        ),
        ChoiceValue(
            "ARM64 / Apple Silicon", "arm64", enabled=is_arm,
            reason="Machine is not reporting ARM64"
        ),
        ChoiceValue("Generic portable", "generic"),
    ]

    runtime_choices = [
        ChoiceValue("Balanced", "balanced"),
        ChoiceValue("High-memory throughput", "high_mem"),
        ChoiceValue("Memory constrained", "low_mem"),
        ChoiceValue("Multi-GPU", "multi_gpu"),
    ]

    quant_choices = [
        ChoiceValue("Auto", "auto"),
        ChoiceValue("FP16 focus", "fp16"),
        ChoiceValue("INT8 speed", "int8"),
        ChoiceValue("Q4_K_M compact", "q4_k_m"),
    ]

    options: List[OptionBase] = [
        InputOption(
            "ref",
            "Target ref",
            "Git tag, branch, or commit to build. Use 'latest' to fetch the newest release.",
            value="latest",
            placeholder="latest",
            help_text=(
                "Set which llama.cpp revision to build. "
                "The wizard will fetch releases automatically when 'latest' is used."
            ),
        ),
        ToggleOption(
            "now",
            "Build immediately",
            "Run the build as soon as you exit this wizard.",
            value=True,
            help_text=(
                "If enabled, autodevops.py is launched as soon as you press Start. "
                "Disable it to just print the recommended commands without running them."
            ),
        ),
        ToggleOption(
            "_show_unavailable",
            "Show unavailable choices",
            "Display disabled presets with explanations below each selector.",
            value=show_unavailable_enabled(),
            help_text=(
                "Enabling this reveals options that are currently disabled along with the"
                " reason they are unavailable on this system."
            ),
            on_change=set_show_unavailable,
        ),
        ToggleOption(
            "_show_hardware_badges",
            "Show hardware detection badges",
            "Toggle informational badges that summarise detected CPUs and GPUs.",
            value=hardware_badges_enabled(),
            help_text=(
                "Hardware badges highlight detected vendors and suggested build tips."
                " Disable this if you prefer a minimal menu."
            ),
            on_change=set_show_hardware_badges,
        ),
        ChoiceOption(
            "backend",
            "GPU backend",
            "Available GPU accelerators based on detected hardware.",
            backend_choices,
            show_unavailable_fn=show_unavailable_enabled,
            help_text=(
                "Choose which accelerator backend to prepare for.\n"
                " • CUDA: native NVIDIA support with MMQ/cuBLAS kernels.\n"
                " • ROCm: AMD RDNA/CDNA GPUs using hipcc (see advanced cmake example above).\n"
                " • oneAPI/SYCL: Intel GPUs via icx/icpx compilers.\n"
                " • Vulkan: universal backend ~7% slower but works across vendors.\n"
                " • CPU only: build without GPU offload."
            ),
        ),
        ChoiceOption(
            "cpu_profile",
            "CPU optimization profile",
            "Selects tuned CMake flags derived from the comprehensive compilation guide.",
            cpu_choices,
            show_unavailable_fn=show_unavailable_enabled,
            help_text=(
                "Curated CPU build presets:\n"
                " • Intel AVX2: -DGGML_AVX=ON -DGGML_AVX2=ON with -O3 -march=native.\n"
                " • Intel AVX-512 + MKL: adds -DGGML_AVX512=ON and Intel oneAPI MKL toolchain.\n"
                " • AMD Zen 3/4: enables AVX2/VNNI paths with OpenBLAS or BLIS.\n"
                " • ARM64: lean build relying on Apple Metal/Accelerate or -mcpu flags.\n"
                "Auto mode picks the best option based on detected vendor/features."
            ),
        ),
        ChoiceOption(
            "blas",
            "CPU BLAS backend",
            "Select BLAS acceleration for CPU fallbacks.",
            blas_choices,
            initial="auto",
            show_unavailable_fn=show_unavailable_enabled,
            help_text=(
                "Compare BLAS libraries: MKL excels on Intel, OpenBLAS is versatile, and AMD BLIS"
                " leads on Ryzen. Built-in GGML kernels can win for token generation."
            ),
        ),
        ChoiceOption(
            "force_mmq",
            "Force MMQ kernels",
            "Controls GGML_CUDA_FORCE_MMQ. 'Auto' enables it on newer NVIDIA GPUs.",
            [
                ChoiceValue("Auto", "auto"),
                ChoiceValue("On", "on"),
                ChoiceValue("Off", "off"),
            ],
            show_unavailable_fn=show_unavailable_enabled,
            help_text=(
                "Force the mixed-memory (MMQ) CUDA kernels. Auto lets llama.cpp choose between"
                " cuBLAS and MMQ depending on tensor-core support. Disable if older GPUs misbehave."
            ),
        ),
        ToggleOption(
            "fast_math",
            "Enable fast math",
            "Adds --use_fast_math to NVCC for potential speedups at the cost of precision.",
            value=False,
            disabled=fast_math_disabled,
            reason=fast_math_reason,
            help_text=(
                "Fast math maps transcendental ops to lower-precision CUDA intrinsics."
                " Use it on inference-only systems when you can tolerate minor accuracy drift."
            ),
        ),
        ToggleOption(
            "distributed",
            "Enable distributed RPC backend",
            "Compile llama.cpp with GGML_RPC so rpc-server and multi-host inference are available.",
            value=False,
            help_text=(
                "Turns on the GGML RPC build path. Requires NCCL/MPI and is currently considered"
                " proof-of-concept; run only on trusted networks."
            ),
        ),
        ToggleOption(
            "unified_memory",
            "Enable CUDA unified memory",
            "Sets GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 at runtime for oversized models.",
            value=False,
            help_text=(
                "Unified memory lets CUDA spill activations into system RAM, enabling models"
                " larger than VRAM at the cost of PCIe traffic. Combine with partial CPU offload."
            ),
        ),
        ToggleOption(
            "flash_attention",
            "Prefer Flash Attention",
            "Reminds you to run llama-cli with -fa for faster prompt processing.",
            value=True,
            help_text=(
                "Flash Attention reduces memory usage and improves prompt throughput."
                " The runtime flag is '-fa' in llama-cli."
            ),
        ),
        ChoiceOption(
            "runtime_profile",
            "Runtime tuning profile",
            "Suggests llama-cli runtime arguments tailored to your system budget.",
            runtime_choices,
            show_unavailable_fn=show_unavailable_enabled,
            help_text=(
                "Quick-start runtime templates:\n"
                " • Balanced: -t (nproc) -ngl 35 -c 8192 -b 1024 --cache-reuse 256.\n"
                " • High-memory: -t (nproc) -ngl 35 -c 16384 -b 2048 --mlock --no-mmap.\n"
                " • Memory constrained: -t 8 -ngl 20 -c 4096 -b 512 --tensor-split 0.6,0.4.\n"
                " • Multi-GPU: --tensor-split auto --main-gpu 0 -ngl 80 (NVLink) or manual splits."
            ),
        ),
        ChoiceOption(
            "quantization",
            "Quantization focus",
            "Guides which GGUF quantization families to prioritise when downloading models.",
            quant_choices,
            show_unavailable_fn=show_unavailable_enabled,
            help_text=(
                "FP16 maximises quality on capable GPUs. INT8 (Q8_0) balances speed and size."
                " Q4_K_M keeps small VRAM footprints with acceptable quality for chatbots."
            ),
        ),
    ]

    # Informational badges
    if cpu_vendor == "intel":
        options.append(
            InfoBadgeOption(
                "_info_cpu",
                "Intel CPU detected",
                "Optimised Intel builds with MKL are available when libraries are installed.",
                help_text=(
                    "Detected Intel CPU. Consider the advanced MKL recipe:\n"
                    "  source /opt/intel/oneapi/setvars.sh\n"
                    "  cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Intel10_64lp\n"
                    "        -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx\n"
                    "        -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_AVX512=ON\n"
                    "        -DCMAKE_C_FLAGS=\"-O3 -ipo -static -fp-model=fast -march=native\""
                ),
                icon="🧠",
                visible_fn=hardware_badges_enabled,
            )
        )
    elif cpu_vendor == "amd":
        options.append(
            InfoBadgeOption(
                "_info_cpu",
                "AMD CPU detected",
                "Install OpenBLAS/BLIS for the best CPU throughput.",
                help_text=(
                    "Detected AMD CPU. Recommended build:\n"
                    "  cmake -B build -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_AVX_VNNI=ON\n"
                    "        -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=BLIS\n"
                    "        -DCMAKE_C_FLAGS=\"-O3 -march=native -mavx2 -mcpu=native\""
                ),
                icon="🧠",
                visible_fn=hardware_badges_enabled,
            )
        )
    elif is_arm:
        options.append(
            InfoBadgeOption(
                "_info_cpu",
                "ARM64 CPU detected",
                "Apple/ARM builds can enable Metal or Accelerate backends automatically.",
                help_text=(
                    "On Apple Silicon run: cmake -B build -DGGML_METAL=ON and make llama-cli."
                ),
                icon="🧠",
                visible_fn=hardware_badges_enabled,
            )
        )

    if gpu_vendor == "nvidia":
        options.append(
            InfoBadgeOption(
                "_info_gpu",
                "NVIDIA GPU detected",
                "CUDA builds are available on this system.",
                help_text=(
                    "Recommended CUDA command:\n"
                    "  cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_F16=ON\n"
                    "        -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 -DCMAKE_CUDA_ARCHITECTURES=native"
                ),
                icon="🎮",
                visible_fn=hardware_badges_enabled,
            )
        )
    elif gpu_vendor == "amd":
        options.append(
            InfoBadgeOption(
                "_info_gpu",
                "AMD GPU detected",
                "ROCm builds are supported manually via hipcc.",
                help_text=(
                    "Export CC=/opt/rocm/llvm/bin/clang and run cmake -DGGML_HIP=ON -DAMDGPU_TARGETS=native."
                ),
                icon="🎮",
                visible_fn=hardware_badges_enabled,
            )
        )
    elif gpu_vendor == "intel":
        options.append(
            InfoBadgeOption(
                "_info_gpu",
                "Intel GPU detected",
                "Use oneAPI SYCL builds with icx/icpx compilers.",
                help_text=(
                    "source /opt/intel/oneapi/setvars.sh and cmake -DGGML_SYCL=ON -DGGML_SYCL_F16=ON."
                ),
                icon="🎮",
                visible_fn=hardware_badges_enabled,
            )
        )

    return options


def compile_config(options: Sequence[OptionBase]) -> dict:
    config: dict = {}
    for opt in options:
        if opt.key.startswith("_"):
            continue
        if isinstance(opt, InputOption):
            config[opt.key] = opt.value
        elif isinstance(opt, ToggleOption):
            config[opt.key] = opt.value
        elif isinstance(opt, ChoiceOption):
            config[opt.key] = opt.value.value
    return config


def cpu_profile_instructions(profile: str) -> str:
    mapping = {
        "intel_avx2": "cmake -B build -DGGML_AVX=ON -DGGML_AVX2=ON -DCMAKE_BUILD_TYPE=Release \\\n+  -DCMAKE_C_FLAGS=\"-O3 -march=native -mavx2\" -DCMAKE_CXX_FLAGS=\"-O3 -march=native -mavx2\"",
        "intel_avx512": "source /opt/intel/oneapi/setvars.sh\ncmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Intel10_64lp \\\n+  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_AVX512=ON \\\n+  -DCMAKE_C_FLAGS=\"-O3 -ipo -static -fp-model=fast -march=native\"",
        "amd_zen": "cmake -B build -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_AVX_VNNI=ON -DCMAKE_BUILD_TYPE=Release \\\n+  -DCMAKE_C_FLAGS=\"-O3 -march=native -mavx2 -mcpu=native\" -DCMAKE_CXX_FLAGS=\"-O3 -march=native -mavx2 -mcpu=native\"",
        "amd_zen4": "cmake -B build -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_AVX_VNNI=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=BLIS",
        "arm64": "cmake -B build -DCMAKE_C_FLAGS=\"-O3 -mcpu=native -march=native\" -DCMAKE_CXX_FLAGS=\"-O3 -mcpu=native -march=native\"",
        "generic": "cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF",
    }
    return mapping.get(profile, "")


def backend_instructions(backend: str) -> str:
    mapping = {
        "cuda": "cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_F16=ON -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 \\\n+  -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=native",
        "rocm": "export CC=/opt/rocm/llvm/bin/clang\nexport CXX=/opt/rocm/llvm/bin/clang++\ncmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=native -DGGML_HIP_ROCWMMA_FATTN=ON",
        "oneapi": "source /opt/intel/oneapi/setvars.sh\ncmake -B build -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_TARGET=INTEL -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx",
        "vulkan": "cmake -B build -DGGML_VULKAN=ON -DGGML_AVX=ON -DGGML_AVX2=ON",
        "cpu": "cmake -B build -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_VULKAN=OFF",
    }
    return mapping.get(backend, "")


def blas_instructions(vendor: str) -> str:
    if vendor == "mkl":
        return autodevops.blas_hint("mkl")
    if vendor == "openblas":
        return autodevops.blas_hint("openblas")
    if vendor == "blis":
        return (
            "Install AMD BLIS (part of AOCL) and export BLIS_NUM_THREADS=auto for optimal scaling."
        )
    return ""


def runtime_profile_instructions(profile: str) -> str:
    mapping = {
        "balanced": "./llama-cli -t $(nproc) -ngl 35 -c 8192 -b 1024 --cache-reuse 256",
        "high_mem": "./llama-cli -t $(nproc) -ngl 35 -c 16384 -b 2048 --mlock --no-mmap --cache-reuse 256",
        "low_mem": "./llama-cli -t 8 -ngl 20 -c 4096 -b 512 --tensor-split 0.6,0.4",
        "multi_gpu": "./llama-cli --tensor-split auto --main-gpu 0 -ngl 80",
    }
    return mapping.get(profile, "")


def quantization_notes(flavour: str) -> str:
    mapping = {
        "fp16": "Use GGUF models ending with -F16 for highest fidelity on modern GPUs.",
        "int8": "Prioritise GGUF Q8_0 for the best balance of speed and quality when VRAM allows.",
        "q4_k_m": "Q4_K_M GGUF saves VRAM (~4 bits) yet keeps good chat accuracy; ideal for 8-12GB GPUs.",
    }
    return mapping.get(flavour, "")


def _draw_wide_layout(
    stdscr: "curses._CursesWindow",
    options: List[OptionBase],
    visible_options: List[tuple[int, OptionBase, int]],
    selected: int,
    selected_visible_idx: int | None,
    scroll: int,
    message: str | None,
    help_lines: List[str],
    ui_state: dict,
    height: int,
    width: int,
) -> tuple[int | None, int | None]:
    body_top = 3
    footer_y = height - 2
    body_height = max(0, footer_y - body_top)
    left_margin = 2
    column_gap = 2
    min_left_width = 24
    min_right_width = 24

    left_width = max(min_left_width, int(width * 0.58))
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
        warning = "Not enough space to render the dashboard. Enlarge the window."
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
        opt_index, opt, _ = visible_options[visible_idx]
        if y >= footer_y:
            break
        opt_height = opt.height(left_width)
        if opt_height <= 0:
            continue
        render_start = y
        if opt_height > body_height and first_rendered is not None:
            break
        if y + opt_height > footer_y:
            if first_rendered is None:
                extra = opt.render(stdscr, y, left_width, selected == opt_index)
                first_render_row = render_start
                y += extra
                first_rendered = visible_idx
                last_rendered = visible_idx
                last_render_row = y - 1
            break
        extra = opt.render(stdscr, y, left_width, selected == opt_index)
        if first_render_row is None:
            first_render_row = render_start
        y += extra
        if first_rendered is None:
            first_rendered = visible_idx
        last_rendered = visible_idx
        last_render_row = y - 1
        if y >= footer_y:
            break
        y += 1

    separator_x = right_start - 1
    if 0 <= separator_x < width:
        stdscr.vline(body_top, separator_x, curses.ACS_VLINE, max(0, min(body_height, height - body_top)))

    has_more_above = start > 0
    has_more_below = last_rendered is not None and last_rendered < total_visible - 1
    arrow_attr = curses.A_DIM | curses.A_BOLD
    if has_more_above and first_render_row is not None:
        arrow_row = max(body_top, min(first_render_row, height - 1))
        stdscr.addnstr(arrow_row, left_margin - 1, "↑", 1, arrow_attr)
    if has_more_below and last_render_row is not None:
        arrow_row = max(body_top, min(last_render_row, height - 1))
        stdscr.addnstr(arrow_row, left_margin - 1, "↓", 1, arrow_attr)

    right_y = body_top
    wrap_width = max(10, right_width)
    if message:
        for line in textwrap.wrap(message, wrap_width):
            if right_y >= footer_y:
                break
            stdscr.addnstr(right_y, right_start, line, wrap_width, curses.A_BOLD | curses.color_pair(2))
            right_y += 1
        if right_y < footer_y:
            right_y += 1

    focus_area = ui_state.get("focus_area", 0)
    help_focus = focus_area == 1

    selected_title = ""
    title_attr = curses.A_BOLD | (curses.A_REVERSE if help_focus else 0)
    header_row = right_y
    if right_y < footer_y:
        if selected < len(options):
            selected_title = f"Selected: {options[selected].name}"
        elif selected == len(options):
            selected_title = "Selected: Start build"
        else:
            selected_title = "Selected"

    help_content = help_lines or ["No details available."]
    available_help_rows = max(0, footer_y - right_y)
    ui_state.setdefault("help_offset", 0)
    if available_help_rows > 0:
        max_offset = max(0, len(help_content) - available_help_rows)
        help_offset = min(max(0, ui_state.get("help_offset", 0)), max_offset)
        ui_state["help_offset"] = help_offset
        ui_state["help_max_offset"] = max_offset
        visible_help = help_content[help_offset : help_offset + available_help_rows]
        ui_state["help_visible_lines"] = len(visible_help)
    else:
        ui_state["help_max_offset"] = 0
        ui_state["help_visible_lines"] = 0
        visible_help = []

    indicator = ""
    if ui_state.get("help_offset", 0) > 0:
        indicator += "↑"
    if ui_state.get("help_offset", 0) < ui_state.get("help_max_offset", 0):
        indicator += "↓"
    if selected_title and header_row < footer_y:
        header_text = selected_title
        if indicator:
            header_text = f"{selected_title} ({indicator})"
        stdscr.addnstr(header_row, right_start, header_text[:wrap_width], wrap_width, title_attr)
        right_y = header_row + 1

    if help_content:
        for line in visible_help:
            if right_y >= footer_y:
                break
            attr = curses.A_DIM | (curses.A_REVERSE if help_focus else 0)
            stdscr.addnstr(right_y, right_start, line[:wrap_width], wrap_width, attr)
            right_y += 1

    confirm_attr = curses.A_BOLD | (curses.A_REVERSE if selected == len(options) else 0)
    confirm_text = "Start build"
    confirm_x = max(2, (width - len(confirm_text)) // 2)
    stdscr.addnstr(footer_y, confirm_x, confirm_text, len(confirm_text), confirm_attr)

    # Draw logs beneath confirmation line if space allows
    draw_logs(stdscr, ui_state, max(body_top, footer_y - 3), width, height)

    stdscr.addnstr(footer_y, confirm_x, confirm_text, len(confirm_text), confirm_attr)

    return (first_rendered, last_rendered)


def _draw_tablet_layout(
    stdscr: "curses._CursesWindow",
    options: List[OptionBase],
    visible_options: List[tuple[int, OptionBase, int]],
    selected: int,
    scroll: int,
    message: str | None,
    help_lines: List[str],
    ui_state: dict,
    height: int,
    width: int,
) -> tuple[int | None, int | None]:
    body_top = 3
    confirm_y = height - 2
    if confirm_y <= body_top:
        confirm_y = min(height - 1, body_top + 1)
    space_below = max(0, confirm_y - body_top)

    top_message_lines: List[str] = []
    y = body_top
    if message:
        top_message_lines = textwrap.wrap(message, width - 4)
        for idx, line in enumerate(top_message_lines[:3]):
            stdscr.addnstr(body_top + idx, 2, line, width - 4, curses.A_BOLD | curses.color_pair(2))
        y = max(y, body_top + len(top_message_lines[:3]))

    selected_help_lines = help_lines or [""]
    max_help_lines_area = max(0, space_below - 2)
    displayable_help_lines = min(len(selected_help_lines), max_help_lines_area)
    help_height = 1 + displayable_help_lines if displayable_help_lines else 0

    body_height = max(0, (confirm_y - y) - help_height)
    if body_height <= 0:
        warning = "Not enough vertical space to render menu. Enlarge the window."
        stdscr.addnstr(y, 2, warning[: max(1, width - 4)], max(1, width - 4), curses.A_BOLD | curses.color_pair(2))
        confirm_attr = curses.A_BOLD | (curses.A_REVERSE if selected == len(options) else 0)
        confirm_text = "Start build"
        confirm_x = max(2, (width - len(confirm_text)) // 2)
        stdscr.addnstr(confirm_y, confirm_x, confirm_text, len(confirm_text), confirm_attr)
        stdscr.refresh()
        return (None, None)

    column_gap = 3
    available_width = width - 4
    column_count = 2 if available_width >= 40 else 1
    if column_count <= 1:
        column_count = 1
        column_gap = 0
    column_width = max(20, (available_width - column_gap * (column_count - 1)) // column_count)
    x_positions = [2 + (column_width + column_gap) * idx for idx in range(column_count)]
    y_positions = [y] * column_count

    total_visible = len(visible_options)
    start = max(0, min(scroll, max(0, total_visible - 1)))
    first_rendered: int | None = None
    last_rendered: int | None = None
    first_render_row: int | None = None
    last_render_row: int | None = None

    idx = start
    current_col = 0
    while idx < total_visible:
        opt_index, opt, _ = visible_options[idx]
        opt_height = opt.height(column_width)
        placed = False
        col = current_col
        while col < column_count:
            if y_positions[col] + opt_height <= y + body_height:
                placed = True
                break
            col += 1
        if not placed:
            break
        current_col = col
        draw_y = y_positions[col]
        draw_x = x_positions[col]
        extra = opt.render(stdscr, draw_y, column_width, selected == opt_index)
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
        stdscr.addnstr(max(y, first_render_row), 0, "↑", 1, arrow_attr)
    if has_more_below and last_render_row is not None:
        stdscr.addnstr(min(y + body_height - 1, last_render_row), 0, "↓", 1, arrow_attr)

    help_start = y + body_height
    focus_area = ui_state.get("focus_area", 0)
    help_content = selected_help_lines
    available_help_rows = max(0, confirm_y - help_start - 1)
    ui_state.setdefault("help_offset", 0)
    if help_content and available_help_rows > 0:
        header_attr = curses.A_BOLD | (curses.A_REVERSE if focus_area == 1 else curses.A_DIM)
        stdscr.hline(help_start, 1, curses.ACS_HLINE, width - 2)
        max_offset = max(0, len(help_content) - available_help_rows)
        help_offset = min(max(0, ui_state.get("help_offset", 0)), max_offset)
        ui_state["help_offset"] = help_offset
        ui_state["help_max_offset"] = max_offset
        visible_help = help_content[help_offset : help_offset + available_help_rows]
        ui_state["help_visible_lines"] = len(visible_help)
        indicator = ""
        if help_offset > 0:
            indicator += "↑"
        if help_offset < max_offset:
            indicator += "↓"
        header_text = f" {STRINGS['help_heading']} "
        if indicator:
            header_text = f"{header_text}({indicator})"
        stdscr.addnstr(help_start, 3, header_text[: max(1, width - 6)], max(1, width - 6), header_attr)
        row = help_start + 1
        for line in visible_help:
            if row >= confirm_y:
                break
            line_attr = curses.A_DIM | (curses.A_REVERSE if focus_area == 1 else 0)
            stdscr.addnstr(row, 2, line[: width - 4], width - 4, line_attr)
            row += 1
    else:
        ui_state["help_max_offset"] = 0
        ui_state["help_visible_lines"] = 0

    confirm_attr = curses.A_BOLD | (curses.A_REVERSE if selected == len(options) else 0)
    confirm_text = "Start build"
    confirm_x = max(2, (width - len(confirm_text)) // 2)
    stdscr.addnstr(confirm_y, confirm_x, confirm_text, len(confirm_text), confirm_attr)

    logs_start = confirm_y - 3
    draw_logs(stdscr, ui_state, max(body_top, logs_start), width, height)
    stdscr.addnstr(confirm_y, confirm_x, confirm_text, len(confirm_text), confirm_attr)

    stdscr.refresh()
    return (first_rendered, last_rendered)


def draw_screen(
    stdscr: "curses._CursesWindow",
    options: List[OptionBase],
    visible_options: List[tuple[int, OptionBase, int]],
    selected: int,
    selected_visible_idx: int | None,
    scroll: int,
    message: str | None,
    ui_state: dict,
) -> tuple[int | None, int | None]:
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    if height < 5 or width < 20:
        warning = "Terminal window too small for the menu. Please resize."
        stdscr.addnstr(0, 0, warning[: max(1, width - 1)], max(1, width - 1), curses.A_BOLD)
        stdscr.refresh()
        return (None, None)

    title = STRINGS["title"]
    stdscr.addnstr(0, max(0, (width - len(title)) // 2), title, width - 1, curses.A_BOLD)
    instructions = STRINGS["instructions"]
    stdscr.addnstr(1, max(0, (width - len(instructions)) // 2), instructions, width - 1, curses.A_DIM)

    if width >= 140:
        layout_mode = "wide"
    elif width >= 80:
        layout_mode = "tablet"
    else:
        layout_mode = "narrow"

    y = 3
    selected_help = ""
    if selected < len(options):
        selected_help = options[selected].get_help()
    elif selected == len(options):
        selected_help = "Start the build with the currently selected presets."

    if not ui_state.get("show_full_help", False):
        if selected_help:
            selected_help = selected_help.splitlines()[0]
    help_lines = textwrap.wrap(selected_help or "", max(10, width // 2)) or [""]
    ui_state.setdefault("help_offset", 0)
    ui_state["help_total_lines"] = len(help_lines)

    if layout_mode == "wide":
        return _draw_wide_layout(
            stdscr,
            options,
            visible_options,
            selected,
            selected_visible_idx,
            scroll,
            message,
            help_lines,
            ui_state,
            height,
            width,
        )

    if layout_mode == "tablet":
        return _draw_tablet_layout(
            stdscr,
            options,
            visible_options,
            selected,
            scroll,
            message,
            help_lines,
            ui_state,
            height,
            width,
        )

    top_message_lines: List[str] = []
    if message:
        top_message_lines = textwrap.wrap(message, width - 4)
        for idx, line in enumerate(top_message_lines[:3]):
            stdscr.addnstr(2 + idx, 2, line, width - 4, curses.A_BOLD | curses.color_pair(2))
        y = max(y, 3 + len(top_message_lines[:3]))

    body_width = max(1, width - 4)
    body_top = y

    raw_help_lines = textwrap.wrap(selected_help or "", width - 4) or [""]
    ui_state["help_total_lines"] = len(raw_help_lines)

    confirm_y = height - 2
    if confirm_y <= y:
        confirm_y = min(height - 1, y + 1)
    space_below_body = max(0, confirm_y - y)

    available_help_rows = max(0, space_below_body - 1)
    help_height = 1 + min(len(raw_help_lines), available_help_rows) if available_help_rows > 0 else 0

    body_height = max(0, space_below_body - help_height)
    body_bottom = y + body_height
    y = max(y, 3)
    if body_height <= 0:
        warning = "Not enough vertical space to render menu. Enlarge the window."
        stdscr.addnstr(y, 2, warning[: max(1, width - 4)], max(1, width - 4), curses.A_BOLD | curses.color_pair(2))
        confirm_attr = curses.A_BOLD | (curses.A_REVERSE if selected == len(options) else 0)
        confirm_text = "Start build"
        confirm_x = max(2, (width - len(confirm_text)) // 2)
        if 0 <= confirm_y < height:
            stdscr.addnstr(confirm_y, confirm_x, confirm_text, len(confirm_text), confirm_attr)
        stdscr.refresh()
        return (None, None)

    first_rendered: int | None = None
    last_rendered: int | None = None
    first_render_row: int | None = None
    last_render_row: int | None = None
    total_visible = len(visible_options)
    start = max(0, min(scroll, max(0, total_visible - 1)))
    rendered_count = 0
    for visible_idx in range(start, total_visible):
        opt_index, opt, _ = visible_options[visible_idx]
        if y >= body_bottom:
            break
        opt_height = opt.height(body_width)
        if opt_height > body_height and rendered_count > 0:
            break
        render_start = y
        if y + opt_height > body_bottom:
            if rendered_count == 0:
                extra = opt.render(stdscr, y, body_width, selected == opt_index)
                if first_render_row is None:
                    first_render_row = render_start
                y += extra
                first_rendered = visible_idx
                last_rendered = visible_idx
                last_render_row = y - 1
            break
        extra = opt.render(stdscr, y, body_width, selected == opt_index)
        if first_render_row is None:
            first_render_row = render_start
        y += extra
        rendered_count += 1
        if first_rendered is None:
            first_rendered = visible_idx
        last_rendered = visible_idx
        last_render_row = y - 1
        if y >= body_bottom:
            break
        y += 1

    has_more_above = start > 0
    has_more_below = last_rendered is not None and last_rendered < total_visible - 1
    arrow_attr = curses.A_DIM | curses.A_BOLD
    if has_more_above and first_render_row is not None:
        arrow_row = max(body_top, min(first_render_row, height - 1))
        stdscr.addnstr(arrow_row, 0, "↑", 1, arrow_attr)
    if has_more_below and last_render_row is not None:
        arrow_row = max(body_top, min(last_render_row, height - 1))
        stdscr.addnstr(arrow_row, 0, "↓", 1, arrow_attr)

    help_start = body_bottom
    help_end = body_bottom - 1
    help_focus = ui_state.get("focus_area", 0) == 1
    if help_height > 0 and help_start < confirm_y:
        header_attr = curses.A_BOLD | (curses.A_REVERSE if help_focus else curses.A_DIM)
        stdscr.hline(help_start, 1, curses.ACS_HLINE, width - 2)
        available_help_rows_draw = max(0, confirm_y - help_start - 1)
        max_offset = max(0, len(raw_help_lines) - available_help_rows_draw)
        ui_state.setdefault("help_offset", 0)
        help_offset = min(max(0, ui_state.get("help_offset", 0)), max_offset)
        ui_state["help_offset"] = help_offset
        ui_state["help_max_offset"] = max_offset
        visible_help = raw_help_lines[help_offset : help_offset + available_help_rows_draw]
        ui_state["help_visible_lines"] = len(visible_help)
        indicator = ""
        if help_offset > 0:
            indicator += "↑"
        if help_offset < max_offset:
            indicator += "↓"
        header_text = f" {STRINGS['help_heading']} "
        if indicator:
            header_text = f"{header_text}({indicator})"
        stdscr.addnstr(help_start, 3, header_text[: max(1, width - 6)], max(1, width - 6), header_attr)
        for idx, line in enumerate(visible_help):
            row = help_start + 1 + idx
            if row >= confirm_y:
                break
            line_attr = curses.A_DIM | (curses.A_REVERSE if help_focus else 0)
            stdscr.addnstr(row, 2, line, width - 4, line_attr)
        help_end = min(help_start + len(visible_help), confirm_y - 1)
    else:
        ui_state["help_max_offset"] = 0
        ui_state["help_visible_lines"] = 0

    confirm_attr = curses.A_BOLD | (curses.A_REVERSE if selected == len(options) else 0)
    confirm_text = "Start build"
    confirm_x = max(2, (width - len(confirm_text)) // 2)
    if 0 <= confirm_y < height:
        stdscr.addnstr(confirm_y, confirm_x, confirm_text, len(confirm_text), confirm_attr)

    if message and confirm_y > 0:
        bottom_message_lines = textwrap.wrap(message, width - 4)[:2]
        available_rows = max(0, confirm_y - help_end - 1)
        bottom_message_lines = bottom_message_lines[:available_rows]
        for idx, line in enumerate(bottom_message_lines):
            row = confirm_y - 1 - idx
            if row <= help_end or row < 0:
                break
            stdscr.addnstr(row, 2, line, width - 4, curses.A_BOLD | curses.color_pair(2))

    draw_logs(stdscr, ui_state, max(3, confirm_y - 3), width, height)

    stdscr.refresh()
    return (first_rendered, last_rendered)


def run_wizard(stdscr: "curses._CursesWindow") -> dict | None:
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)
    curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)
    system_info = collect_system_info()
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
    if system_info.cuda_home is None:
        append_log(ui_state, STRINGS["suggest_install_cuda"])

    options = build_options(system_info=system_info)
    saved_values = load_saved_state()
    apply_saved_values(options, saved_values)

    selected = 0
    scroll = 0

    while True:
        height, width = stdscr.getmaxyx()
        body_width = max(1, width - 4)

        visible_options: List[tuple[int, OptionBase, int]] = []
        selected_visible_idx: int | None = None
        for idx, opt in enumerate(options):
            opt_height = opt.height(body_width)
            if opt_height <= 0:
                continue
            visible_options.append((idx, opt, opt_height))
            if idx == selected:
                selected_visible_idx = len(visible_options) - 1

        if not visible_options:
            stdscr.erase()
            stdscr.addnstr(0, 0, "No options available.", width - 1, curses.A_BOLD | curses.color_pair(2))
            stdscr.addnstr(1, 0, "Press 'q' to quit.", width - 1, curses.A_DIM)
            stdscr.refresh()
            key = stdscr.getch()
            if key in (ord("q"), ord("Q")):
                save_state(options)
                return None
            append_log(ui_state, "No options available in this context")
            continue

        if selected < len(options) and selected_visible_idx is None:
            selected = visible_options[0][0]
            selected_visible_idx = 0

        if ui_state.get("last_selected") != selected:
            ui_state["help_offset"] = 0
            ui_state["last_selected"] = selected

        scroll = max(0, min(scroll, max(0, len(visible_options) - 1)))

        message = ui_state.get("status_message")
        first_last = draw_screen(
            stdscr,
            options,
            visible_options,
            selected,
            selected_visible_idx,
            scroll,
            message,
            ui_state,
        )
        ui_state["status_message"] = None

        if selected < len(options) and selected_visible_idx is not None:
            first_rendered, last_rendered = first_last
            if first_rendered is not None and selected_visible_idx < first_rendered:
                scroll = selected_visible_idx
                continue
            if last_rendered is not None and selected_visible_idx > last_rendered:
                span = last_rendered - first_rendered if first_rendered is not None else 0
                if span < 0:
                    span = 0
                scroll = max(0, selected_visible_idx - span)
                continue

        key = stdscr.getch()
        if key in (ord("q"), ord("Q")):
            save_state(options)
            return None
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
        if key in (ord("c"), ord("C")):
            set_compact_list(not compact_list_enabled())
            append_log(ui_state, "Compact list enabled" if compact_list_enabled() else "Compact list disabled")
            continue

        focus_area = ui_state.get("focus_area", 0)

        if focus_area == 0:
            if key in (curses.KEY_DOWN, ord("j")):
                if selected < len(options):
                    if selected_visible_idx is not None and selected_visible_idx < len(visible_options) - 1:
                        selected = visible_options[selected_visible_idx + 1][0]
                    else:
                        selected = len(options)
                else:
                    selected = len(options)
                continue
            if key in (curses.KEY_UP, ord("k")):
                if selected == len(options):
                    if visible_options:
                        selected = visible_options[-1][0]
                    else:
                        selected = 0
                else:
                    if selected_visible_idx is not None and selected_visible_idx > 0:
                        selected = visible_options[selected_visible_idx - 1][0]
                    else:
                        selected = visible_options[0][0]
                continue
            if key in (curses.KEY_PPAGE,):
                scroll = max(0, scroll - max(1, len(visible_options) // 2))
                continue
            if key in (curses.KEY_NPAGE,):
                scroll = min(len(visible_options) - 1, scroll + max(1, len(visible_options) // 2))
                continue
            if selected < len(options):
                opt = options[selected]
                if isinstance(opt, InputOption):
                    opt.handle_key(key, stdscr)
                else:
                    opt.handle_key(key)
                continue
            if selected == len(options) and key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
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
            if key in (curses.KEY_ENTER, ord("\n"), ord("\r")) and selected == len(options):
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
            if key in (curses.KEY_ENTER, ord("\n"), ord("\r")) and selected == len(options):
                break
            continue

    save_state(options)
    config = compile_config(options)
    return _attach_detection_metadata(config, system_info)


def launch_build(config: dict) -> int:
    def emit_section(title: str, details: str) -> None:
        details = textwrap.dedent(details).strip()
        if not details:
            return
        print(f"\n{title}")
        print(textwrap.indent(details, "  "))

    print("\n=== Build plan ===")
    print(f"Target ref: {config.get('ref', 'latest')}")
    print(f"GPU backend: {config.get('backend', 'cuda')}")
    print(f"CPU profile: {config.get('cpu_profile', 'auto')}")
    print(f"BLAS preference: {config.get('blas', 'auto')}")
    if config.get("distributed"):
        print("Distributed RPC: enabled (GGML_RPC=ON)")

    gpu_vendor = config.get("detected_gpu_vendor") or "unknown"
    print(f"Detected GPU vendor: {gpu_vendor}")
    cuda_home = config.get("detected_cuda_home")
    if cuda_home:
        print(f"Detected CUDA toolkit: {cuda_home}")
    else:
        print("Detected CUDA toolkit: not found (set CUDA_HOME or install cuda-toolkit)")

    emit_section("CPU optimisation recipe", cpu_profile_instructions(config.get("cpu_profile", "")))
    emit_section("GPU backend recipe", backend_instructions(config.get("backend", "")))
    emit_section("BLAS setup", blas_instructions(config.get("blas", "")))
    runtime_cmd = runtime_profile_instructions(config.get("runtime_profile", ""))
    if runtime_cmd:
        emit_section("Suggested runtime", runtime_cmd)
    quant_note = quantization_notes(config.get("quantization", ""))
    if quant_note:
        emit_section("Quantisation tip", quant_note)

    if config.get("unified_memory"):
        emit_section("Runtime environment", "export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1")
    if config.get("flash_attention"):
        emit_section("Flash Attention", "Remember to add '-fa' when invoking llama-cli for prompt acceleration.")
    if config.get("force_mmq") and config.get("force_mmq") != "auto":
        emit_section(
            "MMQ override",
            f"GGML_CUDA_FORCE_MMQ will be forced to {config['force_mmq']}. Disable if kernels regress on older GPUs.",
        )

    if not config.get("now", True):
        print("\nBuild immediately is disabled; review the above instructions to run the commands manually.")
        return 0

    if config.get("backend") != "cuda":
        print(
            "\nAutomatic builds currently support CUDA only. Use the guidance above to run the backend-specific commands manually.",
            file=sys.stderr,
        )
        return 0

    cmd: List[str] = [sys.executable, str(AUTO_SCRIPT), "--now"]
    ref = config.get("ref")
    if ref and ref != "latest":
        cmd.extend(["--ref", ref])
    if config.get("fast_math"):
        cmd.append("--fast-math")
    force = config.get("force_mmq")
    if force and force != "auto":
        cmd.extend(["--force-mmq", force])
    blas = config.get("blas")
    if blas and blas != "auto":
        cmd.extend(["--blas", blas])
    if config.get("distributed"):
        cmd.append("--distributed")

    print("\nLaunching autodevops.py with:")
    print("  " + " ".join(cmd))
    return subprocess.call(cmd)


def main() -> None:
    config: dict | None = curses.wrapper(run_wizard)
    if config is None:
        print("Aborted by user")
        return
    status = launch_build(config)
    if status != 0:
        raise SystemExit(status)


if __name__ == "__main__":
    main()
