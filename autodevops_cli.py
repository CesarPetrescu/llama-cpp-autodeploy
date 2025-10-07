#!/usr/bin/env python3
"""Interactive TUI front-end for autodevops.py builds."""
from __future__ import annotations

import curses
import curses.textpad
import locale
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import autodevops

SCRIPT_DIR = Path(__file__).resolve().parent
AUTO_SCRIPT = SCRIPT_DIR / "autodevops.py"

locale.setlocale(locale.LC_ALL, "")


@dataclass
class ChoiceValue:
    label: str
    value: str
    enabled: bool = True
    reason: str | None = None


class OptionBase:
    name: str
    description: str
    disabled: bool
    reason: str | None

    def render(self, win: "curses._CursesWindow", y: int, width: int, selected: bool) -> int:
        raise NotImplementedError

    def handle_key(self, key: int) -> None:
        pass

    def height(self, width: int) -> int:
        return 1


class ToggleOption(OptionBase):
    def __init__(self, name: str, description: str, value: bool = False, *, disabled: bool = False, reason: str | None = None) -> None:
        self.name = name
        self.description = description
        self.value = value
        self.disabled = disabled
        self.reason = reason

    def toggle(self) -> None:
        if not self.disabled:
            self.value = not self.value

    def handle_key(self, key: int) -> None:
        if key in (curses.KEY_ENTER, ord(" "), ord("t")):
            self.toggle()

    def render(self, win: "curses._CursesWindow", y: int, width: int, selected: bool) -> int:
        attr = curses.A_REVERSE if selected else curses.A_NORMAL
        if self.disabled:
            attr |= curses.color_pair(2)
        label = f"[{'x' if self.value else ' '}] {self.name}"
        win.addnstr(y, 2, label, width - 4, attr)
        line_count = 1
        desc_attr = curses.A_DIM
        if self.disabled:
            desc_attr |= curses.color_pair(2)
        for line in textwrap.wrap(self.description, width - 6):
            win.addnstr(y + line_count, 6, line, width - 8, desc_attr)
            line_count += 1
        if self.disabled and self.reason:
            reason = f"⚠ {self.reason}"
            win.addnstr(y + line_count, 6, reason, width - 8, curses.color_pair(2) | curses.A_DIM)
            line_count += 1
        return line_count

    def height(self, width: int) -> int:
        base = 1 + len(textwrap.wrap(self.description, width - 6))
        if self.disabled and self.reason:
            base += 1
        return base


class ChoiceOption(OptionBase):
    def __init__(self, name: str, description: str, choices: Sequence[ChoiceValue], *, initial: str | None = None) -> None:
        if not choices:
            raise ValueError("choices cannot be empty")
        self.name = name
        self.description = description
        self.choices = list(choices)
        self.disabled = all(not c.enabled for c in self.choices)
        self.reason = None
        if self.disabled:
            self.reason = "No enabled options"
        self.index = 0
        if initial is not None:
            for idx, c in enumerate(self.choices):
                if c.value == initial:
                    self.index = idx
                    break
        if not self.choices[self.index].enabled:
            self._select_next_enabled(1)

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
        label = f"{self.name}: {current.label}"
        win.addnstr(y, 2, label, width - 4, attr)
        line_count = 1
        desc_attr = curses.A_DIM
        if self.disabled:
            desc_attr |= curses.color_pair(2)
        for line in textwrap.wrap(self.description, width - 6):
            win.addnstr(y + line_count, 6, line, width - 8, desc_attr)
            line_count += 1
        if not current.enabled and current.reason:
            reason = f"⚠ {current.reason}"
            win.addnstr(y + line_count, 6, reason, width - 8, curses.color_pair(2) | curses.A_DIM)
            line_count += 1
        return line_count

    def height(self, width: int) -> int:
        base = 1 + len(textwrap.wrap(self.description, width - 6))
        current = self.choices[self.index]
        if not current.enabled and current.reason:
            base += 1
        return base

    @property
    def value(self) -> ChoiceValue:
        return self.choices[self.index]


class InputOption(OptionBase):
    def __init__(self, name: str, description: str, value: str, placeholder: str = "") -> None:
        self.name = name
        self.description = description
        self.value = value
        self.placeholder = placeholder
        self.disabled = False
        self.reason = None

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
        if self.value:
            edit_win.addstr(0, 0, self.value)
        curses.curs_set(1)
        textpad = curses.textpad.Textbox(edit_win)
        win.refresh()
        try:
            new_value = textpad.edit().strip()
        except Exception:
            new_value = self.value
        curses.curs_set(0)
        if new_value:
            self.value = new_value

    def handle_key(self, key: int, stdscr: "curses._CursesWindow" | None = None) -> None:
        if key in (curses.KEY_ENTER, ord("\n"), ord("\r")) and stdscr is not None:
            self.edit(stdscr)

    def render(self, win: "curses._CursesWindow", y: int, width: int, selected: bool) -> int:
        attr = curses.A_REVERSE if selected else curses.A_NORMAL
        display = self.value or self.placeholder
        win.addnstr(y, 2, f"{self.name}: {display}", width - 4, attr)
        line_count = 1
        for line in textwrap.wrap(self.description, width - 6):
            win.addnstr(y + line_count, 6, line, width - 8, curses.A_DIM)
            line_count += 1
        return line_count

    def height(self, width: int) -> int:
        return 1 + len(textwrap.wrap(self.description, width - 6))


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


def build_options() -> List[OptionBase]:
    cpu_vendor = detect_cpu_vendor()
    gpu_vendor = detect_gpu_vendor()
    cuda_home = autodevops.pick_cuda_home()
    has_cuda = cuda_home is not None
    fast_math_disabled = not has_cuda
    fast_math_reason = None
    if fast_math_disabled:
        fast_math_reason = "NVCC not found on this system"

    blas_choices = [
        ChoiceValue("Auto", "auto"),
        ChoiceValue("Intel oneAPI MKL", "mkl", autodevops.mkl_present(), "Intel MKL libraries not detected"),
        ChoiceValue("OpenBLAS", "openblas", autodevops.openblas_present(), "OpenBLAS libraries not detected"),
        ChoiceValue("Disabled", "off"),
    ]

    backend_choices = [
        ChoiceValue("CUDA (NVIDIA)", "cuda", has_cuda, "CUDA Toolkit not found" if not has_cuda else None),
        ChoiceValue("ROCm (AMD)", "rocm", False, "ROCm builds not supported by this tool"),
        ChoiceValue("oneAPI / SYCL (Intel)", "oneapi", False, "oneAPI GPU builds not supported"),
    ]

    options: List[OptionBase] = [
        InputOption(
            "Target ref",
            "Git tag, branch, or commit to build. Use 'latest' to fetch the newest release.",
            value="latest",
            placeholder="latest",
        ),
        ToggleOption(
            "Build immediately",
            "Run the build as soon as you exit this wizard.",
            value=True,
        ),
        ToggleOption(
            "Enable fast math",
            "Adds --use_fast_math to NVCC for potential speedups at the cost of precision.",
            value=False,
            disabled=fast_math_disabled,
            reason=fast_math_reason,
        ),
        ChoiceOption(
            "Force MMQ kernels",
            "Controls GGML_CUDA_FORCE_MMQ. 'Auto' enables it on newer NVIDIA GPUs.",
            [
                ChoiceValue("Auto", "auto"),
                ChoiceValue("On", "on"),
                ChoiceValue("Off", "off"),
            ],
        ),
        ChoiceOption(
            "CPU BLAS backend",
            "Selects BLAS acceleration for CPU paths.",
            blas_choices,
            initial="auto",
        ),
        ChoiceOption(
            "GPU backend",
            "Available GPU accelerators based on detected hardware.",
            backend_choices,
        ),
    ]

    if cpu_vendor == "intel":
        options.append(
            ToggleOption(
                "Intel CPU detected",
                "Optimised Intel builds (MKL) available when libraries are installed.",
                value=True,
                disabled=True,
            )
        )
    elif cpu_vendor == "amd":
        options.append(
            ToggleOption(
                "AMD CPU detected",
                "Consider installing OpenBLAS for optimal CPU performance.",
                value=True,
                disabled=True,
            )
        )

    if gpu_vendor == "nvidia":
        options.append(
            ToggleOption(
                "NVIDIA GPU detected",
                "CUDA builds are available on this system.",
                value=True,
                disabled=True,
            )
        )
    elif gpu_vendor == "amd":
        options.append(
            ToggleOption(
                "AMD GPU detected",
                "ROCm builds are not yet automated by this wizard.",
                value=True,
                disabled=True,
            )
        )
    elif gpu_vendor == "intel":
        options.append(
            ToggleOption(
                "Intel GPU detected",
                "Intel oneAPI GPU builds are not currently supported.",
                value=True,
                disabled=True,
            )
        )

    return options


def draw_screen(stdscr: "curses._CursesWindow", options: List[OptionBase], selected: int, message: str | None) -> None:
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    title = "llama.cpp AutodevOps Builder"
    stdscr.addnstr(0, max(0, (width - len(title)) // 2), title, width - 1, curses.A_BOLD)
    instructions = "Arrows: navigate • Space: toggle/cycle • Enter: edit/apply • Q: quit"
    stdscr.addnstr(1, max(0, (width - len(instructions)) // 2), instructions, width - 1, curses.A_DIM)
    y = 3
    for idx, opt in enumerate(options):
        if y >= height - 4:
            break
        extra = opt.render(stdscr, y, width - 4, selected == idx)
        y += extra
        if y >= height - 4:
            break
        y += 1
    confirm_attr = curses.A_BOLD | (curses.A_REVERSE if selected == len(options) else 0)
    confirm_text = "Start build"
    stdscr.addnstr(height - 3, max(2, (width - len(confirm_text)) // 2), confirm_text, len(confirm_text), confirm_attr)
    if message:
        for offset, line in enumerate(textwrap.wrap(message, width - 4)):
            if height - 2 + offset < height:
                stdscr.addnstr(height - 2 + offset, 2, line, width - 4, curses.A_DIM)
    stdscr.refresh()


def run_wizard(stdscr: "curses._CursesWindow") -> dict | None:
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)
    curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)

    options = build_options()
    selected = 0
    message: str | None = None
    while True:
        draw_screen(stdscr, options, selected, message)
        key = stdscr.getch()
        message = None
        if key in (ord("q"), ord("Q")):
            return None
        if key in (curses.KEY_RESIZE,):
            continue
        if key in (curses.KEY_DOWN, ord("j")):
            selected = min(selected + 1, len(options))
        elif key in (curses.KEY_UP, ord("k")):
            selected = max(selected - 1, 0)
        elif selected < len(options):
            opt = options[selected]
            if isinstance(opt, InputOption):
                opt.handle_key(key, stdscr)
            else:
                opt.handle_key(key)
        else:
            if key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
                break
    config = {
        "ref": options[0].value if isinstance(options[0], InputOption) else "latest",
        "now": isinstance(options[1], ToggleOption) and options[1].value,
        "fast_math": isinstance(options[2], ToggleOption) and options[2].value,
        "force_mmq": options[3].value.value if isinstance(options[3], ChoiceOption) else "auto",
        "blas": options[4].value.value if isinstance(options[4], ChoiceOption) else "auto",
        "backend": options[5].value.value if isinstance(options[5], ChoiceOption) else "cuda",
    }
    return config


def launch_build(config: dict) -> int:
    cmd: List[str] = [sys.executable, str(AUTO_SCRIPT)]
    if config.get("now", True):
        cmd.append("--now")
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
    if config.get("backend") != "cuda":
        print("Selected backend is not currently supported by autodevops.py", file=sys.stderr)
        return 1
    print("Running:", " ".join(cmd))
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
