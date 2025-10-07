#!/usr/bin/env python3
"""Interactive TUI front-end for autodevops.py builds."""
from __future__ import annotations

import curses
import curses.textpad
import locale
import platform
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Set

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


class OptionBase:
    key: str
    name: str
    description: str
    help_text: str
    disabled: bool
    reason: str | None

    def render(self, win: "curses._CursesWindow", y: int, width: int, selected: bool) -> int:
        raise NotImplementedError

    def handle_key(self, key: int) -> None:
        pass

    def height(self, width: int) -> int:
        return 1

    def get_help(self) -> str:
        return self.help_text or self.description


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
    ) -> None:
        self.key = key
        self.name = name
        self.description = description
        self.help_text = help_text or description
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
    def __init__(
        self,
        key: str,
        name: str,
        description: str,
        choices: Sequence[ChoiceValue],
        *,
        initial: str | None = None,
        help_text: str = "",
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
    def __init__(self, key: str, name: str, description: str, value: str, placeholder: str = "", help_text: str = "") -> None:
        self.key = key
        self.name = name
        self.description = description
        self.help_text = help_text or description
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
        fast_math_reason = "NVCC not found on this system"

    blas_choices = [
        ChoiceValue("Auto", "auto"),
        ChoiceValue("Intel oneAPI MKL", "mkl", info.has_mkl, "Intel MKL libraries not detected"),
        ChoiceValue("OpenBLAS", "openblas", info.has_openblas, "OpenBLAS libraries not detected"),
        ChoiceValue("AMD BLIS", "blis", info.has_blis, "AMD BLIS libraries not detected"),
        ChoiceValue("Disabled", "off"),
    ]

    backend_choices = [
        ChoiceValue("CUDA (NVIDIA)", "cuda", has_cuda, "CUDA Toolkit not found" if not has_cuda else None),
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
        ChoiceOption(
            "backend",
            "GPU backend",
            "Available GPU accelerators based on detected hardware.",
            backend_choices,
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
            help_text=(
                "FP16 maximises quality on capable GPUs. INT8 (Q8_0) balances speed and size."
                " Q4_K_M keeps small VRAM footprints with acceptable quality for chatbots."
            ),
        ),
    ]

    # Informational badges
    if cpu_vendor == "intel":
        options.append(
            ToggleOption(
                "_info_cpu",
                "Intel CPU detected",
                "Optimised Intel builds with MKL are available when libraries are installed.",
                value=True,
                disabled=True,
                help_text=(
                    "Detected Intel CPU. Consider the advanced MKL recipe:\n"
                    "  source /opt/intel/oneapi/setvars.sh\n"
                    "  cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Intel10_64lp\n"
                    "        -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx\n"
                    "        -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_AVX512=ON\n"
                    "        -DCMAKE_C_FLAGS=\"-O3 -ipo -static -fp-model=fast -march=native\""
                ),
            )
        )
    elif cpu_vendor == "amd":
        options.append(
            ToggleOption(
                "_info_cpu",
                "AMD CPU detected",
                "Install OpenBLAS/BLIS for the best CPU throughput.",
                value=True,
                disabled=True,
                help_text=(
                    "Detected AMD CPU. Recommended build:\n"
                    "  cmake -B build -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_AVX_VNNI=ON\n"
                    "        -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=BLIS\n"
                    "        -DCMAKE_C_FLAGS=\"-O3 -march=native -mavx2 -mcpu=native\""
                ),
            )
        )
    elif is_arm:
        options.append(
            ToggleOption(
                "_info_cpu",
                "ARM64 CPU detected",
                "Apple/ARM builds can enable Metal or Accelerate backends automatically.",
                value=True,
                disabled=True,
                help_text=(
                    "On Apple Silicon run: cmake -B build -DGGML_METAL=ON and make llama-cli."
                ),
            )
        )

    if gpu_vendor == "nvidia":
        options.append(
            ToggleOption(
                "_info_gpu",
                "NVIDIA GPU detected",
                "CUDA builds are available on this system.",
                value=True,
                disabled=True,
                help_text=(
                    "Recommended CUDA command:\n"
                    "  cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_F16=ON\n"
                    "        -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 -DCMAKE_CUDA_ARCHITECTURES=native"
                ),
            )
        )
    elif gpu_vendor == "amd":
        options.append(
            ToggleOption(
                "_info_gpu",
                "AMD GPU detected",
                "ROCm builds are supported manually via hipcc.",
                value=True,
                disabled=True,
                help_text=(
                    "Export CC=/opt/rocm/llvm/bin/clang and run cmake -DGGML_HIP=ON -DAMDGPU_TARGETS=native."
                ),
            )
        )
    elif gpu_vendor == "intel":
        options.append(
            ToggleOption(
                "_info_gpu",
                "Intel GPU detected",
                "Use oneAPI SYCL builds with icx/icpx compilers.",
                value=True,
                disabled=True,
                help_text=(
                    "source /opt/intel/oneapi/setvars.sh and cmake -DGGML_SYCL=ON -DGGML_SYCL_F16=ON."
                ),
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


def draw_screen(stdscr: "curses._CursesWindow", options: List[OptionBase], selected: int, message: str | None) -> None:
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    title = "llama.cpp AutodevOps Builder"
    stdscr.addnstr(0, max(0, (width - len(title)) // 2), title, width - 1, curses.A_BOLD)
    instructions = "Arrows: navigate • Space: toggle/cycle • Enter: edit/apply • Q: quit"
    stdscr.addnstr(1, max(0, (width - len(instructions)) // 2), instructions, width - 1, curses.A_DIM)

    y = 3
    if message:
        message_lines = textwrap.wrap(message, width - 4)
        for idx, line in enumerate(message_lines[:3]):
            stdscr.addnstr(2 + idx, 2, line, width - 4, curses.A_BOLD | curses.color_pair(2))
        y = max(y, 3 + len(message_lines[:3]))

    selected_help = ""
    if selected < len(options):
        selected_help = options[selected].get_help()
    elif selected == len(options):
        selected_help = "Start the build with the currently selected presets."
    else:
        selected_help = ""

    max_help_lines = max(4, min(10, height // 2))
    help_lines = textwrap.wrap(selected_help or "", width - 4) or [""]
    help_lines = help_lines[:max_help_lines]
    help_height = len(help_lines) + 2

    body_bottom = height - help_height - 2
    y = max(y, 3)
    for idx, opt in enumerate(options):
        if y >= body_bottom:
            break
        extra = opt.render(stdscr, y, width - 4, selected == idx)
        y += extra
        if y >= body_bottom:
            break
        y += 1

    stdscr.hline(body_bottom, 1, curses.ACS_HLINE, width - 2)
    stdscr.addnstr(body_bottom, 3, " Help ", width - 6, curses.A_DIM | curses.A_BOLD)
    for idx, line in enumerate(help_lines):
        stdscr.addnstr(body_bottom + 1 + idx, 2, line, width - 4, curses.A_DIM)

    confirm_attr = curses.A_BOLD | (curses.A_REVERSE if selected == len(options) else 0)
    confirm_text = "Start build"
    stdscr.addnstr(height - 2, max(2, (width - len(confirm_text)) // 2), confirm_text, len(confirm_text), confirm_attr)

    if message:
        message_lines = textwrap.wrap(message, width - 4)
        for idx, line in enumerate(message_lines[:2]):
            stdscr.addnstr(height - 4 - idx, 2, line, width - 4, curses.A_BOLD | curses.color_pair(2))

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
    return compile_config(options)


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
