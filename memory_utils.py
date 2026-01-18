"""Helpers for estimating llama.cpp memory usage for the TUI."""
from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from huggingface_hub import model_info  # type: ignore
    _HF_OK = True
except Exception:  # pragma: no cover - optional dependency
    model_info = None  # type: ignore
    _HF_OK = False


@dataclass
class GPUInfo:
    index: int
    name: str
    total: Optional[int]
    free: Optional[int]


@dataclass
class GPUUsage:
    info: GPUInfo
    weights: int
    kv: int


@dataclass
class MemoryProfile:
    source_label: str
    model_label: str
    quant: Optional[str]
    param_count: Optional[float]
    weights_total: Optional[int]
    weights_gpu: int
    weights_cpu: int
    kv_total: int
    ctx_size: Optional[int]
    layers_est: Optional[int]
    gpus: List[GPUUsage]
    cpu_total: Optional[int]
    cpu_available: Optional[int]
    cpu_weights: int
    cpu_kv: int
    warnings: List[str]


_QUANT_PATTERNS: Tuple[Tuple[str, int], ...] = (
    ("Q2_K", 2),
    ("Q3_K_S", 3),
    ("Q3_K_M", 3),
    ("Q3_K_L", 3),
    ("Q3_K", 3),
    ("Q4_K_S", 4),
    ("Q4_K_M", 4),
    ("Q4_K_L", 4),
    ("Q4_K", 4),
    ("Q4_1", 4),
    ("Q4_0", 4),
    ("Q5_K_S", 5),
    ("Q5_K_M", 5),
    ("Q5_K_L", 5),
    ("Q5_K", 5),
    ("Q5_1", 5),
    ("Q5_0", 5),
    ("Q6_K", 6),
    ("Q6_0", 6),
    ("Q8_0", 8),
    ("IQ2", 2),
    ("IQ3", 3),
    ("IQ4", 4),
    ("IQ4_XS", 4),
    ("IQ4_XL", 4),
    ("IQ5", 5),
    ("IQ6", 6),
    ("F16", 16),
    ("FP16", 16),
    ("F32", 32),
    ("FP32", 32),
)


_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)([bmk])", re.IGNORECASE)


def format_bytes(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return "?"
    if num_bytes < 0:
        num_bytes = 0
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def format_params(params: Optional[float]) -> str:
    if not params:
        return "?"
    if params >= 1e9:
        return f"{params / 1e9:.2f}B"
    if params >= 1e6:
        return f"{params / 1e6:.2f}M"
    if params >= 1e3:
        return f"{params / 1e3:.2f}K"
    return f"{int(params)}"


def _infer_quant_bits(name: str) -> Optional[int]:
    up = name.upper()
    for pattern, bits in _QUANT_PATTERNS:
        if pattern in up:
            return bits
    return None


def _infer_params_from_name(name: str) -> Optional[float]:
    match = _PARAM_RE.search(name)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "b":
        return value * 1e9
    if unit == "m":
        return value * 1e6
    if unit == "k":
        return value * 1e3
    return None


def _estimate_layers(param_count: Optional[float]) -> Optional[int]:
    if not param_count or param_count <= 0:
        return None
    # Empirical scaling for decoder-only transformers (LLaMA/Qwen family).
    layers = int(round((param_count / 196608.0) ** (1.0 / 3.0)))
    return max(layers, 1)


def _detect_system_memory() -> Tuple[Optional[int], Optional[int]]:
    meminfo = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if ":" not in line:
                    continue
                key, rest = line.split(":", 1)
                parts = rest.strip().split()
                if not parts:
                    continue
                try:
                    value = int(parts[0]) * 1024
                except ValueError:
                    continue
                meminfo[key] = value
    except FileNotFoundError:  # pragma: no cover - non-Linux fallback
        try:
            import psutil  # type: ignore

            info = psutil.virtual_memory()
            return int(info.total), int(info.available)
        except Exception:
            return None, None
    total = meminfo.get("MemTotal")
    available = meminfo.get("MemAvailable") or meminfo.get("MemFree")
    return total, available


def detect_gpus() -> List[GPUInfo]:
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        infos: List[GPUInfo] = []
        prev_device: Optional[int] = None
        try:
            if torch.cuda.is_initialized():  # type: ignore[attr-defined]
                prev_device = torch.cuda.current_device()
        except Exception:
            prev_device = None
        for idx in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(idx)
                name = props.name
                total = getattr(props, "total_memory", None)
            except Exception:
                name = f"CUDA {idx}"
                total = None
            free = None
            try:
                torch.cuda.set_device(idx)
                free, total_runtime = torch.cuda.mem_get_info()
                if total is None:
                    total = int(total_runtime)
            except Exception:
                free = None
            infos.append(GPUInfo(index=idx, name=name, total=None if total is None else int(total), free=None if free is None else int(free)))
        if prev_device is not None:
            try:
                torch.cuda.set_device(prev_device)
            except Exception:
                pass
        return infos
    except Exception:
        return []


def parse_device_list(value: str | None, gpus: Optional[List[GPUInfo]] = None) -> tuple[Optional[List[int]], Optional[str]]:
    """Parse a GPU device list (e.g. '0,1' or 'cuda:0,cuda:1'). Returns (indices, error)."""
    raw = str(value or "").strip()
    if not raw:
        return (None, None)
    lowered = raw.lower()
    if lowered in ("all", "auto", "default"):
        return (None, None)
    if lowered in ("none", "cpu", "off"):
        return ([], None)

    tokens = [tok.strip() for tok in re.split(r"[,\s]+", raw) if tok.strip()]
    if not tokens:
        return (None, "GPU device list cannot be empty.")
    indices: List[int] = []
    for token in tokens:
        cleaned = token.lower()
        for prefix in ("cuda:", "gpu:"):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
        if not cleaned.isdigit():
            return (None, f"Invalid GPU entry '{token}'. Use a comma-separated list like 0,1.")
        idx = int(cleaned)
        if idx not in indices:
            indices.append(idx)

    if gpus is not None and gpus:
        valid = {gpu.index for gpu in gpus}
        missing = [str(idx) for idx in indices if idx not in valid]
        if missing:
            return (None, f"GPU index(es) not available: {', '.join(missing)}.")
    return (indices, None)


def filter_gpus_by_selection(
    gpus: List[GPUInfo],
    selection: str | None,
) -> tuple[List[GPUInfo], Optional[str], Optional[List[int]]]:
    indices, err = parse_device_list(selection, gpus)
    if indices is None:
        return gpus, err, None
    selected = [gpu for gpu in gpus if gpu.index in indices]
    return selected, err, indices


def _distribute(total: int, count: int, strategy: str) -> List[int]:
    if count <= 0:
        return []
    if total <= 0:
        return [0 for _ in range(count)]
    if strategy == "single" or count == 1:
        return [total] + [0 for _ in range(count - 1)]
    if strategy == "priority":
        main = min(total, int(total * 0.6))
        remaining = total - main
        per_other = remaining // (count - 1)
        parts = [main]
        for i in range(1, count):
            share = per_other
            if i - 1 < remaining - per_other * (count - 1):
                share += 1
            parts.append(share)
        parts[0] += total - sum(parts)
        return parts
    # balanced default
    base = total // count
    remainder = total - base * count
    parts = [base for _ in range(count)]
    for i in range(remainder):
        parts[i % count] += 1
    return parts


def parse_tensor_split(value: str | None) -> tuple[str, Optional[List[float]], Optional[str]]:
    """Parse a llama.cpp --tensor-split string.

    Returns (kind, ratios, error):
      - kind="none": empty/unspecified
      - kind="auto": literal "auto"
      - kind="ratios": ratios is a list of floats (not normalised)
      - kind="invalid": error is a human-friendly message
    """
    raw = str(value or "").strip()
    if not raw:
        return ("none", None, None)
    if raw.lower() == "auto":
        return ("auto", None, None)

    if "," in raw or ";" in raw:
        tokens = [t.strip() for t in raw.replace(";", ",").split(",") if t.strip()]
    elif ":" in raw:
        tokens = [t.strip() for t in raw.split(":") if t.strip()]
    else:
        tokens = [t.strip() for t in raw.split() if t.strip()]

    if not tokens:
        return ("invalid", None, "--tensor-split cannot be empty.")

    ratios: List[float] = []
    for token in tokens:
        if token.endswith("%"):
            token = token[:-1].strip()
        try:
            number = float(token)
        except ValueError:
            return (
                "invalid",
                None,
                f"Invalid --tensor-split '{raw}'. Use e.g. '0.6,0.4', '60,40', or 'auto'.",
            )
        if number < 0:
            return ("invalid", None, f"--tensor-split values must be >= 0 (got {token}).")
        ratios.append(number)

    if sum(ratios) <= 0:
        return ("invalid", None, f"--tensor-split must contain at least one value > 0 (got '{raw}').")

    return ("ratios", ratios, None)


def auto_tensor_split(
    gpus: Optional[List[GPUInfo]] = None,
    *,
    policy: str = "vram",
) -> Optional[str]:
    """Return a comma-separated split string based on the chosen policy, or None if not applicable."""
    if gpus is None:
        gpus = detect_gpus()
    if not gpus or len(gpus) < 2:
        return None
    policy = (policy or "vram").strip().lower()
    if policy == "even":
        totals = [1.0 for _ in gpus]
    else:
        totals = []
        for gpu in gpus:
            if policy == "free":
                value = gpu.free if gpu.free is not None else gpu.total
            else:
                value = gpu.total
            if value is None or value <= 0:
                value = 1.0
            totals.append(float(value))
    denom = sum(totals)
    if denom <= 0:
        base = 100 // len(gpus)
        parts = [base for _ in range(len(gpus))]
        for idx in range(100 - base * len(gpus)):
            parts[idx % len(gpus)] += 1
    else:
        raw = [(total / denom) * 100.0 for total in totals]
        parts = [int(x) for x in raw]
        remainder = 100 - sum(parts)
        order = sorted(((raw[i] - parts[i], i) for i in range(len(gpus))), reverse=True)
        for offset in range(remainder):
            parts[order[offset % len(gpus)][1]] += 1
    total = sum(parts)
    if total != 100 and total > 0:
        parts[-1] += 100 - total
    return ",".join(str(max(p, 0)) for p in parts)


def _distribute_by_ratios(total: int, ratios: List[float], count: int) -> List[int]:
    if count <= 0:
        return []
    if total <= 0:
        return [0 for _ in range(count)]

    weights = list(ratios[:count])
    if len(weights) < count:
        weights.extend([0.0 for _ in range(count - len(weights))])

    denom = sum(weights)
    if denom <= 0:
        return _distribute(total, count, "balanced")

    raw = [total * (w / denom) for w in weights]
    parts = [int(x) for x in raw]
    remainder = total - sum(parts)
    if remainder > 0:
        order = sorted(((raw[i] - parts[i], i) for i in range(count)), reverse=True)
        for offset in range(remainder):
            parts[order[offset % count][1]] += 1

    diff = total - sum(parts)
    if diff != 0 and parts:
        parts[-1] += diff

    return parts


def _resolve_remote_size(model_ref: str, state: dict) -> Optional[int]:
    if not model_ref or ":" not in model_ref or not _HF_OK:
        return None
    cache: Dict[str, Optional[int]] = state.setdefault("_remote_size_cache", {})
    if model_ref in cache:
        return cache[model_ref]
    repo, filename = model_ref.split(":", 1)
    token = (state.get("hf_token") or "").strip() or None
    size: Optional[int] = None
    if model_info is None:
        cache[model_ref] = None
        return None
    try:
        info = model_info(repo, token=token)
        for sibling in getattr(info, "siblings", []):
            if getattr(sibling, "rfilename", None) == filename:
                sz = getattr(sibling, "size", None)
                if sz is not None:
                    size = int(sz)
                    break
    except Exception:
        size = None
    cache[model_ref] = size
    return size


def _model_display_name(source: str, state: dict, model_ref: str) -> Tuple[str, Optional[Path]]:
    models_dir = Path(state.get("models_dir", ".")).expanduser()
    local_choice = (state.get("selected_local_model") or "").strip()
    if source == "local":
        if local_choice:
            candidate = (models_dir / local_choice).expanduser()
            if candidate.exists():
                return local_choice, candidate
        if model_ref:
            path = Path(model_ref).expanduser()
            if path.exists():
                return path.name, path
        return local_choice or (Path(model_ref).name if model_ref else "(no model selected)"), None
    return model_ref or "(no model selected)", None


def _bytes_per_param(bits: Optional[int]) -> Optional[float]:
    if not bits:
        return None
    return bits / 8.0


def estimate_memory_profile(state: dict, *, refresh: bool = False) -> MemoryProfile:
    cache_key = (
        state.get("mode"),
        state.get("model_source"),
        state.get("model_ref"),
        state.get("selected_local_model"),
        state.get("models_dir"),
        state.get("ctx_size"),
        state.get("n_gpu_layers"),
        state.get("tensor_split"),
        state.get("gpu_strategy"),
        state.get("gpu_devices"),
        state.get("auto_split_policy"),
        state.get("hf_token"),
    )
    cache = state.get("_memory_profile_cache")
    if cache and not refresh and cache.get("key") == cache_key:
        profile = cache.get("profile")
        if isinstance(profile, MemoryProfile):
            return profile

    warnings: List[str] = []
    source = state.get("model_source", "local")
    model_ref = str(state.get("model_ref") or "").strip()
    model_label, model_path = _model_display_name(source, state, model_ref)

    quant_bits = _infer_quant_bits(model_label or model_ref)
    quant_tag = None
    if quant_bits is not None:
        for pattern, bits in _QUANT_PATTERNS:
            if bits == quant_bits and pattern in (model_label or model_ref).upper():
                quant_tag = pattern
                break
    param_count = _infer_params_from_name(model_label or model_ref)

    file_size: Optional[int] = None
    if model_path is not None and model_path.exists():
        try:
            file_size = model_path.stat().st_size
        except OSError:
            file_size = None
    if file_size is None and source == "remote":
        remote_size = _resolve_remote_size(model_ref, state)
        if remote_size:
            file_size = remote_size
        elif model_ref:
            warnings.append("Unable to determine remote file size; estimates may be rough.")

    if file_size and (quant_bits is None or param_count is None):
        # Back-compute missing attributes when possible.
        if quant_bits is not None and param_count is None:
            param_count = (file_size * 8.0) / float(quant_bits)
        elif quant_bits is None and param_count is not None and param_count > 0:
            quant_bits = max(int(round((file_size * 8.0) / param_count)), 1)
        elif quant_bits is None and param_count is None:
            param_count = file_size / 2.0  # fallback guess

    bytes_per_param = _bytes_per_param(quant_bits)
    weights_total: Optional[int]
    if param_count and bytes_per_param:
        weights_total = int(param_count * bytes_per_param)
    else:
        weights_total = file_size
    if weights_total is None:
        warnings.append("Model size unknown; memory planner will be limited.")
        weights_total = 0

    layers_est = _estimate_layers(param_count)
    if layers_est is None:
        warnings.append("Could not infer layer count; assuming all layers on GPU when allowed.")
    ctx_value = str(state.get("ctx_size") or "").strip()
    ctx_size: Optional[int]
    if ctx_value:
        try:
            ctx_size = max(int(ctx_value), 1)
        except ValueError:
            warnings.append("Context size is not an integer; using 4096.")
            ctx_size = 4096
    else:
        ctx_size = 4096

    hidden_dim = layers_est * 128 if layers_est else None
    kv_total = 0
    if hidden_dim and ctx_size and layers_est:
        kv_total = int(ctx_size * layers_est * hidden_dim * 4)
    elif ctx_size:
        kv_total = int(ctx_size * 1024 * 16)
        warnings.append("KV cache estimated using generic transformer scaling.")

    n_gpu_layers_raw = str(state.get("n_gpu_layers") or "").strip()
    gpu_ratio = 1.0
    if source == "remote" and not model_ref:
        gpu_ratio = 0.0
    if n_gpu_layers_raw:
        try:
            ngl_value = int(n_gpu_layers_raw)
        except ValueError:
            warnings.append("--n-gpu-layers is not an integer; using automatic placement.")
            ngl_value = 999
    else:
        ngl_value = 999
    if ngl_value <= 0:
        gpu_ratio = 0.0
    elif layers_est and ngl_value not in (999, 1000):
        gpu_ratio = min(max(ngl_value / float(layers_est), 0.0), 1.0)
    elif layers_est is None and ngl_value not in (999, 1000):
        warnings.append("Layer count unknown; treating --n-gpu-layers as full offload.")
        gpu_ratio = 1.0 if ngl_value > 0 else 0.0

    gpus_all = detect_gpus()
    gpus, selection_err, selection_indices = filter_gpus_by_selection(
        gpus_all,
        state.get("gpu_devices"),
    )
    if selection_err:
        warnings.append(selection_err)
    if selection_indices == []:
        warnings.append("No CUDA GPUs selected; weights will remain on system RAM.")
    strategy = state.get("gpu_strategy") or ("cpu" if not gpus else "single")
    if not gpus and strategy != "cpu" and gpu_ratio > 0:
        warnings.append("No CUDA GPUs available; weights will remain on system RAM.")
        gpu_ratio = 0.0
    weights_gpu = int(weights_total * gpu_ratio)
    weights_cpu = max(weights_total - weights_gpu, 0)
    kv_gpu = int(kv_total * gpu_ratio) if kv_total > 0 else 0
    kv_cpu = max(kv_total - kv_gpu, 0)

    if strategy == "cpu" or not gpus:
        weights_gpu = 0
        weights_cpu = weights_total
        kv_gpu = 0
        kv_cpu = kv_total
    weights_gpu = min(weights_gpu, weights_total)
    kv_gpu = min(kv_gpu, kv_total)

    split_raw = str(state.get("tensor_split") or "").strip()
    split_kind, split_ratios, split_err = parse_tensor_split(split_raw)
    if split_kind == "invalid" and split_err:
        warnings.append(split_err)
        split_kind = "none"

    ratio_weights: Optional[List[float]] = None
    if split_kind == "auto" and gpus:
        policy = str(state.get("auto_split_policy") or "vram")
        if policy.strip().lower() == "even":
            ratio_weights = [1.0 for _ in gpus]
        else:
            ratio_weights = []
            for gpu in gpus:
                if policy.strip().lower() == "free":
                    value = gpu.free if gpu.free is not None else gpu.total
                else:
                    value = gpu.total
                if value is None or value <= 0:
                    value = 1.0
                ratio_weights.append(float(value))
    elif split_kind == "ratios" and split_ratios:
        ratio_weights = split_ratios

    if ratio_weights is not None and gpus and strategy != "cpu":
        if len(ratio_weights) != len(gpus):
            warnings.append(
                f"--tensor-split has {len(ratio_weights)} value(s) but {len(gpus)} GPU(s) detected; estimates may be off."
            )
        distribution = _distribute_by_ratios(weights_gpu, ratio_weights, len(gpus))
        kv_distribution = _distribute_by_ratios(kv_gpu, ratio_weights, len(gpus))
    else:
        distribution = _distribute(weights_gpu, len(gpus), strategy)
        kv_distribution = _distribute(kv_gpu, len(gpus), strategy)
    gpu_usages: List[GPUUsage] = []
    for idx, gpu in enumerate(gpus):
        weights_share = distribution[idx] if idx < len(distribution) else 0
        kv_share = kv_distribution[idx] if idx < len(kv_distribution) else 0
        gpu_usages.append(GPUUsage(info=gpu, weights=weights_share, kv=kv_share))

    cpu_total, cpu_available = _detect_system_memory()
    cpu_kv = kv_cpu
    cpu_usage_profile = MemoryProfile(
        source_label="Local" if source == "local" else "Remote",
        model_label=model_label,
        quant=quant_tag,
        param_count=param_count,
        weights_total=weights_total,
        weights_gpu=weights_gpu,
        weights_cpu=weights_cpu,
        kv_total=kv_total,
        ctx_size=ctx_size,
        layers_est=layers_est,
        gpus=gpu_usages,
        cpu_total=cpu_total,
        cpu_available=cpu_available,
        cpu_weights=weights_cpu,
        cpu_kv=cpu_kv,
        warnings=warnings,
    )

    state["_memory_profile_cache"] = {"key": cache_key, "profile": cpu_usage_profile}
    return cpu_usage_profile


def clear_cached_profile(state: dict) -> None:
    state.pop("_memory_profile_cache", None)


def profile_summary_lines(profile: MemoryProfile) -> List[str]:
    lines: List[str] = []
    summary = f"{profile.source_label}: {profile.model_label}"
    bits = []
    if profile.quant:
        bits.append(profile.quant)
    if profile.param_count:
        bits.append(f"≈{format_params(profile.param_count)} params")
    if bits:
        summary += " (" + ", ".join(bits) + ")"
    lines.append(summary)
    if profile.weights_total is not None:
        lines.append(
            f"Weights {format_bytes(profile.weights_total)} → GPU {format_bytes(profile.weights_gpu)} / CPU {format_bytes(profile.weights_cpu)}"
        )
    if profile.kv_total:
        ctx_desc = profile.ctx_size or "auto"
        kv_gpu = sum(gpu.kv for gpu in profile.gpus)
        if kv_gpu or profile.cpu_kv:
            lines.append(
                f"KV cache @ ctx {ctx_desc}: {format_bytes(profile.kv_total)} → GPU {format_bytes(kv_gpu)} / CPU {format_bytes(profile.cpu_kv)}"
            )
        else:
            lines.append(f"KV cache @ ctx {ctx_desc}: {format_bytes(profile.kv_total)}")
    return lines


def warning_lines(profile: MemoryProfile) -> Iterable[str]:
    for warn in profile.warnings:
        yield from textwrap.wrap(warn, 80)
