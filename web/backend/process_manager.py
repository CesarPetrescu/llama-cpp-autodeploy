"""Lifecycle manager for managed llama-server (and autodevops build) subprocesses."""
from __future__ import annotations

import asyncio
import os
import shlex
import signal
import sys
import time
import uuid
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import REPO_ROOT, WebConfig, ensure_log_dir
from .log_buffer import LogBuffer, read_log_tail
from .state import BuildRecord, InstanceRecord, StateStore

# Import the reusable helpers from the existing scripts. These imports assume
# the backend is launched from the repo root so `loadmodel` etc. are importable.
import loadmodel  # type: ignore  # noqa: E402
import memory_utils  # type: ignore  # noqa: E402

LLAMA_SERVER = REPO_ROOT / "bin" / "llama-server"
AUTODEVOPS_SCRIPT = REPO_ROOT / "autodevops.py"
LOG_BUFFER_CAPACITY = 2000
LOG_TAIL_LINES_DEFAULT = 500
FOLLOW_POLL_INTERVAL = 0.2
INSTANCE_ACTIVE_STATUSES = {"running", "stopping"}
BUILD_ACTIVE_STATUSES = {"running", "cancelling"}
RECOVERY_MARKER_ENV = "LLAMA_AUTODEPLOY_MANAGED"
RECOVERY_INSTANCE_ID_ENV = "LLAMA_AUTODEPLOY_INSTANCE_ID"
RECOVERY_INSTANCE_NAME_ENV = "LLAMA_AUTODEPLOY_INSTANCE_NAME"
RECOVERY_LOG_FILE_ENV = "LLAMA_AUTODEPLOY_LOG_FILE"


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _resolve_gpu_device_selection(config: Dict[str, Any]) -> Any:
    selection = str(config.get("gpu_devices") or "").strip()
    strategy = str(config.get("gpu_strategy") or "").strip().lower()
    if not selection and strategy == "cpu":
        selection = "cpu"
    return selection


def resolve_selected_gpus(
    config: Dict[str, Any],
) -> tuple[List[memory_utils.GPUInfo], Optional[str], Optional[List[int]]]:
    gpus_all = memory_utils.detect_gpus()
    selection = _resolve_gpu_device_selection(config)
    return memory_utils.filter_gpus_by_selection(gpus_all, selection)


def build_cuda_visible_devices(config: Dict[str, Any]) -> Optional[str]:
    gpus_all = memory_utils.detect_gpus()
    selection = _resolve_gpu_device_selection(config)
    indices, err = memory_utils.parse_device_list(selection, gpus_all)
    if err:
        raise ValueError(err)
    if indices is None:
        return None
    if not indices:
        return ""
    return ",".join(str(idx) for idx in indices)


def build_llama_server_cmd(config: Dict[str, Any], model_path: Path) -> List[str]:
    """Translate an instance config dict into a llama-server command line."""
    host = str(config.get("host") or "127.0.0.1")
    port = _coerce_int(config.get("port")) or 45540
    n_gpu_layers = _coerce_int(config.get("n_gpu_layers"))
    gpu_strategy = str(config.get("gpu_strategy") or "").strip().lower()
    auto_split_policy = str(config.get("auto_split_policy") or "vram").strip().lower() or "vram"
    tensor_split = (str(config.get("tensor_split") or "")).strip() or None
    split_mode = (str(config.get("split_mode") or "")).strip() or None
    if split_mode and split_mode.lower() == "default":
        split_mode = None
    selected_gpus, selection_err, _selected_indices = resolve_selected_gpus(config)
    if selection_err:
        raise ValueError(selection_err)
    if gpu_strategy == "cpu":
        n_gpu_layers = 0
        tensor_split = None
    ctx_size = _coerce_int(config.get("ctx_size"))
    n_cpu_moe = _coerce_int(config.get("n_cpu_moe"))
    cpu_moe = _coerce_bool(config.get("cpu_moe"))
    mmproj = (str(config.get("mmproj") or "")).strip() or None
    jinja = _coerce_bool(config.get("jinja"))
    reasoning_format = (str(config.get("reasoning_format") or "")).strip() or None
    no_context_shift = _coerce_bool(config.get("no_context_shift"))

    cmd: List[str] = [
        str(LLAMA_SERVER),
        "--model", str(model_path),
        "--host", host,
        "--port", str(port),
    ]
    if n_gpu_layers is not None and n_gpu_layers > 0:
        cmd += ["--n-gpu-layers", str(n_gpu_layers)]
    if split_mode:
        cmd += ["--split-mode", split_mode]

    if tensor_split:
        kind, _ratios, err = memory_utils.parse_tensor_split(tensor_split)
        if kind == "invalid":
            raise ValueError(err or "Invalid tensor_split")
        if kind == "auto":
            tensor_split = memory_utils.auto_tensor_split(selected_gpus, policy=auto_split_policy)
        if tensor_split:
            cmd += ["--tensor-split", tensor_split]
    if ctx_size:
        cmd += ["--ctx-size", str(ctx_size)]
    if cpu_moe:
        cmd.append("--cpu-moe")
    elif n_cpu_moe is not None and n_cpu_moe > 0:
        cmd += ["--n-cpu-moe", str(n_cpu_moe)]
    if mmproj:
        cmd += ["--mmproj", str(mmproj)]
    if jinja:
        cmd.append("--jinja")
    if reasoning_format:
        cmd += ["--reasoning-format", reasoning_format]
    if no_context_shift:
        cmd.append("--no-context-shift")

    mode = str(config.get("mode") or "llm").lower()
    extra_raw = config.get("extra_flags") or ""
    if isinstance(extra_raw, list):
        extra: List[str] = [str(x) for x in extra_raw]
    else:
        import shlex
        extra = shlex.split(str(extra_raw))
    if mode == "embed" and "--embeddings" not in extra:
        extra = ["--embeddings"] + extra
    if "--flash-attn" not in extra and "-fa" not in extra:
        extra = ["--flash-attn", "on"] + extra

    primary, _secondary = loadmodel.detect_ubatch_flag()
    rewritten: List[str] = []
    i = 0
    while i < len(extra):
        tok = extra[i]
        if tok in ("--ubatch", "--n-ubatch"):
            val = extra[i + 1] if (i + 1 < len(extra) and not str(extra[i + 1]).startswith("--")) else None
            if primary is None:
                i += 2 if val else 1
                continue
            rewritten.append(primary)
            if val:
                rewritten.append(val)
            i += 2 if val else 1
            continue
        rewritten.append(tok)
        i += 1
    cmd += rewritten
    return cmd


# ---------------------------------------------------------------------------
# Event broadcaster
# ---------------------------------------------------------------------------


class EventBroadcaster:
    """Tiny pub-sub that fans JSON-ready payloads out to subscribers."""

    def __init__(self) -> None:
        self._subscribers: "set[asyncio.Queue[dict]]" = set()

    def subscribe(self) -> "asyncio.Queue[dict]":
        queue: "asyncio.Queue[dict]" = asyncio.Queue(maxsize=64)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: "asyncio.Queue[dict]") -> None:
        self._subscribers.discard(queue)

    def publish(self, payload: Dict[str, Any]) -> None:
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                    queue.put_nowait(payload)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_path(path: Optional[str]) -> Optional[Path]:
    return Path(path) if path else None


def _file_size(path: Optional[Path]) -> int:
    if path is None:
        return 0
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _cmdline_matches(pid: int, expected: List[str]) -> bool:
    if pid <= 0 or not expected:
        return False
    return _read_proc_cmdline(pid) == list(expected)


def _pgid_for_pid(pid: Optional[int]) -> Optional[int]:
    if not pid or pid <= 0:
        return None
    try:
        return os.getpgid(pid)
    except OSError:
        return None


def _safe_pgid_for_control(pid: Optional[int]) -> Optional[int]:
    pgid = _pgid_for_pid(pid)
    if pgid is None:
        return None
    try:
        current_pgid = os.getpgid(0)
    except OSError:
        current_pgid = None
    if current_pgid is not None and pgid == current_pgid:
        return None
    return pgid


def _read_proc_cmdline(pid: int) -> List[str]:
    try:
        data = Path(f"/proc/{pid}/cmdline").read_bytes()
    except Exception:
        return []
    tokens = [t for t in data.split(b"\x00") if t]
    return [t.decode("utf-8", errors="replace") for t in tokens]


def _read_proc_environ(pid: int) -> Dict[str, str]:
    try:
        data = Path(f"/proc/{pid}/environ").read_bytes()
    except Exception:
        return {}
    env: Dict[str, str] = {}
    for chunk in data.split(b"\x00"):
        if not chunk or b"=" not in chunk:
            continue
        key, value = chunk.split(b"=", 1)
        env[key.decode("utf-8", errors="replace")] = value.decode("utf-8", errors="replace")
    return env


def _read_proc_fd_target(pid: int, fd: int) -> Optional[str]:
    try:
        target = os.readlink(f"/proc/{pid}/fd/{fd}")
    except OSError:
        return None
    if target.endswith(" (deleted)"):
        target = target[:-10]
    if not target.startswith("/"):
        return None
    return target


def _path_points_to_repo_llama_server(raw_path: str) -> bool:
    if not raw_path:
        return False
    candidates = [Path(raw_path)]
    path_obj = Path(raw_path)
    if not path_obj.is_absolute():
        candidates.append(REPO_ROOT / path_obj)
    repo_root = REPO_ROOT.resolve()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        if resolved.name != LLAMA_SERVER.name:
            continue
        if LLAMA_SERVER.exists():
            with suppress(Exception):
                if resolved == LLAMA_SERVER.resolve():
                    return True
        with suppress(ValueError):
            resolved.relative_to(repo_root)
            return True
    return False


def _find_llama_server_token_index(cmdline: List[str]) -> Optional[int]:
    for idx, token in enumerate(cmdline):
        if _path_points_to_repo_llama_server(token):
            return idx
        if Path(token).name == LLAMA_SERVER.name:
            return idx
    return None


def _is_recoverable_llama_server(cmdline: List[str], env: Dict[str, str]) -> bool:
    if not cmdline:
        return False
    if env.get(RECOVERY_MARKER_ENV) == "1":
        return True
    return _find_llama_server_token_index(cmdline) is not None


def _parse_recovered_llama_config(cmdline: List[str], env: Dict[str, str]) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    extra: List[str] = []
    start_index = _find_llama_server_token_index(cmdline)
    i = (start_index + 1) if start_index is not None else 1
    while i < len(cmdline):
        tok = cmdline[i]
        next_tok = cmdline[i + 1] if i + 1 < len(cmdline) else None

        if tok == "--model" and next_tok is not None:
            config["model_ref"] = next_tok
            i += 2
            continue
        if tok == "--host" and next_tok is not None:
            config["host"] = next_tok
            i += 2
            continue
        if tok == "--port" and next_tok is not None:
            parsed = _coerce_int(next_tok)
            if parsed is not None:
                config["port"] = parsed
                i += 2
                continue
        if tok == "--n-gpu-layers" and next_tok is not None:
            parsed = _coerce_int(next_tok)
            if parsed is not None:
                config["n_gpu_layers"] = parsed
                i += 2
                continue
        if tok == "--ctx-size" and next_tok is not None:
            parsed = _coerce_int(next_tok)
            if parsed is not None:
                config["ctx_size"] = parsed
                i += 2
                continue
        if tok == "--n-cpu-moe" and next_tok is not None:
            parsed = _coerce_int(next_tok)
            if parsed is not None:
                config["n_cpu_moe"] = parsed
                i += 2
                continue
        if tok == "--mmproj" and next_tok is not None:
            config["mmproj"] = next_tok
            i += 2
            continue
        if tok == "--reasoning-format" and next_tok is not None:
            config["reasoning_format"] = next_tok
            i += 2
            continue
        if tok == "--tensor-split" and next_tok is not None:
            config["tensor_split"] = next_tok
            i += 2
            continue
        if tok == "--split-mode" and next_tok is not None:
            config["split_mode"] = next_tok
            i += 2
            continue
        if tok == "--embeddings":
            config["mode"] = "embed"
            i += 1
            continue
        if tok == "--cpu-moe":
            config["cpu_moe"] = True
            i += 1
            continue
        if tok == "--jinja":
            config["jinja"] = True
            i += 1
            continue
        if tok == "--no-context-shift":
            config["no_context_shift"] = True
            i += 1
            continue

        extra.append(tok)
        i += 1

    cuda_visible_devices = env.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None:
        config["gpu_devices"] = cuda_visible_devices if cuda_visible_devices else "cpu"
        if not cuda_visible_devices:
            config["gpu_strategy"] = "cpu"
    elif config.get("n_gpu_layers") == 0:
        config["gpu_devices"] = "cpu"
        config["gpu_strategy"] = "cpu"

    config["extra_flags"] = shlex.join(extra)
    return config


def _merge_recovered_config(existing: Dict[str, Any], discovered: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(existing or {})
    for key, value in discovered.items():
        if value is None:
            continue
        if isinstance(value, str) and value == "" and key != "extra_flags":
            continue
        if key == "extra_flags":
            if value or not merged.get("extra_flags"):
                merged[key] = value
            continue
        merged[key] = value

    merged.setdefault("mode", "llm")
    merged.setdefault("model_ref", "")
    merged.setdefault("host", "127.0.0.1")
    merged.setdefault("port", 45540)
    merged.setdefault("cpu_moe", False)
    merged.setdefault("jinja", False)
    merged.setdefault("no_context_shift", False)
    merged.setdefault("extra_flags", "")
    return merged


def _recover_log_file(pid: int, env: Dict[str, str]) -> Optional[str]:
    hinted = (env.get(RECOVERY_LOG_FILE_ENV) or "").strip()
    if hinted:
        return hinted
    for fd in (1, 2):
        target = _read_proc_fd_target(pid, fd)
        if target:
            path = Path(target)
            with suppress(OSError):
                if path.exists() and path.is_file():
                    return target
            continue
    return None


def _derive_recovered_name(config: Dict[str, Any], env: Dict[str, str], port: Optional[int]) -> str:
    hinted = (env.get(RECOVERY_INSTANCE_NAME_ENV) or "").strip()
    if hinted:
        return hinted
    model_ref = str(config.get("model_ref") or "").strip()
    mode = str(config.get("mode") or "llm").strip().lower() or "llm"
    stem = Path(model_ref).stem if model_ref else mode
    safe_stem = "-".join(part for part in stem.replace("_", "-").split() if part) or mode
    if port:
        return f"recovered-{safe_stem}-{port}"
    return f"recovered-{safe_stem}"


def _record_matches_process(pid: Optional[int], cmdline: List[str], pgid: Optional[int] = None) -> bool:
    if not pid or not _pid_alive(pid):
        return False
    if not _cmdline_matches(pid, cmdline):
        return False
    if pgid is not None:
        actual = _pgid_for_pid(pid)
        if actual is None or actual != pgid:
            return False
    return True


def _signal_record_process(pid: Optional[int], pgid: Optional[int], sig: int) -> bool:
    target_pgid = pgid or _safe_pgid_for_control(pid)
    if target_pgid is not None and target_pgid > 0:
        try:
            os.killpg(target_pgid, sig)
            return True
        except ProcessLookupError:
            return False
        except OSError:
            pass
    if pid is not None and pid > 0:
        try:
            os.kill(pid, sig)
            return True
        except ProcessLookupError:
            return False
        except OSError:
            return False
    return False


async def _spawn_logged_process(
    cmd: List[str],
    *,
    log_file: Path,
    cwd: Path,
    env: Dict[str, str],
) -> asyncio.subprocess.Process:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8", errors="replace") as sink:
        return await asyncio.create_subprocess_exec(
            *cmd,
            stdout=sink,
            stderr=sink,
            env=env,
            cwd=str(cwd),
            start_new_session=True,
        )


# ---------------------------------------------------------------------------
# Managed items
# ---------------------------------------------------------------------------


class ManagedInstance:
    def __init__(self, record: InstanceRecord, log_buffer: LogBuffer) -> None:
        self.record = record
        self.log_buffer = log_buffer
        self.process: Optional[asyncio.subprocess.Process] = None
        self._watch_task: Optional[asyncio.Task] = None
        self._tail_task: Optional[asyncio.Task] = None
        self._done = asyncio.Event()
        self._done.set()
        self._finalized = False

    @property
    def uptime_s(self) -> Optional[float]:
        if self.record.status not in INSTANCE_ACTIVE_STATUSES or self.record.started_at is None:
            return None
        return max(0.0, time.time() - self.record.started_at)

    def is_alive(self) -> bool:
        return _record_matches_process(self.record.pid, self.record.cmdline, self.record.pgid)

    def is_streamable(self) -> bool:
        return self.record.status in INSTANCE_ACTIVE_STATUSES and self.is_alive()


class ManagedBuild:
    def __init__(self, record: BuildRecord, log_buffer: LogBuffer) -> None:
        self.record = record
        self.log_buffer = log_buffer
        self.process: Optional[asyncio.subprocess.Process] = None
        self._watch_task: Optional[asyncio.Task] = None
        self._tail_task: Optional[asyncio.Task] = None
        self._done = asyncio.Event()
        self._done.set()
        self._finalized = False

    def is_alive(self) -> bool:
        return _record_matches_process(self.record.pid, self.record.cmdline, self.record.pgid)

    def is_streamable(self) -> bool:
        return self.record.status in BUILD_ACTIVE_STATUSES and self.is_alive()


# ---------------------------------------------------------------------------
# ProcessManager
# ---------------------------------------------------------------------------


class ProcessManager:
    def __init__(self, cfg: WebConfig, store: StateStore) -> None:
        self.cfg = cfg
        self.store = store
        self._instances: Dict[str, ManagedInstance] = {}
        self._builds: Dict[str, ManagedBuild] = {}
        self._lock = asyncio.Lock()
        self.instance_events = EventBroadcaster()
        self.build_events = EventBroadcaster()
        ensure_log_dir()

    # ---- event helpers ----

    def instance_snapshot(self) -> Dict[str, Any]:
        instances = self.list_instances()
        running = sum(1 for inst in instances if inst.get("status") == "running" and inst.get("alive"))
        return {
            "type": "instances.snapshot",
            "total": len(instances),
            "running": running,
            "instances": instances,
        }

    def build_snapshot(self) -> Dict[str, Any]:
        builds = self.list_builds()
        active = sum(1 for build in builds if build.get("status") == "running" and build.get("alive"))
        return {
            "type": "builds.snapshot",
            "total": len(builds),
            "running": active,
            "builds": builds,
        }

    def _notify_instances(self) -> None:
        with suppress(Exception):
            self.instance_events.publish(self.instance_snapshot())

    def _notify_builds(self) -> None:
        with suppress(Exception):
            self.build_events.publish(self.build_snapshot())

    # ---- lifecycle ----

    async def startup(self) -> None:
        for rec in self.store.list_instances():
            inst = self._build_managed_instance(rec)
            self._instances[rec.id] = inst
            if rec.status not in INSTANCE_ACTIVE_STATUSES:
                continue
            if rec.cmdline and _record_matches_process(rec.pid, rec.cmdline, rec.pgid):
                if rec.pgid is None:
                    rec.pgid = _safe_pgid_for_control(rec.pid)
                    self.store.upsert_instance(rec)
                self._attach_active_instance(inst, process=None, start_offset=_file_size(inst.log_buffer.log_file))
                if rec.status == "stopping":
                    asyncio.create_task(self.stop_instance(rec.id))
            else:
                self._mark_instance_inactive(inst, "crashed" if rec.status == "running" else "stopped")

        for rec in self.store.list_builds():
            if rec.status not in BUILD_ACTIVE_STATUSES:
                continue
            if rec.cmdline and _record_matches_process(rec.pid, rec.cmdline, rec.pgid):
                if rec.pgid is None:
                    rec.pgid = _safe_pgid_for_control(rec.pid)
                    self.store.upsert_build(rec)
                build = self._build_managed_build(rec)
                self._builds[rec.id] = build
                self._attach_active_build(build, process=None, start_offset=_file_size(build.log_buffer.log_file))
                if rec.status == "cancelling":
                    asyncio.create_task(self.stop_build(rec.id))
            else:
                self._mark_build_inactive(rec, "failure" if rec.status == "running" else "cancelled")

        self._recover_orphan_instances()
        self._notify_instances()
        self._notify_builds()

    async def shutdown(self) -> None:
        # Keep managed processes running across backend restarts; only stop local
        # tail/watch tasks and release subscribers.
        for build in list(self._builds.values()):
            await self._detach_build(build)
        for inst in list(self._instances.values()):
            await self._detach_instance(inst)

    # ---- instances ----

    def _build_managed_instance(self, record: InstanceRecord) -> ManagedInstance:
        buffer = LogBuffer(capacity=LOG_BUFFER_CAPACITY, log_file=_log_path(record.log_file))
        buffer.seed_from_file()
        return ManagedInstance(record=record, log_buffer=buffer)

    def list_instances(self) -> List[Dict[str, Any]]:
        return [self._serialize_instance(inst) for inst in self._instances.values()]

    def get_instance(self, instance_id: str) -> Optional[ManagedInstance]:
        return self._instances.get(instance_id)

    def serialize_instance(self, instance_id: str) -> Optional[Dict[str, Any]]:
        inst = self._instances.get(instance_id)
        if inst is None:
            return None
        return self._serialize_instance(inst)

    def _serialize_instance(self, inst: ManagedInstance) -> Dict[str, Any]:
        data = asdict(inst.record)
        data["uptime_s"] = inst.uptime_s
        data["alive"] = inst.is_alive()
        return data

    def get_instance_logs(self, instance_id: str, tail: int = LOG_TAIL_LINES_DEFAULT) -> List[str]:
        inst = self._instances.get(instance_id)
        if inst is None:
            return []
        if inst.is_streamable():
            return inst.log_buffer.snapshot(tail=tail)
        return read_log_tail(inst.log_buffer.log_file, tail=tail)

    def get_live_instance_buffer(self, instance_id: str) -> Optional[LogBuffer]:
        inst = self._instances.get(instance_id)
        if inst is None or not inst.is_streamable():
            return None
        return inst.log_buffer

    async def create_instance(self, name: str, config: Dict[str, Any], auto_start: bool) -> ManagedInstance:
        instance_id = uuid.uuid4().hex[:12]
        log_file = ensure_log_dir() / f"{instance_id}.log"
        record = InstanceRecord(
            id=instance_id,
            name=name or f"instance-{instance_id[:6]}",
            kind="llama-server",
            config=dict(config),
            host=str(config.get("host") or "127.0.0.1"),
            port=_coerce_int(config.get("port")),
            log_file=str(log_file),
        )
        inst = self._build_managed_instance(record)
        self._instances[instance_id] = inst
        self.store.upsert_instance(record)
        self._notify_instances()
        if auto_start:
            await self.start_instance(instance_id)
        return inst

    async def start_instance(self, instance_id: str) -> ManagedInstance:
        inst = self._instances.get(instance_id)
        if inst is None:
            raise KeyError(instance_id)
        async with self._lock:
            if inst.is_streamable():
                return inst
            config = inst.record.config
            model_ref = str(config.get("model_ref") or config.get("model") or "").strip()
            if not model_ref:
                raise ValueError("config.model_ref is required")
            models_dir = Path(str(config.get("models_dir") or self.cfg.models_dir)).expanduser()
            hf_token = (str(config.get("hf_token") or "")).strip() or None

            loop = asyncio.get_running_loop()
            try:
                model_path: Path = await loop.run_in_executor(
                    None, loadmodel.resolve_gguf, model_ref, models_dir, hf_token
                )
            except SystemExit as exc:
                raise RuntimeError(f"Failed to resolve model: {exc}") from exc

            wants_moe = _coerce_bool(config.get("cpu_moe")) or (
                (_coerce_int(config.get("n_cpu_moe")) or 0) > 0
            )
            try:
                loadmodel.ensure_moe_flags_available(wants_moe)
            except SystemExit as exc:
                raise RuntimeError(str(exc)) from exc

            cmd = build_llama_server_cmd(config, model_path)
            inst.record.cmdline = cmd
            inst.record.status = "running"
            inst.record.last_exit = None
            inst.record.stopped_at = None
            inst.record.started_at = time.time()
            inst.log_buffer.emit(f"[backend] $ {' '.join(cmd)}")

            env = os.environ.copy()
            cuda_visible_devices = build_cuda_visible_devices(config)
            if cuda_visible_devices is not None:
                env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
                inst.log_buffer.emit(f"[backend] CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
            env[RECOVERY_MARKER_ENV] = "1"
            env[RECOVERY_INSTANCE_ID_ENV] = inst.record.id
            env[RECOVERY_INSTANCE_NAME_ENV] = inst.record.name
            if inst.record.log_file:
                env[RECOVERY_LOG_FILE_ENV] = inst.record.log_file
            start_offset = _file_size(inst.log_buffer.log_file)
            try:
                proc = await _spawn_logged_process(
                    cmd,
                    log_file=_log_path(inst.record.log_file) or ensure_log_dir() / f"{instance_id}.log",
                    cwd=REPO_ROOT,
                    env=env,
                )
            except Exception:
                inst.record.status = "crashed"
                inst.record.started_at = None
                inst.record.pid = None
                inst.record.pgid = None
                self.store.upsert_instance(inst.record)
                self._notify_instances()
                raise

            inst.record.pid = proc.pid
            inst.record.pgid = _pgid_for_pid(proc.pid) or proc.pid
            self.store.upsert_instance(inst.record)
            self._attach_active_instance(inst, process=proc, start_offset=start_offset)
            self._notify_instances()
            return inst

    async def stop_instance(self, instance_id: str) -> None:
        inst = self._instances.get(instance_id)
        if inst is None:
            raise KeyError(instance_id)
        if inst.record.status == "stopping" and inst.is_alive():
            await self._await_done(inst._done, timeout=12.0)
            return
        if not inst.is_alive() and not inst._done.is_set():
            await self._await_done(inst._done, timeout=2.0)
            return
        if not inst.is_alive():
            self._mark_instance_inactive(inst, "stopped")
            self._notify_instances()
            return

        inst.record.status = "stopping"
        self.store.upsert_instance(inst.record)
        self._notify_instances()

        _signal_record_process(inst.record.pid, inst.record.pgid, signal.SIGTERM)
        if not await self._await_done(inst._done, timeout=10.0):
            _signal_record_process(inst.record.pid, inst.record.pgid, signal.SIGKILL)
            await self._await_done(inst._done, timeout=2.0)

    async def restart_instance(self, instance_id: str) -> ManagedInstance:
        await self.stop_instance(instance_id)
        return await self.start_instance(instance_id)

    async def recover_instances(self) -> List[ManagedInstance]:
        async with self._lock:
            recovered = self._recover_orphan_instances()
        if recovered:
            self._notify_instances()
        return recovered

    async def delete_instance(self, instance_id: str) -> bool:
        inst = self._instances.get(instance_id)
        if inst is None:
            return False
        if inst.is_alive():
            await self.stop_instance(instance_id)
        await self._detach_instance(inst)
        inst.log_buffer.close()
        self._instances.pop(instance_id, None)
        ok = self.store.delete_instance(instance_id)
        self._notify_instances()
        return ok

    # ---- builds ----

    def _build_managed_build(self, record: BuildRecord) -> ManagedBuild:
        buffer = LogBuffer(capacity=LOG_BUFFER_CAPACITY, log_file=_log_path(record.log_file))
        buffer.seed_from_file()
        return ManagedBuild(record=record, log_buffer=buffer)

    def list_builds(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for rec in self.store.list_builds():
            data = asdict(rec)
            data["alive"] = _record_matches_process(rec.pid, rec.cmdline, rec.pgid) if rec.cmdline else False
            out.append(data)
        return out

    def get_build(self, build_id: str) -> Optional[Dict[str, Any]]:
        rec = self.store.get_build(build_id)
        if rec is None:
            return None
        data = asdict(rec)
        data["alive"] = _record_matches_process(rec.pid, rec.cmdline, rec.pgid) if rec.cmdline else False
        return data

    def get_build_logs(self, build_id: str, tail: int = LOG_TAIL_LINES_DEFAULT) -> List[str]:
        build = self._builds.get(build_id)
        if build is not None and build.is_streamable():
            return build.log_buffer.snapshot(tail=tail)
        rec = self.store.get_build(build_id)
        if rec is None:
            return []
        return read_log_tail(_log_path(rec.log_file), tail=tail)

    def get_live_build_buffer(self, build_id: str) -> Optional[LogBuffer]:
        build = self._builds.get(build_id)
        if build is None or not build.is_streamable():
            return None
        return build.log_buffer

    async def start_build(self, config: Dict[str, Any]) -> BuildRecord:
        build_id = uuid.uuid4().hex[:12]
        log_file = ensure_log_dir() / f"build-{build_id}.log"
        record = BuildRecord(
            id=build_id,
            config=dict(config),
            started_at=time.time(),
            status="running",
            log_file=str(log_file),
        )
        build = self._build_managed_build(record)
        cmd = _build_autodevops_cmd(config)
        build.record.cmdline = cmd
        build.log_buffer.emit(f"[backend] $ {' '.join(cmd)}")
        start_offset = _file_size(build.log_buffer.log_file)

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        try:
            proc = await _spawn_logged_process(cmd, log_file=log_file, cwd=REPO_ROOT, env=env)
        except Exception as exc:
            build.log_buffer.emit(f"[backend] failed to start build: {exc}")
            record.status = "failure"
            record.finished_at = time.time()
            record.exit_code = None
            self.store.upsert_build(record)
            self._notify_builds()
            return record

        record.pid = proc.pid
        record.pgid = _pgid_for_pid(proc.pid) or proc.pid
        self.store.upsert_build(record)
        self._builds[build_id] = build
        self._attach_active_build(build, process=proc, start_offset=start_offset)
        self._notify_builds()
        return record

    async def stop_build(self, build_id: str) -> None:
        build = self._builds.get(build_id)
        if build is None:
            return
        if build.record.status == "cancelling" and build.is_alive():
            await self._await_done(build._done, timeout=7.0)
            return
        if not build.is_alive() and not build._done.is_set():
            await self._await_done(build._done, timeout=2.0)
            return
        if not build.is_alive():
            await self._finalize_build(build, exit_code=build.record.exit_code)
            return

        build.record.status = "cancelling"
        self.store.upsert_build(build.record)
        self._notify_builds()

        _signal_record_process(build.record.pid, build.record.pgid, signal.SIGTERM)
        if not await self._await_done(build._done, timeout=5.0):
            _signal_record_process(build.record.pid, build.record.pgid, signal.SIGKILL)
            await self._await_done(build._done, timeout=2.0)

    # ---- tracking ----

    def _attach_active_instance(
        self,
        inst: ManagedInstance,
        *,
        process: Optional[asyncio.subprocess.Process],
        start_offset: int,
    ) -> None:
        inst.process = process
        inst._finalized = False
        inst._done.clear()
        inst._tail_task = asyncio.create_task(self._follow_log_file(inst.log_buffer, start_offset))
        if process is not None:
            inst._watch_task = asyncio.create_task(self._wait_instance_exit(inst))
        else:
            inst._watch_task = asyncio.create_task(self._poll_instance_exit(inst))

    def _attach_active_build(
        self,
        build: ManagedBuild,
        *,
        process: Optional[asyncio.subprocess.Process],
        start_offset: int,
    ) -> None:
        build.process = process
        build._finalized = False
        build._done.clear()
        build._tail_task = asyncio.create_task(self._follow_log_file(build.log_buffer, start_offset))
        if process is not None:
            build._watch_task = asyncio.create_task(self._wait_build_exit(build))
        else:
            build._watch_task = asyncio.create_task(self._poll_build_exit(build))

    def _recover_orphan_instances(self) -> List[ManagedInstance]:
        recovered: List[ManagedInstance] = []
        live_pids = {
            inst.record.pid
            for inst in self._instances.values()
            if inst.record.pid is not None and inst.is_alive()
        }
        for proc_dir in Path("/proc").iterdir():
            if not proc_dir.name.isdigit():
                continue
            pid = int(proc_dir.name)
            if pid == os.getpid() or pid in live_pids:
                continue
            cmdline = _read_proc_cmdline(pid)
            if not cmdline:
                continue
            env = _read_proc_environ(pid)
            if not _is_recoverable_llama_server(cmdline, env):
                continue
            inst = self._adopt_recovered_instance(pid=pid, cmdline=cmdline, env=env)
            if inst is None:
                continue
            live_pids.add(pid)
            recovered.append(inst)
        return recovered

    def _find_recovery_target(
        self,
        *,
        pid: int,
        cmdline: List[str],
        recovered_id: Optional[str],
        host: Optional[str],
        port: Optional[int],
        log_file: Optional[str],
    ) -> Optional[ManagedInstance]:
        if recovered_id:
            inst = self._instances.get(recovered_id)
            if inst is not None:
                return inst

        for inst in self._instances.values():
            if inst.is_alive():
                continue
            rec = inst.record
            if rec.pid == pid and rec.pid is not None:
                return inst
            if rec.cmdline and rec.cmdline == cmdline:
                return inst
            if log_file and rec.log_file and rec.log_file == log_file:
                return inst
            if port is not None and rec.port == port and rec.host == host:
                return inst
        return None

    def _adopt_recovered_instance(
        self,
        *,
        pid: int,
        cmdline: List[str],
        env: Dict[str, str],
    ) -> Optional[ManagedInstance]:
        if not _pid_alive(pid):
            return None

        discovered = _parse_recovered_llama_config(cmdline, env)
        host = str(discovered.get("host") or "127.0.0.1")
        port = _coerce_int(discovered.get("port")) or 45540
        log_file = _recover_log_file(pid, env)
        recovered_id = (env.get(RECOVERY_INSTANCE_ID_ENV) or "").strip() or None
        target = self._find_recovery_target(
            pid=pid,
            cmdline=cmdline,
            recovered_id=recovered_id,
            host=host,
            port=port,
            log_file=log_file,
        )

        if target is not None:
            previous = target.record
            record = InstanceRecord(
                id=previous.id,
                name=previous.name or _derive_recovered_name(discovered, env, port),
                kind=previous.kind or "llama-server",
                config=_merge_recovered_config(previous.config, discovered),
                pid=pid,
                pgid=_pgid_for_pid(pid) or pid,
                cmdline=list(cmdline),
                started_at=previous.started_at or time.time(),
                stopped_at=None,
                last_exit=None,
                status="running",
                host=host,
                port=port,
                log_file=log_file or previous.log_file,
                restart_policy=previous.restart_policy,
            )
        else:
            instance_id = recovered_id if recovered_id and recovered_id not in self._instances else uuid.uuid4().hex[:12]
            record = InstanceRecord(
                id=instance_id,
                name=_derive_recovered_name(discovered, env, port),
                kind="llama-server",
                config=_merge_recovered_config({}, discovered),
                pid=pid,
                pgid=_pgid_for_pid(pid) or pid,
                cmdline=list(cmdline),
                started_at=time.time(),
                stopped_at=None,
                last_exit=None,
                status="running",
                host=host,
                port=port,
                log_file=log_file,
            )

        inst = self._build_managed_instance(record)
        inst.log_buffer.emit(f"[backend] recovered existing llama-server pid={pid}")
        self._instances[record.id] = inst
        self.store.upsert_instance(record)
        self._attach_active_instance(inst, process=None, start_offset=_file_size(inst.log_buffer.log_file))
        return inst

    async def _follow_log_file(self, buffer: LogBuffer, start_offset: int) -> None:
        log_file = buffer.log_file
        if log_file is None:
            return
        offset = start_offset
        pending = ""
        while True:
            try:
                if log_file.exists():
                    size = log_file.stat().st_size
                    if size < offset:
                        offset = 0
                        pending = ""
                    if size > offset:
                        with log_file.open("r", encoding="utf-8", errors="replace") as fh:
                            fh.seek(offset)
                            chunk = fh.read()
                            offset = fh.tell()
                        if chunk:
                            data = pending + chunk
                            pending = ""
                            parts = data.splitlines(keepends=True)
                            if parts and not parts[-1].endswith("\n"):
                                pending = parts.pop()
                            for part in parts:
                                buffer.push(part)
                await asyncio.sleep(FOLLOW_POLL_INTERVAL)
            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(FOLLOW_POLL_INTERVAL)

    async def _wait_instance_exit(self, inst: ManagedInstance) -> None:
        assert inst.process is not None
        try:
            rc = await inst.process.wait()
        except asyncio.CancelledError:
            return
        await self._finalize_instance(inst, rc)

    async def _poll_instance_exit(self, inst: ManagedInstance) -> None:
        try:
            while _record_matches_process(inst.record.pid, inst.record.cmdline, inst.record.pgid):
                await asyncio.sleep(FOLLOW_POLL_INTERVAL)
        except asyncio.CancelledError:
            return
        await self._finalize_instance(inst, None)

    async def _wait_build_exit(self, build: ManagedBuild) -> None:
        assert build.process is not None
        try:
            rc = await build.process.wait()
        except asyncio.CancelledError:
            return
        await self._finalize_build(build, rc)

    async def _poll_build_exit(self, build: ManagedBuild) -> None:
        try:
            while _record_matches_process(build.record.pid, build.record.cmdline, build.record.pgid):
                await asyncio.sleep(FOLLOW_POLL_INTERVAL)
        except asyncio.CancelledError:
            return
        await self._finalize_build(build, None)

    async def _finalize_instance(self, inst: ManagedInstance, exit_code: Optional[int]) -> None:
        if inst._finalized:
            return
        inst._finalized = True
        await self._cancel_task(inst._tail_task)
        inst.log_buffer.seed_from_file()

        status_before = inst.record.status
        inst.process = None
        inst.record.last_exit = exit_code
        inst.record.stopped_at = time.time()
        inst.record.pid = None
        inst.record.pgid = None
        if status_before == "stopping":
            inst.record.status = "stopped"
        elif exit_code == 0:
            inst.record.status = "stopped"
        else:
            inst.record.status = "crashed"
        inst.log_buffer.emit(f"[backend] process exited with code {exit_code!r}")
        self.store.upsert_instance(inst.record)
        inst._done.set()
        self._notify_instances()

    async def _finalize_build(self, build: ManagedBuild, exit_code: Optional[int]) -> None:
        if build._finalized:
            return
        build._finalized = True
        await self._cancel_task(build._tail_task)
        build.log_buffer.seed_from_file()

        status_before = build.record.status
        build.process = None
        build.record.exit_code = exit_code
        build.record.finished_at = time.time()
        build.record.pid = None
        build.record.pgid = None
        if status_before == "cancelling":
            build.record.status = "cancelled"
        elif exit_code == 0:
            build.record.status = "success"
        else:
            build.record.status = "failure"
        build.log_buffer.emit(f"[backend] build exited with code {exit_code!r}")
        self.store.upsert_build(build.record)
        self._builds.pop(build.record.id, None)
        build._done.set()
        self._notify_builds()

    async def _detach_instance(self, inst: ManagedInstance) -> None:
        await self._cancel_task(inst._watch_task)
        await self._cancel_task(inst._tail_task)
        inst.log_buffer.close()

    async def _detach_build(self, build: ManagedBuild) -> None:
        await self._cancel_task(build._watch_task)
        await self._cancel_task(build._tail_task)
        build.log_buffer.close()

    async def _cancel_task(self, task: Optional[asyncio.Task]) -> None:
        if task is None or task.done() or task is asyncio.current_task():
            return
        task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await task

    async def _await_done(self, done: asyncio.Event, timeout: float) -> bool:
        try:
            await asyncio.wait_for(done.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def _mark_instance_inactive(self, inst: ManagedInstance, status: str) -> None:
        inst._finalized = True
        inst._done.set()
        inst.process = None
        inst.record.status = status
        inst.record.stopped_at = time.time()
        inst.record.pid = None
        inst.record.pgid = None
        self.store.upsert_instance(inst.record)

    def _mark_build_inactive(self, rec: BuildRecord, status: str) -> None:
        rec.status = status
        rec.finished_at = time.time()
        rec.pid = None
        rec.pgid = None
        self.store.upsert_build(rec)


# ---------------------------------------------------------------------------
# Build command construction
# ---------------------------------------------------------------------------


def _build_autodevops_cmd(config: Dict[str, Any]) -> List[str]:
    cmd = [sys.executable, str(AUTODEVOPS_SCRIPT)]
    if _coerce_bool(config.get("now", True)):
        cmd.append("--now")
    ref = str(config.get("ref") or "").strip()
    if ref:
        cmd += ["--ref", ref]
    if _coerce_bool(config.get("fast_math")):
        cmd.append("--fast-math")
    force_mmq = str(config.get("force_mmq") or "").strip()
    if force_mmq:
        cmd += ["--force-mmq", force_mmq]
    blas = str(config.get("blas") or "").strip()
    if blas:
        cmd += ["--blas", blas]
    if _coerce_bool(config.get("distributed")):
        cmd.append("--distributed")
    if _coerce_bool(config.get("cpu_only")):
        cmd.append("--cpu-only")
    return cmd
