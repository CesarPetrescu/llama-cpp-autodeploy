"""Persistent instance/build registry backed by ``.web_state.json``."""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class InstanceRecord:
    id: str
    name: str
    kind: str = "llama-server"  # future: "rpc-server" etc.
    config: Dict[str, Any] = field(default_factory=dict)
    pid: Optional[int] = None
    pgid: Optional[int] = None
    cmdline: List[str] = field(default_factory=list)
    started_at: Optional[float] = None
    stopped_at: Optional[float] = None
    last_exit: Optional[int] = None
    status: str = "stopped"  # running | stopping | stopped | crashed | starting
    host: Optional[str] = None
    port: Optional[int] = None
    log_file: Optional[str] = None
    restart_policy: str = "never"  # never | on-failure


@dataclass
class BuildRecord:
    id: str
    config: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    exit_code: Optional[int] = None
    status: str = "pending"  # pending | running | cancelling | cancelled | success | failure
    log_file: Optional[str] = None
    pid: Optional[int] = None
    pgid: Optional[int] = None
    cmdline: List[str] = field(default_factory=list)


@dataclass
class PersistentState:
    instances: List[InstanceRecord] = field(default_factory=list)
    builds: List[BuildRecord] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "instances": [asdict(r) for r in self.instances],
            "builds": [asdict(r) for r in self.builds],
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "PersistentState":
        instances = [InstanceRecord(**r) for r in data.get("instances", [])]
        builds = [BuildRecord(**r) for r in data.get("builds", [])]
        return cls(instances=instances, builds=builds)


class StateStore:
    """Thread-safe wrapper around the JSON-backed persistent state file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._state = self._load()

    def _load(self) -> PersistentState:
        if not self._path.exists():
            return PersistentState()
        try:
            return PersistentState.from_json(json.loads(self._path.read_text()))
        except Exception:
            return PersistentState()

    def _save_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._state.to_json(), indent=2) + "\n")
        tmp.replace(self._path)

    def save(self) -> None:
        with self._lock:
            self._save_locked()

    # ---- instances ----

    def list_instances(self) -> List[InstanceRecord]:
        with self._lock:
            return list(self._state.instances)

    def get_instance(self, instance_id: str) -> Optional[InstanceRecord]:
        with self._lock:
            for rec in self._state.instances:
                if rec.id == instance_id:
                    return rec
        return None

    def upsert_instance(self, record: InstanceRecord) -> None:
        with self._lock:
            for idx, rec in enumerate(self._state.instances):
                if rec.id == record.id:
                    self._state.instances[idx] = record
                    break
            else:
                self._state.instances.append(record)
            self._save_locked()

    def delete_instance(self, instance_id: str) -> bool:
        with self._lock:
            before = len(self._state.instances)
            self._state.instances = [r for r in self._state.instances if r.id != instance_id]
            if len(self._state.instances) != before:
                self._save_locked()
                return True
        return False

    # ---- builds ----

    def list_builds(self) -> List[BuildRecord]:
        with self._lock:
            return list(self._state.builds)

    def get_build(self, build_id: str) -> Optional[BuildRecord]:
        with self._lock:
            for rec in self._state.builds:
                if rec.id == build_id:
                    return rec
        return None

    def upsert_build(self, record: BuildRecord) -> None:
        with self._lock:
            for idx, rec in enumerate(self._state.builds):
                if rec.id == record.id:
                    self._state.builds[idx] = record
                    break
            else:
                self._state.builds.append(record)
            self._save_locked()
