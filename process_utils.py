from __future__ import annotations

import atexit
import subprocess
from contextlib import contextmanager
from typing import Iterable, List


_active_processes: List[subprocess.Popen] = []


def register_process(proc: subprocess.Popen) -> subprocess.Popen:
    if proc not in _active_processes:
        _active_processes.append(proc)
    return proc


def unregister_process(proc: subprocess.Popen) -> None:
    try:
        _active_processes.remove(proc)
    except ValueError:
        pass


def terminate_process(proc: subprocess.Popen, *, timeout: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=timeout)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def cleanup_processes() -> None:
    for proc in list(_active_processes):
        terminate_process(proc)
    _active_processes.clear()


def cleanup_specific(procs: Iterable[subprocess.Popen]) -> None:
    for proc in procs:
        terminate_process(proc)


@contextmanager
def managed_process(cmd: List[str], **kwargs):
    proc = subprocess.Popen(cmd, **kwargs)
    register_process(proc)
    try:
        yield proc
    finally:
        unregister_process(proc)


atexit.register(cleanup_processes)
