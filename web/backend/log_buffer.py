"""Per-instance log ring buffer with durable file helpers."""
from __future__ import annotations

import asyncio
from collections import deque
from pathlib import Path
from typing import AsyncIterator, Deque, List, Optional, Set


def _normalize_line(line: str) -> str:
    return line if line.endswith("\n") else line + "\n"


def read_log_tail(path: Optional[Path], tail: Optional[int] = None) -> List[str]:
    if path is None or not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            if tail is None:
                return [_normalize_line(line) for line in fh]
            if tail <= 0:
                return []
            return [_normalize_line(line) for line in deque(fh, maxlen=tail)]
    except Exception:
        return []


class LogBuffer:
    """Fixed-size line buffer + asyncio broadcaster for file-backed logs."""

    def __init__(self, capacity: int = 2000, log_file: Optional[Path] = None) -> None:
        self._capacity = capacity
        self._lines: Deque[str] = deque(maxlen=capacity)
        self._subscribers: Set[asyncio.Queue[str]] = set()
        self._log_file = log_file

    @property
    def log_file(self) -> Optional[Path]:
        return self._log_file

    def seed_from_file(self, tail: Optional[int] = None) -> None:
        self._lines.clear()
        for line in read_log_tail(self._log_file, self._capacity if tail is None else tail):
            self._lines.append(line)

    def push(self, line: str) -> None:
        normalized = _normalize_line(line)
        self._lines.append(normalized)
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(normalized)
            except asyncio.QueueFull:
                pass

    def emit(self, line: str) -> None:
        normalized = _normalize_line(line)
        if self._log_file is not None:
            try:
                self._log_file.parent.mkdir(parents=True, exist_ok=True)
                with self._log_file.open("a", encoding="utf-8", errors="replace") as fh:
                    fh.write(normalized)
            except Exception:
                pass
        self.push(normalized)

    def snapshot(self, tail: Optional[int] = None) -> List[str]:
        if tail is None or tail >= len(self._lines):
            return list(self._lines)
        return list(self._lines)[-tail:]

    def subscribe(self) -> asyncio.Queue[str]:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1024)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[str]) -> None:
        self._subscribers.discard(queue)

    async def stream(self, include_history: bool = True) -> AsyncIterator[str]:
        queue = self.subscribe()
        try:
            if include_history:
                for line in self.snapshot():
                    yield line
            while True:
                line = await queue.get()
                yield line
        finally:
            self.unsubscribe(queue)

    def close(self) -> None:
        self._subscribers.clear()
