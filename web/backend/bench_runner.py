"""Small wrapper around llama-bench that preserves JSON results and live logs."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _forward(stream, sink, capture: list[str] | None = None) -> None:
    while True:
        chunk = stream.readline()
        if not chunk:
            break
        text = chunk.decode("utf-8", errors="replace")
        if capture is not None:
            capture.append(text)
        sink.write(text)
        sink.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run llama-bench and persist stdout JSON.")
    parser.add_argument("--result-file", required=True)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("[bench-runner] missing benchmark command", file=sys.stderr, flush=True)
        return 2

    result_path = Path(args.result_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    stdout_chunks: list[str] = []
    stderr_thread = None
    stdout_thread = None
    try:
        import threading

        stdout_thread = threading.Thread(
            target=_forward,
            args=(proc.stdout, sys.stdout, stdout_chunks),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_forward,
            args=(proc.stderr, sys.stdout, None),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        rc = proc.wait()
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
    finally:
        if proc.stdout:
            proc.stdout.close()
        if proc.stderr:
            proc.stderr.close()

    result_text = "".join(stdout_chunks).strip()
    try:
        result_path.write_text(result_text, encoding="utf-8")
    except OSError as exc:
        print(f"[bench-runner] failed to write result file: {exc}", file=sys.stderr, flush=True)
        if rc == 0:
            rc = 1

    if result_text:
        print(
            f"[bench-runner] saved {len(result_text)} bytes of benchmark JSON to {result_path}",
            flush=True,
        )
    else:
        print("[bench-runner] benchmark produced no stdout JSON", flush=True)

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
