#!/usr/bin/env python3
"""Simple launcher for vLLM scoring server."""

import argparse
import os
import shlex


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch vLLM server for reranking")
    parser.add_argument("model", help="model repo or local path")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--dtype", default="auto")
    args = parser.parse_args()

    cmd = [
        "vllm",
        "serve",
        args.model,
        "--task",
        "score",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--dtype",
        args.dtype,
    ]

    print("Starting vLLM server:\n", shlex.join(cmd))
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
