#!/usr/bin/env python3
"""Model launcher for llama-server (Python version).

This script replaces `loadmodel.bash` and supports the same arguments.
Models can be local files or references to Hugging Face repositories in the
form ``user/repo:TAG``. The model will be downloaded using ``huggingface-cli``
if not already present in ``./models``.
"""

import argparse
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv

# Defaults
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_CONTEXT_SIZE = 2048
DEFAULT_THREADS = os.cpu_count() or 1
DEFAULT_GPU_LAYERS = 999

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"

load_dotenv(SCRIPT_DIR / ".env")
load_dotenv(SCRIPT_DIR / ".env.local")
HF_TOKEN = os.getenv("HF_TOKEN", "")


def find_llama_server() -> Path:
    candidates = [
        SCRIPT_DIR / "bin" / "llama-server",
        SCRIPT_DIR / "llama-cpp-latest" / "build" / "bin" / "llama-server",
        Path.home() / "llama-current" / "build" / "bin" / "llama-server",
        Path("/usr/local/bin/llama-server"),
        shutil.which("llama-server")
    ]
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    raise FileNotFoundError("llama-server binary not found. Run autodevops.py first")


def resolve_model(spec: str, local: bool) -> Path:
    if local or Path(spec).is_file():
        return Path(spec)

    spec = spec.replace("hf.co/", "")
    if ":" not in spec:
        raise ValueError("Model spec must be repo:tag when downloading")
    repo, tag = spec.split(":", 1)
    repo_base = repo.split("/")[-1]
    candidates = [f"{repo_base}-{tag.upper()}.gguf", f"{repo_base}.{tag.upper()}.gguf", f"{tag.upper()}.gguf"]
    if tag.endswith(".gguf"):
        candidates = [tag]

    MODELS_DIR.mkdir(exist_ok=True)
    for file in candidates:
        dest = MODELS_DIR / file
        if dest.exists():
            return dest
        cmd = ["huggingface-cli", "download", repo, file, "--local-dir", str(MODELS_DIR), "--local-dir-use-symlinks", "False"]
        if HF_TOKEN:
            cmd += ["--token", HF_TOKEN]
        result = subprocess.run(cmd)
        if result.returncode == 0 and dest.exists():
            return dest
        if dest.exists():
            dest.unlink()
    raise RuntimeError("Could not download model from Hugging Face")


def build_args(model_path: Path, mode: str, host: str, port: int, ctx: int, threads: int, gpu_layers: int, pooling: str, verbose: bool):
    args = ["--model", str(model_path), "--host", host, "--port", str(port), "--ctx-size", str(ctx), "--threads", str(threads)]
    if shutil.which("nvidia-smi"):
        args += ["--n-gpu-layers", str(gpu_layers)]
    if mode == "embedding":
        args.append("--embeddings")
        if pooling:
            args += ["--pooling", pooling]
    elif mode == "rerank":
        args.append("--reranking")
        # For reranker models, use 'rank' pooling by default if not specified
        pooling_type = pooling if pooling else "rank"
        args += ["--pooling", pooling_type]
    if verbose:
        args.append("--verbose")
    return args


def main():
    parser = argparse.ArgumentParser(description="Start llama-server with a model")
    parser.add_argument("model", help="model path or repo:tag")
    parser.add_argument("--embedding", action="store_true", help="embedding mode")
    parser.add_argument("--rerank", action="store_true", help="rerank mode")
    parser.add_argument("--llm", action="store_true", help="LLM mode (default)")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--ctx-size", type=int, default=DEFAULT_CONTEXT_SIZE)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--gpu-layers", type=int, default=DEFAULT_GPU_LAYERS)
    parser.add_argument("--pooling", default="")
    parser.add_argument("--local", action="store_true", help="treat model as local file")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    mode = "llm"
    if args.embedding:
        mode = "embedding"
    elif args.rerank:
        mode = "rerank"

    model_path = resolve_model(args.model, args.local)
    server = find_llama_server()
    cmd = [str(server)] + build_args(model_path, mode, args.host, args.port, args.ctx_size, args.threads, args.gpu_layers, args.pooling, args.verbose)

    print("Starting llama-server:\n", shlex.join(cmd))
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
