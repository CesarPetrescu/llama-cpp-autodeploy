#!/usr/bin/env python3
"""Automated build and deployment for llama.cpp (Python version).

This script mirrors the behaviour of `autodevops.bash` using Python.
It fetches the latest release of llama.cpp, builds it with CUDA support
and manages symlinks inside this repository.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

# Script directories
SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_BIN = SCRIPT_DIR / "bin"
BUILD_DIR = SCRIPT_DIR / "llama-builds"
CURRENT_DIR = SCRIPT_DIR / "llama-current"
LOG_FILE = SCRIPT_DIR / "autodevops.log"
VERSION_FILE = SCRIPT_DIR / ".llama-version"

REPO_URL = "https://github.com/ggml-org/llama.cpp"
API_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"

logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format="[%(asctime)s] %(message)s")

def log(msg: str) -> None:
    print(msg)
    logging.info(msg)


def run(cmd, cwd=None, check=True):
    log(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=check)


def check_dependencies():
    deps = ["git", "cmake", "make", "gcc", "g++", "curl", "jq"]
    missing = [d for d in deps if shutil.which(d) is None]
    if missing:
        log(f"Missing dependencies: {' '.join(missing)}")
        raise SystemExit("Install the required packages and re-run")
    if shutil.which("nvcc") is None:
        raise SystemExit("CUDA compiler (nvcc) not found in PATH")
    if shutil.which("nvidia-smi") is None:
        raise SystemExit("NVIDIA drivers not found")


def get_latest_release() -> str:
    with urlopen(API_URL) as resp:
        data = json.load(resp)
        tag = data.get("tag_name")
        if not tag:
            raise RuntimeError("Could not fetch latest release")
        return tag


def is_new_version(version: str) -> bool:
    if VERSION_FILE.exists():
        current = VERSION_FILE.read_text().strip()
        return current != version
    return True


def get_compute_capability() -> str:
    try:
        out = subprocess.check_output([
            "nvidia-smi", "--query-gpu=compute_cap",
            "--format=csv,noheader,nounits"], text=True)
        cap = out.strip().splitlines()[0]
        return cap.replace('.', '')
    except Exception:
        log("Could not determine GPU compute capability, using default 75")
        return "75"


def build_llama(version: str):
    build_path = BUILD_DIR / f"llama-cpp-{version}"
    if build_path.exists():
        shutil.rmtree(build_path)
    run(["git", "clone", "--depth", "1", "--branch", version, REPO_URL, str(build_path)])
    compute_arch = get_compute_capability()
    (build_path / "build").mkdir(parents=True, exist_ok=True)
    cmake_cmd = [
        "cmake", "..",
        f"-DGGML_CUDA=ON",
        f"-DCMAKE_CUDA_ARCHITECTURES={compute_arch}",
        "-DLLAMA_BLAS=ON",
        "-DLLAMA_BLAS_VENDOR=Intel10_64lp",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGGML_CUDA_FORCE_MMQ=OFF",
        "-DGGML_CUDA_F16=ON",
        "-DLLAMA_CURL=ON",
        "-DLLAMA_SERVER_RERANK=ON",
    ]
    run(cmake_cmd, cwd=build_path / "build")
    run(["make", f"-j{os.cpu_count()}"], cwd=build_path / "build")

    if CURRENT_DIR.is_symlink() or CURRENT_DIR.exists():
        CURRENT_DIR.unlink() if CURRENT_DIR.is_symlink() else shutil.rmtree(CURRENT_DIR)
    CURRENT_DIR.symlink_to(build_path)
    LOCAL_BIN.mkdir(exist_ok=True)
    for f in (build_path / "build" / "bin").glob("*"):
        if f.is_file():
            dest = LOCAL_BIN / f.name
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            dest.symlink_to(f)
    VERSION_FILE.write_text(version)


def test_build() -> bool:
    server = CURRENT_DIR / "build" / "bin" / "llama-server"
    if not server.exists():
        log("llama-server binary not found")
        return False
    try:
        subprocess.run([str(server), "--help"], check=True, stdout=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def schedule_build(version: str):
    temp_script = f"/tmp/llama_build_{version}.sh"
    Path(temp_script).write_text(f"#!/bin/bash\n{sys.executable} {__file__} --now\n")
    os.chmod(temp_script, 0o755)
    if shutil.which("at"):
        run(["bash", "-c", f"echo {temp_script} | at 02:00"])
    else:
        log("'at' command not available; build will run on next invocation")


def main(now: bool):
    check_dependencies()
    version = get_latest_release()
    log(f"Latest version: {version}")
    if now:
        build_llama(version)
        if test_build():
            log("Build completed successfully")
        else:
            log("Build failed")
    else:
        if is_new_version(version):
            current_hour = datetime.now().strftime("%H")
            if current_hour == "02":
                build_llama(version)
                test_build()
                log("Scheduled build completed")
            else:
                schedule_build(version)
                log("New version found; build scheduled for 2 AM")
        else:
            log("Already up to date")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated llama.cpp build script")
    parser.add_argument("--now", action="store_true", help="force immediate build")
    args = parser.parse_args()
    main(args.now)
