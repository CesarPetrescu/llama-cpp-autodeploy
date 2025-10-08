#!/usr/bin/env python3
import argparse, glob, json, logging, os, re, shutil, subprocess, sys
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

SCRIPT_DIR  = Path(__file__).resolve().parent
LOCAL_BIN   = SCRIPT_DIR / "bin"
BUILD_ROOT  = SCRIPT_DIR / "llama-builds"
CURRENT_DIR = SCRIPT_DIR / "llama-current"
LOG_FILE    = SCRIPT_DIR / "autodevops.log"
VERSION_FILE= SCRIPT_DIR / ".llama-version"

REPO_URL = "https://github.com/ggml-org/llama.cpp"
API_URL  = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="[%(asctime)s] %(message)s")
def log(msg): print(msg); logging.info(msg)

def run(cmd, cwd=None, check=True, env=None):
    log(f"Running: {' '.join(map(str, cmd))}")
    return subprocess.run(cmd, cwd=cwd, check=check, env=env)

def package_hint(missing: list[str]) -> str:
    if shutil.which("apt-get"):
        pkg_map = {
            "gcc": "build-essential",
            "g++": "build-essential",
            "make": "build-essential",
            "cmake": "cmake",
            "git": "git",
            "pkg-config": "pkg-config",
        }
        pkgs = sorted({pkg_map.get(m, m) for m in missing})
        return "Try: sudo apt install -y " + " ".join(pkgs)
    if shutil.which("pacman"):
        pkg_map = {
            "gcc": "base-devel",
            "g++": "base-devel",
            "make": "base-devel",
            "cmake": "cmake",
            "git": "git",
            "pkg-config": "pkgconf",
        }
        pkgs = sorted({pkg_map.get(m, m) for m in missing})
        return "Try: sudo pacman -S --needed " + " ".join(pkgs)
    return ""


def check_dependencies():
    missing = [d for d in ["git","cmake","make","gcc","g++","pkg-config"] if shutil.which(d) is None]
    if missing:
        hint = package_hint(missing)
        msg = "Missing dependency: " + ", ".join(missing)
        if hint:
            msg += f"\n{hint}"
        raise SystemExit(msg)
    if shutil.which("nvidia-smi") is None:
        raise SystemExit("NVIDIA drivers not found (nvidia-smi missing)")

def get_latest_release_tag():
    with urlopen(API_URL) as resp:
        data = json.load(resp)
        tag = data.get("tag_name")
        if not tag:
            raise RuntimeError("Could not fetch latest release")
        return tag

def get_ref(ref):
    if ref.lower() == "latest":
        tag = get_latest_release_tag()
        log(f"Latest release tag: {tag}")
        return tag
    return ref

def is_new_version(version: str) -> bool:
    return not VERSION_FILE.exists() or VERSION_FILE.read_text().strip() != version

def get_compute_capability_str() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi","--query-gpu=compute_cap","--format=csv,noheader,nounits"],
            text=True
        ).strip().splitlines()
        return out[0].replace('.','')  # e.g. "89"
    except Exception:
        log("Could not determine GPU compute capability, using default 75")
        return "75"

def _is_cuda_home(path: Path) -> bool:
    return (path / "bin" / "nvcc").exists()


def _candidate_cuda_directories() -> list[Path]:
    canonical = Path("/usr/local/cuda")
    candidates: list[Path] = [canonical]
    cuda_root = Path("/usr/local")
    if cuda_root.exists():
        candidates.extend(sorted(cuda_root.glob("cuda-*"), reverse=True))
    return candidates


def pick_cuda_home() -> Path | None:
    env_home = os.environ.get("CUDA_HOME")
    if env_home:
        env_path = Path(env_home)
        if _is_cuda_home(env_path):
            return env_path

    for candidate in _candidate_cuda_directories():
        if _is_cuda_home(candidate):
            return candidate

    nvcc = shutil.which("nvcc")
    return Path(nvcc).resolve().parent.parent if nvcc else None

def nvcc_version_tuple(nvcc_bin: Path | str):
    try:
        m = re.search(r"release\s+(\d+)\.(\d+)", subprocess.check_output([str(nvcc_bin),"--version"], text=True))
        return (int(m.group(1)), int(m.group(2))) if m else None
    except Exception:
        return None

def make_env(cuda_home: Path | None) -> dict:
    env = os.environ.copy()
    if cuda_home:
        env["CUDA_HOME"] = str(cuda_home)
        env["PATH"] = str(cuda_home/"bin") + os.pathsep + env.get("PATH","")
        env["LD_LIBRARY_PATH"] = str(cuda_home/"lib64") + os.pathsep + env.get("LD_LIBRARY_PATH","")
    return env

def write_math_fix_header(build_dir: Path) -> Path:
    hdr = build_dir / "cuda_glibc_math_fix.h"
    hdr.write_text(
        "// auto-generated: see autodevops.py\n"
        "#pragma push_macro(\"_GNU_SOURCE\")\n"
        "#undef _GNU_SOURCE\n"
        "#include <math.h>\n"
        "#pragma pop_macro(\"_GNU_SOURCE\")\n"
    )
    return hdr

def _lib_present(patterns: list[str]) -> bool:
    return any(glob.glob(p) for p in patterns)


def mkl_present() -> bool:
    candidates = [
        "/usr/lib/x86_64-linux-gnu/libmkl_rt.so",
        "/usr/lib/x86_64-linux-gnu/libmkl_rt.so.*",
        "/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_rt.so",
        "/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_rt.so.*",
    ]
    return _lib_present(candidates)


def openblas_present() -> bool:
    candidates = [
        "/usr/lib/x86_64-linux-gnu/libopenblas.so",
        "/usr/lib/x86_64-linux-gnu/libopenblas.so.*",
        "/usr/lib/libopenblas.so",
        "/usr/lib/libopenblas.so.*",
    ]
    return _lib_present(candidates)


def blas_hint(vendor: str) -> str:
    if vendor == "mkl":
        return (
            "Install Intel oneAPI MKL runtime first.\n"
            "  • Debian/Ubuntu: sudo apt install -y intel-oneapi-mkl intel-oneapi-mkl-devel intel-oneapi-openmp\n"
            "  • Arch/Manjaro: yay -S intel-oneapi-mkl"
        )
    if vendor == "openblas":
        return (
            "Install OpenBLAS development libraries.\n"
            "  • Debian/Ubuntu: sudo apt install -y libopenblas-dev\n"
            "  • Arch/Manjaro: sudo pacman -S --needed openblas"
        )
    return ""

def clone_llama(version: str, build_path: Path):
    if build_path.exists():
        shutil.rmtree(build_path)
    run(["git","clone","--depth","1","--branch",version,REPO_URL,str(build_path)])

def link_outputs(build_path: Path):
    # Update ./llama-current symlink
    if CURRENT_DIR.is_symlink() or CURRENT_DIR.exists():
        CURRENT_DIR.unlink() if CURRENT_DIR.is_symlink() else shutil.rmtree(CURRENT_DIR)
    CURRENT_DIR.symlink_to(build_path)

    # Symlink binaries into ./bin/
    LOCAL_BIN.mkdir(exist_ok=True)
    for f in (build_path/"build"/"bin").glob("*"):
        if f.is_file():
            dest = LOCAL_BIN / f.name
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            dest.symlink_to(f)

def build_llama(version: str, force_mmq: str, fast_math: bool, blas_mode: str, enable_rpc: bool):
    build_path = BUILD_ROOT / f"llama-cpp-{version}"
    clone_llama(version, build_path)

    (build_path/"build").mkdir(parents=True, exist_ok=True)

    cuda_home = pick_cuda_home()
    if not cuda_home:
        raise SystemExit("CUDA Toolkit not found: set CUDA_HOME or install CUDA")
    nvcc_ver = nvcc_version_tuple(cuda_home/"bin"/"nvcc")

    # GPU arch
    arch = get_compute_capability_str()
    arch_i = int(re.sub(r'[^0-9]','', arch) or "75")
    log(f"Using CUDA at: {cuda_home} (nvcc {nvcc_ver[0]}.{nvcc_ver[1]})" if nvcc_ver else f"Using CUDA at: {cuda_home}")

    # BLAS selection
    if blas_mode == "off":
        ggml_blas = "OFF"; blas_vendor = "Generic"
    elif blas_mode == "mkl":
        if not mkl_present():
            raise SystemExit("Intel oneAPI MKL not detected on this system.\n" + blas_hint("mkl"))
        ggml_blas = "ON";  blas_vendor = "Intel10_64lp"
    elif blas_mode == "openblas":
        if not openblas_present():
            raise SystemExit("OpenBLAS not detected on this system.\n" + blas_hint("openblas"))
        ggml_blas = "ON";  blas_vendor = "OpenBLAS"
    else:
        if mkl_present():
            ggml_blas = "ON";  blas_vendor = "Intel10_64lp"
        elif openblas_present():
            ggml_blas = "ON";  blas_vendor = "OpenBLAS"
        else:
            log("No MKL or OpenBLAS libraries detected; building without BLAS acceleration.")
            ggml_blas = "OFF"; blas_vendor = "Generic"

    # CUDA flags
    cuda_flags = os.environ.get("CMAKE_CUDA_FLAGS","")
    # glibc header workaround for <= 12.9 if needed; CUDA 13.0 doesn't need it
    if (not cuda_flags) and nvcc_ver and (nvcc_ver[0] == 12 and nvcc_ver[1] >= 9):
        fix_hdr = write_math_fix_header(build_path/"build")
        cuda_flags = f"-include {fix_hdr}"

    if fast_math:
        cuda_flags = (cuda_flags + " --use_fast_math").strip()

    # MMQ default heuristic: enable on Ampere/Ada (SM >= 80)
    mmq = "ON" if (force_mmq == "on" or (force_mmq == "auto" and arch_i >= 80)) else "OFF"

    log(f"Host glibc version: " + os.popen("ldd --version 2>/dev/null | head -1").read().strip())
    # Try to detect patched headers quickly (informational only)
    math_h = cuda_home / "targets/x86_64-linux/include/crt/math_functions.h"
    if math_h.exists():
        if b"noexcept" in math_h.read_bytes():
            log(f"{math_h} already has noexcept decorations; no patch needed.")
    log(f"CUDA arch       : {arch}")
    log(f"BLAS            : {ggml_blas} (vendor={blas_vendor})")
    log(f"GGML_CUDA_FORCE_MMQ: {mmq}")
    log(f"GGML_RPC         : {'ON' if enable_rpc else 'OFF'}")
    log(f"CMAKE_CUDA_FLAGS: {cuda_flags or '(none)'}")

    cmake_cmd = [
        "cmake","..",
        "-DGGML_CUDA=ON",
        f"-DCMAKE_CUDA_ARCHITECTURES={arch}",
        f"-DGGML_BLAS={ggml_blas}",
        f"-DGGML_BLAS_VENDOR={blas_vendor}",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DGGML_CUDA_FORCE_MMQ={mmq}",
        "-DGGML_CUDA_F16=ON",
        "-DLLAMA_CURL=ON",
        f"-DCMAKE_CUDA_COMPILER={cuda_home/'bin'/'nvcc'}",
        f"-DCUDAToolkit_ROOT={cuda_home}",
    ]
    if cuda_flags:
        cmake_cmd.append(f"-DCMAKE_CUDA_FLAGS={cuda_flags}")
    cmake_cmd.append(f"-DGGML_RPC={'ON' if enable_rpc else 'OFF'}")

    env = {
        **os.environ,
        "PATH": str(cuda_home/"bin") + os.pathsep + os.environ.get("PATH",""),
        "LD_LIBRARY_PATH": str(cuda_home/"lib64") + os.pathsep + os.environ.get("LD_LIBRARY_PATH",""),
    }

    run(cmake_cmd, cwd=build_path/"build", env=env)
    run(["make", f"-j{os.cpu_count() or 1}"], cwd=build_path/"build", env=env)

    link_outputs(build_path)
    VERSION_FILE.write_text(version)
    log("Build files linked under ./llama-current and ./bin/*")

def test_build() -> bool:
    server = CURRENT_DIR/"build"/"bin"/"llama-server"
    if not server.exists():
        log("llama-server binary not found")
        return False
    try:
        subprocess.run([str(server),"--help"], check=True, stdout=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

def schedule_build(version: str, enable_rpc: bool):
    tmp = f"/tmp/llama_build_{version}.sh"
    distributed_flag = " --distributed" if enable_rpc else ""
    Path(tmp).write_text(f"#!/bin/bash\n{sys.executable} {__file__} --now{distributed_flag}\n")
    os.chmod(tmp, 0o755)
    if shutil.which("at"):
        run(["bash","-c",f"echo {tmp} | at 02:00"])
    else:
        log("'at' command not available; build will run on next invocation")

def main(args):
    check_dependencies()
    ref = get_ref(args.ref)
    log(f"Target llama.cpp ref: {ref}")
    if args.now:
        build_llama(ref, args.force_mmq, args.fast_math, args.blas, args.distributed)
        log("Build completed successfully" if test_build() else "Build failed")
    else:
        if is_new_version(ref):
            if datetime.now().strftime("%H") == "02":
                build_llama(ref, args.force_mmq, args.fast_math, args.blas, args.distributed)
                test_build()
                log("Scheduled build completed")
            else:
                schedule_build(ref, args.distributed); log("New version found; build scheduled for 2 AM")
        else:
            log("Already up to date")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Automated llama.cpp build (CUDA + BLAS).")
    p.add_argument("--now", action="store_true", help="build immediately")
    p.add_argument("--ref", default="b6119", help="git tag/branch/commit, or 'latest'")
    p.add_argument("--fast-math", action="store_true", help="pass --use_fast_math to NVCC")
    p.add_argument("--force-mmq", choices=["auto","on","off"], default="auto", help="toggle MMQ CUDA kernels")
    p.add_argument("--blas", choices=["auto","openblas","mkl","off"], default="auto", help="choose BLAS for CPU path")
    p.add_argument("--distributed", action="store_true", help="enable GGML RPC backend for distributed inference")
    args = p.parse_args()
    main(args)
