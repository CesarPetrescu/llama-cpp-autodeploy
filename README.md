# llama.cpp Automated Build & Deployment (Python)

Automated scripts for building [llama.cpp](https://github.com/ggml-org/llama.cpp) and running
`llama-server` with your models. Supports LLM generation, embeddings and document reranking
with NVIDIA GPUs and Intel CPUs via oneAPI MKL.

## Supported platforms

- **Debian & Ubuntu** (22.04+ recommended)
- **Arch-based distros** such as Manjaro (fully tested with `pacman` package tooling)

Other modern Linux distributions with CUDA 12.8+ should work, but you may need to adapt the
package manager commands below.

## Features

- Automated build with CUDA, MMQ kernels and optional Intel oneAPI MKL
- Multi-GPU support via tensor splitting
- Interactive TUIs:
  - `autodevops_cli.py` for guided llama.cpp builds with hardware-aware presets
  - `loadmodel_cli.py` for launching LLM, embedding, and reranker servers with live memory planning
  - `loadmodel_dist_cli.py` for orchestrating distributed RPC inference with network auto-discovery on port 5515
  - `rpc_server_cli.py` for launching standalone rpc-server instances with custom ports and device bindings
- Sample launch scripts for Qwen models in `run/`

## Interactive CLIs

Two text user interfaces are included for day-to-day workflows:

- `autodevops_cli.py` probes your hardware and walks you through recommended llama.cpp build presets. Wide terminals automatically split the view, keeping toggles on the left and rich help text on the right.
- `loadmodel_cli.py` lets you browse local GGUFs or remote Hugging Face models, tune runtime flags, and preview GPU/CPU memory usage with live scrollable charts.

Launch either script with `python <script_name>` inside the virtual environment. Use the arrow keys to navigate, `PgUp/PgDn` to scroll long lists, and the on-screen instructions for editing values.

### Example: Distributed Setup Across Two PCs

Assume **PC A** will run the main `llama-cli` process and **PC B** will host a worker. Both machines need to build llama.cpp with the distributed RPC backend (`Enable distributed RPC backend` in `autodevops_cli.py`).

1. **On PC B (worker node)**
   ```bash
   cd /path/to/llama-cpp-autodeploy
   source venv/bin/activate
   python rpc_server_cli.py --host 0.0.0.0 --port 5515 --devices 0 --cache /tmp/llama-cache
   ```
   Leave this process running; it listens for RPC requests on the chosen port.

2. **On PC A (controller)**
   ```bash
   cd /path/to/llama-cpp-autodeploy
   source venv/bin/activate
   python loadmodel_dist_cli.py
   ```
   - Select the models directory (defaults to `./models`) and pick a GGUF file.
   - On startup the launcher scans local subnets for rpc-servers on port 5515; use “Scan network for rpc-server” if the worker was started later.
   - Ensure the discovered host (e.g. `192.168.1.20:5515`) appears in “Worker hosts” or add it manually.
   - Optionally enable “Launch local rpc-server” to run an additional worker on PC A.
   - Choose “Launch distributed llama-cli” to start inference; the tool runs `./bin/llama-cli -m <model> --rpc <hosts>` automatically.

3. **Verification**
   - Watch PC B’s terminal for incoming rpc-server logs.
   - On PC A, the status line reports discovery/connection results; rerun the launcher to change prompts or host lists at any time.

## Requirements

- Debian/Ubuntu **or** Arch/Manjaro with NVIDIA driver and CUDA Toolkit (12.8+ recommended)
- Build tools: `git`, `cmake`, `make`, `gcc`, `g++`, `pkg-config`
- Python 3.8+

- BLAS runtime for CPU acceleration (optional but recommended)
  - [Intel oneAPI MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html)
  - or [OpenBLAS](https://www.openblas.net/)

- PyTorch CUDA 12.9 wheels are served from the official PyTorch index and the provided
  `requirements.txt` already sets `--extra-index-url https://download.pytorch.org/whl/cu129`.
  Ensure your `pip` version is recent enough (23.0+) to respect this flag.

### Install system build dependencies

**Debian/Ubuntu**

```bash
sudo apt update
sudo apt install -y git cmake build-essential pkg-config
```

**Manjaro / Arch**

```bash
sudo pacman -Sy --needed git cmake base-devel pkgconf
```


### Install the NVIDIA CUDA Toolkit (required for fast-math builds)

The **Enable fast math** toggle in `autodevops_cli.py` enables NVCC's
`--use_fast_math` flag. The option remains disabled until the CUDA Toolkit (and
its `nvcc` compiler) is detected. The wizard shows *“NVCC (CUDA Toolkit) not
found — set CUDA_HOME or install cuda-toolkit”* when this happens.

1. Install the toolkit for your distribution.
2. Ensure `nvcc` is on your `$PATH` or set `CUDA_HOME` to the toolkit prefix
   (commonly `/usr/local/cuda`).
3. Re-run `python autodevops_cli.py` to refresh the hardware scan.

**Debian / Ubuntu (via NVIDIA’s APT repository)**

```bash
sudo apt update
sudo apt install -y software-properties-common gnupg lsb-release
DIST_ID=$(lsb_release -is | tr '[:upper:]' '[:lower:]')
DIST_VER=$(lsb_release -rs | tr -d .)
CUDA_REPO=${DIST_ID}${DIST_VER}
wget https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO}/x86_64/cuda-${CUDA_REPO}.pin
sudo mv cuda-${CUDA_REPO}.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO}/x86_64/3bf863cc.pub
sudo mv 3bf863cc.pub /usr/share/keyrings/nvidia-cuda-repo.gpg
echo "deb [signed-by=/usr/share/keyrings/nvidia-cuda-repo.gpg] https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO}/x86_64/ /" \
  | sudo tee /etc/apt/sources.list.d/cuda-${CUDA_REPO}.list
sudo apt update
sudo apt install -y cuda-toolkit-12-4
```

> Substitute `12-4` with the latest toolkit branch if needed. Log out/in after
> installation so the PATH additions in `/etc/profile.d/cuda.sh` load.

**Manjaro / Arch**

```bash
sudo pacman -Sy --needed cuda
```

The package installs to `/opt/cuda`. Export `CUDA_HOME` so the wizard can find
it:

```bash
echo 'export CUDA_HOME=/opt/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### Verify CUDA Toolkit detection

Check that `nvcc` is reachable before enabling fast math:

```bash
nvcc --version
which nvcc
```

If the commands print valid paths, the fast-math toggle becomes available. If
they fail, verify `CUDA_HOME` points at the toolkit directory.


### Install system build dependencies

**Debian/Ubuntu**

```bash
sudo apt update
sudo apt install -y git cmake build-essential pkg-config
```

**Manjaro / Arch**

```bash
sudo pacman -Sy --needed git cmake base-devel pkgconf
```

### Install Intel oneAPI MKL via APT

```bash
sudo mkdir -p /usr/share/keyrings
wget -qO- https://apt.repos.intel.com/oneapi/Intel-GPG-KEY-INTEL-PSR2.pub \
  | sudo gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneapi.list
sudo apt update
sudo apt install -y intel-oneapi-mkl intel-oneapi-mkl-devel intel-oneapi-openmp
```

> **Arch/Manjaro:** Intel distributes oneAPI MKL via the AUR (e.g. `yay -S intel-oneapi-mkl`).


### Install OpenBLAS instead of MKL

If you prefer OpenBLAS (or cannot use MKL), install the packaged libraries:

**Debian/Ubuntu**

```bash
sudo apt install -y libopenblas-dev
```

**Manjaro / Arch**

```bash
sudo pacman -S --needed openblas
```

## Quick Start

```bash
# 1) System dependencies
#    Debian/Ubuntu:
#      sudo apt update && sudo apt install -y git cmake build-essential pkg-config
#    Manjaro/Arch:
#      sudo pacman -Sy --needed git cmake base-devel pkgconf

# 2) Python environment (PyTorch CUDA wheels will be pulled from the PyTorch cu129 index)
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# 3a) Build llama.cpp via the interactive wizard (recommended)
python autodevops_cli.py

# 3b) Or run the build directly with CLI flags
#     Use --blas mkl if oneAPI MKL is installed, otherwise use --blas openblas or omit the flag for auto-detect.
python autodevops.py --now --fast-math --force-mmq=on --blas mkl

# 4) Optional: set OpenMP/MKL threading for Intel hybrid CPUs
source /opt/intel/oneapi/setvars.sh --force
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export MKL_DYNAMIC=FALSE
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export GGML_N_THREADS=8

# 5) Launch a model
# Interactive launcher for choosing model/server and tuning memory usage
python loadmodel_cli.py

# Distributed RPC launcher (optional, requires GGML_RPC build)
python loadmodel_dist_cli.py

# Standalone rpc-server helper
python rpc_server_cli.py --port 5515 --devices 0

# LLM
python loadmodel.py --llm <org/repo:quant or path/to/model.gguf>

# Embeddings
python loadmodel.py --embed <org/repo:quant or path/to/model.gguf>

# Reranker (Transformers)
python loadmodel.py --rerank <HF model id>
```

> If you request `--blas mkl` or `--blas openblas` without the corresponding libraries installed,
> `autodevops.py` will abort with installation hints. Leaving `--blas` at the default `auto`
> builds without BLAS when neither runtime is detected.

## AutodevOps CLI presets

The `autodevops_cli.py` wizard autodetects your CPU/GPU capabilities and surfaces only the
presets that make sense for the machine you are on. Each option includes inline help so you
can press the arrow keys to highlight a field and understand what enabling it will do.

- **Hardware-aware backends:** CUDA, ROCm, SYCL, Vulkan, or CPU-only builds with curated
  CMake snippets pulled directly from the comprehensive compilation guide.
- **CPU tuning profiles:** Intel AVX2/AVX-512, AMD Zen, ARM64, or portable builds complete
  with the matching compiler flags.
- **Runtime helpers:** Toggle Flash Attention reminders, unified memory guidance, and runtime
  launch templates for balanced, high-memory, constrained, or multi-GPU systems.
- **Quantisation advice:** Get reminders about which GGUF families (FP16, INT8, Q4_K_M) pair
  best with your hardware budget.

When you exit the wizard the planned build recipe is printed back to the terminal, including
per-backend instructions and runtime suggestions. Builds are only kicked off automatically
when **Build immediately** remains checked.

### Running tests

Unit tests cover the wizard configuration logic and helper recipes. Run them with:

```bash
python -m unittest discover -s tests
```

### Multi-GPU example

```bash
./bin/llama-server \
  --model ./models/llama-7b.gguf \
  --n-gpu-layers 999 \
  --tensor-split 50,50
```

### Verify MKL linkage

```bash
ldd ./bin/llama-server | egrep 'mkl|iomp|omp'
```

## Samples

The `run/` directory contains example launch scripts for:

- `run_qwen30b_llm.sh` – Qwen3 30B LLM server on port 45540
- `run_qwen_embed8b.sh` – Qwen3 Embedding 8B server on port 45541
- `run_qwen_reranker8b.sh` – Qwen3 Reranker 8B server on port 45542

These scripts demonstrate how to expose the services for integration with other tooling.
