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
- `loadmodel.py` launcher for:
  - llama.cpp LLM and embeddings servers with automatic GGUF model download
  - Transformers-based reranker compatible with the llama.cpp `/rerank` API
- Sample launch scripts for Qwen models in `run/`

## Requirements

- Debian/Ubuntu **or** Arch/Manjaro with NVIDIA driver and CUDA Toolkit (12.8+ recommended)
- Build tools: `git`, `cmake`, `make`, `gcc`, `g++`, `pkg-config`
- Python 3.8+
- [Intel oneAPI MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html)
  (optional but recommended for fastest CPU BLAS)

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
> You can skip MKL if you prefer OpenBLAS.

## Quick Start

```bash
# 1) System dependencies
#    Debian/Ubuntu:
#      sudo apt update && sudo apt install -y git cmake build-essential pkg-config
#    Manjaro/Arch:
#      sudo pacman -Sy --needed git cmake base-devel pkgconf

# 2) Python environment
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# 3) Build llama.cpp (CUDA + MKL + MMQ)
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
# LLM
python loadmodel.py --llm <org/repo:quant or path/to/model.gguf>

# Embeddings
python loadmodel.py --embed <org/repo:quant or path/to/model.gguf>

# Reranker (Transformers)
python loadmodel.py --rerank <HF model id>
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

