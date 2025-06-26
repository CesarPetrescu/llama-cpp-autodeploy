# llama.cpp Automated Build & Deployment (Python)

This project provides Python scripts to automatically build [llama.cpp](https://github.com/ggml-org/llama.cpp) and run `llama-server` with your chosen models.

## Quick Start

### Prerequisites
- Debian/Ubuntu system with CUDA capable GPU
- `git`, `cmake`, `make`, `gcc`, `g++`
- NVIDIA drivers and CUDA toolkit
- Python 3.8+

### Installation
```bash
# install Python dependencies
pip install -r requirements.txt

# run the first build immediately
python autodevops.py --now
```

### Loading a Model
```bash
# start the server with a model from Hugging Face
python loadmodel.py bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0 --llm

# use a local model file
python loadmodel.py ./models/llama-7b.gguf --llm --local
```

Set `HF_TOKEN` in `.env.local` if you need to access private models. Downloading uses `huggingface-cli` and models are stored in the `models/` directory.

### Updating
Running `python autodevops.py` without `--now` checks for new releases hourly and schedules a build at 2 AM when a new version is found. Binaries are linked into the `bin/` directory.

## Repository Layout
- `autodevops.py` – automated build script
- `loadmodel.py` – server launcher with model download support
- `requirements.txt` – Python dependencies
- `bin/` – symlinks to the latest built binaries
- `models/` – downloaded models

Both scripts roughly replicate the behaviour of the original Bash versions while providing a more portable Python implementation.
