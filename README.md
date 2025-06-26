# llama.cpp Automated Build & Deployment (Python)

This project provides Python scripts to automatically build [llama.cpp](https://github.com/ggml-org/llama.cpp) and run `llama-server` with your chosen models. Supports LLM text generation, embeddings, and document reranking.

## Features

- **LLM Server**: Text generation and chat completions
- **Embeddings**: Generate vector embeddings for text
- **Reranking**: Semantic document ranking and scoring
- **Auto-build**: Automated compilation with CUDA support
- **Model Management**: Download and manage GGUF models

## Quick Start

### Prerequisites
- Debian/Ubuntu system with CUDA capable GPU
- `git`, `cmake`, `make`, `gcc`, `g++`
- NVIDIA drivers and CUDA toolkit
- Python 3.8+

### Installation
```bash
# install Python dependencies (includes vLLM for reranking)
pip install -r requirements.txt

# run the first build immediately (includes reranking support)
python autodevops.py --now
```

### Model Types & Usage

#### 1. LLM Models (Text Generation)
```bash
# start LLM server
python loadmodel.py bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0 --llm

# use local model
python loadmodel.py ./models/llama-7b.gguf --llm --local
```

#### 2. Embedding Models
```bash
# start embedding server
python loadmodel.py sentence-transformers/all-MiniLM-L6-v2-GGUF:Q8_0 --embed

# local embedding model
python loadmodel.py ./models/bge-embedding-model.gguf --embed --local
```

#### 3. Reranking Models
```bash

# start reranking server (GGUF models)
python loadmodel.py gpustack/bge-reranker-v2-m3-GGUF:Q8_0 --rerank

# start Qwen3 reranker server via vLLM
python loadmodel.py Qwen/Qwen3-Reranker-4B --rerank

# run Qwen3 reranker manually with vLLM
python vllm.py Qwen/Qwen3-Reranker-4B --host 0.0.0.0 --port 8000

# local reranker model
python loadmodel.py ./models/bge-reranker-v2-m3-Q8_0.gguf --rerank --local
```

Set `HF_TOKEN` in `.env.local` if you need to access private models. Downloading uses `huggingface-cli` and models are stored in the `models/` directory.

## API Usage Examples

### LLM Text Generation

#### Health Check
```bash
curl -X GET http://127.0.0.1:8080/health
```

#### Chat Completions
```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama-3.2-3b",
        "messages": [
            {"role": "user", "content": "What is artificial intelligence?"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }'
```

#### Text Completions
```bash
curl -X POST http://127.0.0.1:8080/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama-3.2-3b",
        "prompt": "The future of AI is",
        "max_tokens": 50,
        "temperature": 0.7
    }'
```

### Embeddings

#### Generate Embeddings
```bash
curl -X POST http://127.0.0.1:8080/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{
        "model": "text-embedding-ada-002",
        "input": [
            "Artificial intelligence is transforming technology",
            "Machine learning enables pattern recognition"
        ]
    }'
```

### Document Reranking

#### Basic Reranking
```bash
curl -X POST http://127.0.0.1:8080/rerank \
    -H "Content-Type: application/json" \
    -d '{
        "query": "What is artificial intelligence?",
        "documents": [
            "Artificial intelligence (AI) is intelligence demonstrated by machines.",
            "Machine learning is a subset of artificial intelligence.",
            "Python is a programming language used for AI development.",
            "The weather is nice today."
        ]
    }'
```

#### Multilingual Reranking
```bash
curl -X POST http://127.0.0.1:8080/rerank \
    -H "Content-Type: application/json" \
    -d '{
        "query": "machine learning algorithms",
        "documents": [
            "Linear regression is a basic statistical method.",
            "Random forests are ensemble learning methods.",
            "Les algorithmes d'\''apprentissage automatique sont puissants.",
            "神经网络是深度学习的基础。"
        ]
    }'
```

### Updating
Running `python autodevops.py` without `--now` checks for new releases hourly and schedules a build at 2 AM when a new version is found. Binaries are linked into the `bin/` directory.

## TODO

### Model Downloads from HuggingFace
The `loadmodel.py` script supports automatic model downloads via HuggingFace Hub. To extend model support:

- **LLM Models**: Add popular instruction-tuned models (Llama, Mistral, Qwen families)
- **Embedding Models**: Include more sentence-transformers GGUF models 
- **Reranking Models**: Expand BGE and other reranking model collections
- **Model Caching**: Implement smart caching strategies for frequently used models
- **Model Validation**: Add GGUF format validation before server startup

Use the HuggingFace model format: `organization/model-name:quantization` (e.g., `bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0`)

## Repository Layout
- `autodevops.py` – automated build script
- `loadmodel.py` – server launcher with model download support
- `requirements.txt` – Python dependencies
- `bin/` – symlinks to the latest built binaries
- `models/` – downloaded models

Both scripts roughly replicate the behaviour of the original Bash versions while providing a more portable Python implementation.
