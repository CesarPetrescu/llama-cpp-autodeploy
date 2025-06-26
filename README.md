# llama.cpp Automated Build & Deployment System

A comprehensive automation system for building, deploying, and managing llama.cpp with CUDA and Intel oneMKL support on Debian systems. This system automatically checks for new releases, builds them with optimal CUDA and CPU BLAS configurations, and provides easy model loading capabilities.

## 🚀 Quick Start

### Prerequisites

1. **Debian-based system** (Ubuntu, Debian, etc.)
2. **NVIDIA GPU** with CUDA support
3. **NVIDIA drivers** installed
4. **Root/sudo access** for initial setup
5. **Intel oneMKL** installed for CPU acceleration

### Installation

```bash
# 1. Make scripts executable
chmod +x autodevops.bash loadmodel.bash setup-cron.bash

# 2. Install system dependencies (automatically handled by scripts)
sudo apt update
sudo apt install build-essential cmake git curl jq nvidia-cuda-toolkit libmkl-dev

# 3. Setup automated builds
./setup-cron.bash install

# 4. Run initial build
./autodevops.bash --now
# 5. Use binaries from ./bin (created automatically)
ls -l bin
```

## 📋 System Components

### 1. `autodevops.bash` - Main Automation Script

**Purpose**: Handles automatic fetching, building, and deployment of llama.cpp releases.

**Key Features**:
- Fetches latest release from GitHub API
- Compiles with optimized CUDA settings
- Links against Intel oneMKL for CPU acceleration
- Auto-detects GPU compute capability
- Manages build versions and symlinks
- Schedules builds for 2 AM when new versions are found
- Comprehensive logging and error handling

**Usage**:
```bash
./autodevops.bash          # Check for updates (normal cron mode)
./autodevops.bash --now    # Force immediate build
```

**What it does**:
1. Checks system dependencies
2. Fetches latest release info from GitHub
3. Compares with current version
4. If new version found during hourly check → schedules 2 AM build
5. If new version found at 2 AM → builds immediately
6. Optimizes build for your specific GPU
7. Creates symlink to current build
8. Updates version tracking
9. Links latest binaries into `./bin`

### 2. `loadmodel.bash` - Model Server Launcher

**Purpose**: Provides a unified interface for starting llama-server in different modes. The script automatically uses the binaries in `./bin` created by `autodevops.bash`.

**Usage Examples**:
```bash
# Start LLM server from a local file
./loadmodel.bash ./models/llama-7b.gguf --llm --local

# Start LLM server by downloading a quantized model
./loadmodel.bash bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0 --llm

# Start embedding server
./loadmodel.bash ./models/bge-large-en.gguf --embedding --pooling mean --local

# Start rerank server
./loadmodel.bash ./models/bge-reranker.gguf --rerank --local

# Custom configuration
./loadmodel.bash ./models/model.gguf --llm --port 8081 --ctx-size 4096 --gpu-layers 50 --local
```

The launcher reads environment variables from `.env` or `.env.local` in this
directory. Set `HF_TOKEN` in one of these files to download private models.

**Available Options**:
- `--model PATH`: Explicit model path or repo:tag (optional when first argument is used)
- `--embedding`: Embedding mode
- `--rerank`: Rerank mode
- `--llm`: Normal LLM mode (default)
- `--host HOST`: Server host (default: 127.0.0.1)
- `--port PORT`: Server port (default: 8080)
- `--ctx-size SIZE`: Context size (default: 2048)
- `--threads NUM`: CPU threads (default: auto)
- `--gpu-layers NUM`: GPU layers (default: 999 = auto)
- `--pooling TYPE`: Pooling for embeddings (mean|cls|last|rank)
- `--local`: Treat the model argument as a local file
- `--verbose`: Enable verbose output

### 3. `setup-cron.bash` - Cron Management

**Purpose**: Manages the automated build schedule.

**Usage**:
```bash
./setup-cron.bash install    # Install hourly cron job
./setup-cron.bash remove     # Remove cron job
./setup-cron.bash show       # Show current cron jobs
./setup-cron.bash status     # Show system status
```

## ⚙️ System Architecture

### Directory Structure
```
./
├── llama-builds/                 # All version builds
│   ├── llama-cpp-b5747/         # Version-specific builds
│   └── llama-cpp-b5748/
├── llama-current/               # Symlink to current build
│   └── build/bin/llama-server   # Current server binary
├── bin/                         # Symlinks to latest binaries
├── models/                      # Downloaded GGUF models
├── llama-cpp-latest -> llama-builds/llama-cpp-bXXXX/
├── .llama-version              # Current version tracking
├── autodevops.log              # Main operation log
└── autodevops-cron.log         # Cron execution log
```

### Automation Flow

1. **Hourly Check** (via cron): Check GitHub for new releases
2. **Version Comparison**: Compare with locally stored version
3. **Schedule Decision**: If new version found → schedule 2 AM build
4. **Build Process**: At 2 AM → full compilation with CUDA optimization
5. **Deployment**: Update symlinks and version tracking
6. **Testing**: Verify build functionality

### CUDA Optimization

The system automatically:
- Detects GPU compute capability using `nvidia-smi`
- Configures CMake with optimal CUDA settings
- Enables CUDA graphs and optimized kernels
- Sets appropriate compute architectures
- Maximizes GPU utilization
- Links against Intel oneMKL for fast CPU operations

## 🔧 Configuration

### Environment Variables

```bash
# Optional: Override default directories
export LLAMA_BUILD_DIR="$PWD/llama-builds"
export LLAMA_CURRENT_DIR="$PWD/llama-current"

# Optional: Hugging Face token for private models
export HF_TOKEN="your-hf-token"

# Optional: CUDA paths (usually auto-detected)
export CUDA_PATH="/usr/local/cuda"
export CUDA_HOME="/usr/local/cuda"
# Optional: MKL paths if not in default location
# export MKLROOT="/opt/intel/oneapi/mkl/latest"
```

Create a `.env.local` file with `HF_TOKEN` to authenticate when downloading
private models. The `.env` file is ignored by git so you can keep sensitive
tokens out of version control.

### Cron Schedule

The default cron job runs every hour:
```cron
0 * * * * cd "/path/to/scripts" && ./autodevops.bash >> ./autodevops-cron.log 2>&1
```

To modify the schedule, edit with `crontab -e` or use `setup-cron.bash remove` and manually add a custom schedule.

## 📊 Monitoring & Logging

### Log Files

- `./autodevops.log`: Main operation log with timestamps
- `./autodevops-cron.log`: Cron job execution log

### Status Checking

```bash
# Check system status
./setup-cron.bash status

# View recent logs
tail -f ./autodevops.log

# Check current version
cat ./.llama-version

# Test current build
./bin/llama-server --help
```

## 🚨 Troubleshooting

### Common Issues

**1. CUDA Not Found**
```bash
# Install CUDA toolkit
sudo apt install nvidia-cuda-toolkit

# Verify installation
nvcc --version
nvidia-smi
```

**2. Build Fails**
```bash
# Check dependencies
./setup-cron.bash status

# Manual build with verbose output
./autodevops.bash --now
```

**3. Server Won't Start**
```bash
# Check model path
ls -la /path/to/your/model.gguf

# Test with minimal options
./loadmodel.bash --model "model.gguf" --llm --verbose
```

**4. Cron Job Not Running**
```bash
# Check cron service
sudo systemctl status cron

# Verify cron job
crontab -l

# Check cron logs
grep CRON /var/log/syslog
```

### Manual Recovery

If automation fails, manually build:
```bash
cd $HOME
git clone https://github.com/ggml-org/llama.cpp.git manual-build
cd manual-build
mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## 🔒 Security Considerations

- Scripts run with user permissions (not root)
- GitHub API calls are unauthenticated (public repo)
- Builds are isolated in user directory
- Log files contain no sensitive information
- Cron jobs use full paths to prevent PATH injection

## 🎯 Use Cases

### Development Environment
- Automatic updates ensure latest features and bug fixes
- Consistent build environment across team
- Easy model switching for testing

### Production Environment  
- Controlled 2 AM build schedule minimizes disruption
- Version tracking enables rollbacks
- Automated testing prevents broken deployments

### Research Environment
- Always have latest model support
- Easy switching between embedding and LLM modes
- Optimal CUDA performance

## 📝 Advanced Usage

### Custom Build Options

By default the automation script builds with NVIDIA CUDA and Intel oneMKL. If
you need to tweak the build configuration manually, use options similar to the
following:

Modify `autodevops.bash` to add custom CMake options:
```bash
cmake .. \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$compute_arch" \
    -DLLAMA_BLAS=ON \
    -DLLAMA_BLAS_VENDOR=Intel10_64lp \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA_FORCE_MMQ=OFF \
    -DGGML_CUDA_F16=ON \
    -DLLAMA_CURL=ON \
    -DLLAMA_STATIC=ON      # Add custom options here
```

### Multiple Model Management

Create wrapper scripts for different models:
```bash
#!/bin/bash
# llm-server.sh
./loadmodel.bash ./models/llama-7b.gguf --llm --port 8080 --local

#!/bin/bash  
# embed-server.sh
./loadmodel.bash ./models/bge-large.gguf --embedding --port 8081 --local
```

### Integration with External Tools

The server provides OpenAI-compatible API endpoints:
- `http://localhost:8080/v1/chat/completions` (LLM mode)
- `http://localhost:8080/v1/embeddings` (embedding mode)
- `http://localhost:8080/v1/rerank` (rerank mode)

## 🤝 Contributing

To extend the system:
1. Fork the scripts
2. Add new features to individual components
3. Update documentation
4. Test on clean Debian system
5. Submit improvements

The modular design allows easy customization of individual components without affecting the overall automation flow.# llama-cpp-autodeploy
