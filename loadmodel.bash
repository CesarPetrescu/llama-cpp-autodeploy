#!/bin/bash

# loadmodel.bash - llama-server startup script with multiple modes
# Usage: ./loadmodel.bash --model "path/to/model.gguf" --embedding|--rerank|--llm [additional options]

set -euo pipefail

# Default configuration
DEFAULT_HOST="127.0.0.1"
DEFAULT_PORT="8080"
DEFAULT_CONTEXT_SIZE="2048"
DEFAULT_THREADS=$(nproc)
DEFAULT_GPU_LAYERS="999"  # Auto-detect max layers

# Directory of this script and local bin with symlinked binaries
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_color() {
    local color=$1
    shift
    echo -e "${color}$*${NC}"
}

# Usage function
usage() {
    cat << EOF
Usage: $0 --model "path/to/model.gguf" --embedding|--rerank|--llm [options]

Required arguments:
  --model PATH          Path to the GGUF model file

Mode selection (choose one):
  --embedding           Run in embedding mode
  --rerank             Run in rerank mode  
  --llm                Run in normal LLM mode (default)

Optional arguments:
  --host HOST          Server host (default: $DEFAULT_HOST)
  --port PORT          Server port (default: $DEFAULT_PORT)
  --ctx-size SIZE      Context size (default: $DEFAULT_CONTEXT_SIZE)
  --threads NUM        Number of threads (default: $DEFAULT_THREADS)
  --gpu-layers NUM     GPU layers to offload (default: $DEFAULT_GPU_LAYERS)
  --pooling TYPE       Pooling type for embeddings (mean|cls|last|rank)
  --verbose            Enable verbose output
  --help               Show this help message

Examples:
  # Run LLM server
  $0 --model "./models/llama-7b.gguf" --llm

  # Run embedding server
  $0 --model "./models/bge-large-en.gguf" --embedding --pooling mean

  # Run rerank server
  $0 --model "./models/bge-reranker.gguf" --rerank

  # Custom port and context size
  $0 --model "./models/model.gguf" --llm --port 8081 --ctx-size 4096

EOF
}

# Check if llama.cpp is available
check_llama_cpp() {
    local llama_server_path=""

    # Check multiple possible locations
    local possible_paths=(
        "$SCRIPT_DIR/bin/llama-server"
        "$SCRIPT_DIR/llama-cpp-latest/build/bin/llama-server"
        "$HOME/llama-current/build/bin/llama-server"
        "$HOME/llama-current/llama-server"
        "./llama-server"
        "/usr/local/bin/llama-server"
        "$(which llama-server 2>/dev/null || true)"
    )

    for path in "${possible_paths[@]}"; do
        if [[ -n "$path" && -f "$path" && -x "$path" ]]; then
            llama_server_path="$path"
            break
        fi
    done

    if [[ -z "$llama_server_path" ]]; then
        print_color "$RED" "ERROR: llama-server not found!"
        print_color "$YELLOW" "Please ensure llama.cpp is built and available."
        print_color "$YELLOW" "Run autodevops.bash first to build llama.cpp."
        exit 1
    fi

    echo "$llama_server_path"
}

# Validate model file
validate_model() {
    local model_path="$1"

    if [[ ! -f "$model_path" ]]; then
        print_color "$RED" "ERROR: Model file not found: $model_path"
        exit 1
    fi

    if [[ ! "$model_path" =~ \.gguf$ ]]; then
        print_color "$YELLOW" "WARNING: Model file does not have .gguf extension"
    fi

    print_color "$GREEN" "Model file validated: $model_path"
}

# Get GPU information
get_gpu_info() {
    if command -v nvidia-smi &> /dev/null; then
        local gpu_count
        gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits)
        local gpu_memory
        gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

        print_color "$BLUE" "GPU detected: $gpu_count GPU(s) with ${gpu_memory}MB VRAM"
        return 0
    else
        print_color "$YELLOW" "No NVIDIA GPU detected, using CPU mode"
        return 1
    fi
}

# Build command arguments
build_command() {
    local model_path="$1"
    local mode="$2"
    local host="$3"
    local port="$4"
    local ctx_size="$5"
    local threads="$6"
    local gpu_layers="$7"
    local pooling="$8"
    local verbose="$9"
    local llama_server_path="${10}"

    local cmd_args=()

    # Basic arguments
    cmd_args+=("--model" "$model_path")
    cmd_args+=("--host" "$host")
    cmd_args+=("--port" "$port")
    cmd_args+=("--ctx-size" "$ctx_size")
    cmd_args+=("--threads" "$threads")

    # GPU layers (only if GPU is available)
    if get_gpu_info; then
        cmd_args+=("--n-gpu-layers" "$gpu_layers")
    fi

    # Mode-specific arguments
    case "$mode" in
        "embedding")
            cmd_args+=("--embeddings")
            if [[ -n "$pooling" ]]; then
                cmd_args+=("--pooling" "$pooling")
            fi
            ;;
        "rerank")
            cmd_args+=("--rerank")
            ;;
        "llm")
            # Default LLM mode, no special flags needed
            ;;
    esac

    # Verbose mode
    if [[ "$verbose" == true ]]; then
        cmd_args+=("--verbose")
    fi

    # Additional useful flags
    cmd_args+=("--log-format" "text")
    cmd_args+=("--timeout" "600")

    echo "${cmd_args[@]}"
}

# Main function
main() {
    # Parse command line arguments
    local model_path=""
    local mode="llm"  # default mode
    local host="$DEFAULT_HOST"
    local port="$DEFAULT_PORT"
    local ctx_size="$DEFAULT_CONTEXT_SIZE"
    local threads="$DEFAULT_THREADS"
    local gpu_layers="$DEFAULT_GPU_LAYERS"
    local pooling=""
    local verbose=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --model)
                model_path="$2"
                shift 2
                ;;
            --embedding)
                mode="embedding"
                shift
                ;;
            --rerank)
                mode="rerank"
                shift
                ;;
            --llm)
                mode="llm"
                shift
                ;;
            --host)
                host="$2"
                shift 2
                ;;
            --port)
                port="$2"
                shift 2
                ;;
            --ctx-size)
                ctx_size="$2"
                shift 2
                ;;
            --threads)
                threads="$2"
                shift 2
                ;;
            --gpu-layers)
                gpu_layers="$2"
                shift 2
                ;;
            --pooling)
                pooling="$2"
                shift 2
                ;;
            --verbose)
                verbose=true
                shift
                ;;
            --help)
                usage
                exit 0
                ;;
            *)
                print_color "$RED" "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    # Validate required arguments
    if [[ -z "$model_path" ]]; then
        print_color "$RED" "ERROR: --model argument is required"
        usage
        exit 1
    fi

    # Check system
    local llama_server_path
    llama_server_path=$(check_llama_cpp)
    validate_model "$model_path"

    # Build command
    local cmd_args
    cmd_args=$(build_command "$model_path" "$mode" "$host" "$port" "$ctx_size" "$threads" "$gpu_layers" "$pooling" "$verbose" "$llama_server_path")

    # Display configuration
    print_color "$BLUE" "Starting llama-server with configuration:"
    print_color "$BLUE" "  Mode: $mode"
    print_color "$BLUE" "  Model: $model_path"
    print_color "$BLUE" "  Host: $host"
    print_color "$BLUE" "  Port: $port"
    print_color "$BLUE" "  Context Size: $ctx_size"
    print_color "$BLUE" "  Threads: $threads"
    print_color "$BLUE" "  GPU Layers: $gpu_layers"
    if [[ -n "$pooling" ]]; then
        print_color "$BLUE" "  Pooling: $pooling"
    fi

    print_color "$GREEN" "Server will be available at: http://$host:$port"
    print_color "$YELLOW" "Press Ctrl+C to stop the server"

    # Start the server
    print_color "$GREEN" "Starting server..."
    exec "$llama_server_path" $cmd_args
}

# Error handling
trap 'print_color "$RED" "Server stopped with error code $?"' ERR

# Check if no arguments provided
if [[ $# -eq 0 ]]; then
    usage
    exit 1
fi

# Run main function
main "$@"
