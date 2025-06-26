#!/bin/bash

# loadmodel.bash - llama-server startup script with multiple modes
# Usage:
#   ./loadmodel.bash <repo:tag|path> [--embedding|--rerank|--llm] [options]
#   ./loadmodel.bash --model <repo:tag|path> [--embedding|--rerank|--llm] [options]

set -euo pipefail

# Default configuration
DEFAULT_HOST="127.0.0.1"
DEFAULT_PORT="8080"
DEFAULT_CONTEXT_SIZE="2048"
DEFAULT_THREADS=$(nproc)
DEFAULT_GPU_LAYERS="999"  # Auto-detect max layers

# Directory of this script and local bin with symlinked binaries
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Directory where remote models will be stored
MODELS_DIR="$SCRIPT_DIR/models"

# Load environment variables from .env or .env.local
load_env() {
    if [[ -f "$SCRIPT_DIR/.env" ]]; then
        set -a
        source "$SCRIPT_DIR/.env"
        set +a
    elif [[ -f "$SCRIPT_DIR/.env.local" ]]; then
        set -a
        source "$SCRIPT_DIR/.env.local"
        set +a
    fi
}

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
Usage: $0 <model> [--embedding|--rerank|--llm] [options]

  <model> can be either a local .gguf file or a remote specification
  in the form "user/repo:quantization" (optionally prefixed with "hf.co/").
  When a remote spec is used the model is automatically downloaded to
  "$MODELS_DIR".

Legacy syntax using --model <path> is still supported.

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
  --local             Treat <model> as a local file even if it does not exist
  --verbose            Enable verbose output
  --help               Show this help message

Examples:
  # Download and run a quantized model
  $0 bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0 --llm

  # Use a local embedding model
  $0 ./models/bge-large-en.gguf --embedding --pooling mean --local

  # Use a local reranker
  $0 ./models/bge-reranker.gguf --rerank --local

  # Custom port and context size
  $0 ./models/model.gguf --llm --port 8081 --ctx-size 4096 --local

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

# Resolve a model specification and return a local file path.
# If the specification is a local file it is returned directly.
# Remote specs follow the form "user/repo:tag" or "hf.co/user/repo:tag".
resolve_model() {
    local spec="$1"

    # If the file exists locally, return it
    if [[ -f "$spec" ]]; then
        echo "$spec"
        return 0
    fi

    # Remove optional hf.co/ prefix
    spec="${spec#hf.co/}"

    local repo="${spec%%:*}"
    local tag="${spec#*:}"

    # Determine filename
    local repo_base="${repo##*/}"
    local upper_tag="$(echo "$tag" | tr '[:lower:]' '[:upper:]')"
    local file=""
    if [[ "$tag" == *.gguf ]]; then
        file="$tag"
    else
        file="${repo_base}-${upper_tag}.gguf"
    fi

    local url="https://huggingface.co/${repo}/resolve/main/${file}"
    local local_path="${MODELS_DIR}/${file}"

    mkdir -p "$MODELS_DIR"
    if [[ ! -f "$local_path" ]]; then
        print_color "$BLUE" "Downloading $url"
        local curl_args=(-L -f -o "$local_path")
        if [[ -n "${HF_TOKEN:-}" ]]; then
            curl_args+=( -H "Authorization: Bearer $HF_TOKEN" )
        fi
        if ! curl "${curl_args[@]}" "$url"; then
            print_color "$RED" "Failed to download model from $url"
            exit 1
        fi
    else
        print_color "$GREEN" "Using cached model $local_path"
    fi

    echo "$local_path"
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
    # Load environment variables if available
    load_env

    # Parse command line arguments
    local model_spec=""
    local model_path=""
    local local_flag=false
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
                model_spec="$2"
                shift 2
                ;;
            --local)
                local_flag=true
                shift
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
                if [[ -z "$model_spec" && "$1" != -* ]]; then
                    model_spec="$1"
                    shift
                else
                    print_color "$RED" "Unknown option: $1"
                    usage
                    exit 1
                fi
                ;;
        esac
    done

    # Validate required arguments
    if [[ -z "$model_spec" ]]; then
        print_color "$RED" "ERROR: model specification is required"
        usage
        exit 1
    fi

    # Resolve model
    if [[ "$local_flag" == true ]]; then
        model_path="$model_spec"
    else
        model_path=$(resolve_model "$model_spec")
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
