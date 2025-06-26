#!/bin/bash

# autodevops.bash - Automated llama.cpp build and deployment system
# This script fetches the latest llama.cpp release, compiles with CUDA support,
# and manages automated builds

set -euo pipefail

# Configuration
REPO_URL="https://github.com/ggml-org/llama.cpp"
API_URL="https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
BUILD_DIR="$SCRIPT_DIR/llama-builds"
CURRENT_DIR="$SCRIPT_DIR/llama-current"
LOG_FILE="$SCRIPT_DIR/autodevops.log"
VERSION_FILE="$SCRIPT_DIR/.llama-version"

# Directory of this script and local bin for symlinks
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_BIN="$SCRIPT_DIR/bin"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Print colored output
print_color() {
    local color=$1
    shift
    echo -e "${color}$*${NC}"
}

# Check if --now flag is provided
NOW_FLAG=false
if [[ "${1:-}" == "--now" ]]; then
    NOW_FLAG=true
    print_color "$BLUE" "Running immediate build..."
fi

# Function to check system dependencies
check_dependencies() {
    log "Checking system dependencies..."

    local deps=("git" "cmake" "make" "gcc" "g++" "curl" "jq")
    local missing_deps=()

    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing_deps+=("$dep")
        fi
    done

    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        print_color "$RED" "Missing dependencies: ${missing_deps[*]}"
        print_color "$YELLOW" "Installing missing dependencies..."
        sudo apt update
        sudo apt install -y build-essential cmake git curl jq nvidia-cuda-toolkit libmkl-dev
        log "Dependencies installed"
    fi

    # Check CUDA toolkit
    if ! command -v nvcc &> /dev/null; then
        print_color "$RED" "CUDA compiler (nvcc) not found in your PATH."
        print_color "$YELLOW" "Please run 'source /etc/profile.d/cuda.sh' or log out and log back in."
        exit 1
    fi

    # Check NVIDIA drivers
    if ! command -v nvidia-smi &> /dev/null; then
        print_color "$RED" "NVIDIA drivers not found. Please install NVIDIA drivers first."
        exit 1
    fi

    log "All dependencies satisfied"
}

# Function to get latest release info
get_latest_release() {
    log "Fetching latest release information..." >&2

    local release_info
    release_info=$(curl -s "$API_URL")

    if [[ $? -ne 0 ]]; then
        log "ERROR: Failed to fetch release information"
        exit 1
    fi

    local tag_name
    tag_name=$(echo "$release_info" | jq -r '.tag_name')

    if [[ "$tag_name" == "null" || -z "$tag_name" ]]; then
        log "ERROR: Could not parse tag name from release"
        exit 1
    fi

    echo "$tag_name"
}

# Function to check if version is different
is_new_version() {
    local latest_version=$1
    local current_version=""

    if [[ -f "$VERSION_FILE" ]]; then
        current_version=$(cat "$VERSION_FILE")
    fi

    if [[ "$latest_version" != "$current_version" ]]; then
        return 0  # New version available
    else
        return 1  # Same version
    fi
}

# Function to get GPU compute capability
get_compute_capability() {
    local compute_cap
    compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1)

    if [[ -z "$compute_cap" ]]; then
        log "WARNING: Could not determine GPU compute capability, using default" >&2
        echo "75"  # Default for modern GPUs
    else
        # Remove decimal point for CMake
        echo "$compute_cap" | tr -d '.'
    fi
}

# Function to build llama.cpp
build_llama_cpp() {
    local version=$1
    local build_path="$BUILD_DIR/llama-cpp-$version"

    log "Building llama.cpp version $version..."

    # Create build directory
    mkdir -p "$BUILD_DIR"

    # Clone or update repository
    if [[ -d "$build_path" ]]; then
        log "Removing existing build directory..."
        rm -rf "$build_path"
    fi

    log "Cloning repository..."
    git clone --depth 1 --branch "$version" "$REPO_URL" "$build_path"

    cd "$build_path"

    # Get compute capability
    local compute_arch
    compute_arch=$(get_compute_capability)
    log "Using GPU compute capability: $compute_arch"

    # Create build directory
    mkdir -p build
    cd build

    log "Configuring build with CMake..."
    cmake .. \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="$compute_arch" \
        -DLLAMA_BLAS=ON \
        -DLLAMA_BLAS_VENDOR=Intel10_64lp \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA_FORCE_MMQ=OFF \
        -DGGML_CUDA_F16=ON \
        -DLLAMA_CURL=ON

    log "Building (this may take several minutes)..."
    make -j$(nproc)

    log "Build completed successfully"

    # Update current symlink
    if [[ -L "$CURRENT_DIR" ]]; then
        rm "$CURRENT_DIR"
    elif [[ -d "$CURRENT_DIR" ]]; then
        rm -rf "$CURRENT_DIR"
    fi

    ln -sf "$build_path" "$CURRENT_DIR"
    log "Updated current build symlink"

    # Update local symlinks in repository
    mkdir -p "$LOCAL_BIN"
    ln -sfn "$build_path" "$SCRIPT_DIR/llama-cpp-latest"
    for f in "$build_path"/build/bin/*; do
        if [[ -f "$f" ]]; then
            ln -sfn "$f" "$LOCAL_BIN/$(basename "$f")"
        fi
    done
    log "Updated local repository bin symlinks"

    # Update version file
    echo "$version" > "$VERSION_FILE"
    log "Updated version file"
}

# Function to schedule build for 2 AM
schedule_build() {
    local version=$1
    log "Scheduling build for $version at 2 AM..."

    # Create a temporary script for the 2 AM build
    local temp_script="/tmp/llama_build_$version.sh"
    cat > "$temp_script" << EOF
#!/bin/bash
cd "\$(dirname "\$0")"
$0 --now
EOF
    chmod +x "$temp_script"

    # Schedule with at command
    if command -v at &> /dev/null; then
        echo "$temp_script" | at 02:00
        log "Build scheduled for 2 AM using 'at' command"
    else
        log "WARNING: 'at' command not available, build will run on next hourly cron"
    fi
}

# Function to test build
test_build() {
    log "Testing build..."

    if [[ ! -f "$CURRENT_DIR/build/bin/llama-server" ]]; then
        log "ERROR: llama-server binary not found"
        return 1
    fi

    # Test that the binary can run (just check help)
    if "$CURRENT_DIR/build/bin/llama-server" --help &> /dev/null; then
        log "Build test successful"
        return 0
    else
        log "ERROR: Build test failed"
        return 1
    fi
}

# Main execution
main() {
    log "Starting autodevops script (NOW_FLAG: $NOW_FLAG)"

    check_dependencies

    local latest_version
    latest_version=$(get_latest_release)
    log "Latest version: $latest_version"

    if [[ "$NOW_FLAG" == true ]]; then
        # Force build now
        build_llama_cpp "$latest_version"
        test_build
        print_color "$GREEN" "Build completed successfully!"

    elif is_new_version "$latest_version"; then
        log "New version detected: $latest_version"

        # Check if it's between 2-3 AM, if so build now
        local current_hour
        current_hour=$(date +%H)

        if [[ "$current_hour" == "02" ]]; then
            log "Building now (2 AM schedule)"
            build_llama_cpp "$latest_version"
            test_build
            print_color "$GREEN" "Scheduled build completed successfully!"
        else
            log "Scheduling build for 2 AM"
            schedule_build "$latest_version"
            print_color "$YELLOW" "New version found, build scheduled for 2 AM"
        fi
    else
        log "No new version available"
        print_color "$GREEN" "Already up to date"
    fi

    log "Script completed"
}

# Error handling
trap 'log "Script failed with error code $?"' ERR

# Run main function
main "$@"
