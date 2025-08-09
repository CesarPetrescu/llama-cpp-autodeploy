#!/usr/bin/env bash
set -euo pipefail

# Tunables
REF="${REF:-b6119}"                  # or 'latest'
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
ARCH="${ARCH:-89}"                   # 4090 = 89
FORCE_MMQ="${FORCE_MMQ:-ON}"         # ON/OFF
FAST_MATH="${FAST_MATH:-1}"          # 1/0
BLAS_MODE="${BLAS_MODE:-auto}"       # auto|openblas|mkl|off

ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD_ROOT="$ROOT/llama-builds"
BUILD_DIR="$BUILD_ROOT/llama-cpp-$REF"
CURR_LINK="$ROOT/llama-current"

echo "Target ref: $REF"

# Resolve 'latest'
if [[ "$REF" == "latest" ]]; then
  echo "Resolving latest tag from GitHub..."
  REF="$(curl -fsSL https://api.github.com/repos/ggml-org/llama.cpp/releases/latest | sed -n 's/.*"tag_name": *"\(.*\)".*/\1/p')"
  echo "Latest tag: $REF"
  BUILD_DIR="$BUILD_ROOT/llama-cpp-$REF"
fi

rm -rf "$BUILD_DIR"
git clone --depth 1 --branch "$REF" https://github.com/ggml-org/llama.cpp "$BUILD_DIR"

mkdir -p "$BUILD_DIR/build"
pushd "$BUILD_DIR/build" >/dev/null

CMAKE_FLAGS=()
CMAKE_FLAGS+=(-DGGML_CUDA=ON)
CMAKE_FLAGS+=(-DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc")
CMAKE_FLAGS+=(-DCUDAToolkit_ROOT="$CUDA_HOME")
CMAKE_FLAGS+=(-DCMAKE_BUILD_TYPE=Release)
CMAKE_FLAGS+=(-DCMAKE_CUDA_ARCHITECTURES="$ARCH")
CMAKE_FLAGS+=(-DGGML_CUDA_F16=ON)
CMAKE_FLAGS+=(-DGGML_CUDA_FORCE_MMQ="$FORCE_MMQ")
CMAKE_FLAGS+=(-DLLAMA_CURL=ON)

# BLAS selection
case "$BLAS_MODE" in
  off)  CMAKE_FLAGS+=(-DGGML_BLAS=OFF -DGGML_BLAS_VENDOR=Generic) ;;
  mkl)  CMAKE_FLAGS+=(-DGGML_BLAS=ON  -DGGML_BLAS_VENDOR=Intel10_64lp) ;;
  openblas) CMAKE_FLAGS+=(-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS) ;;
  auto)
    if [[ -e /usr/lib/x86_64-linux-gnu/libmkl_rt.so || -e /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_rt.so ]]; then
      CMAKE_FLAGS+=(-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Intel10_64lp)
    else
      CMAKE_FLAGS+=(-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS)
    fi
    ;;
esac

# CUDA flags
CUDA_FLAGS="${CMAKE_CUDA_FLAGS:-}"
if [[ "${FAST_MATH}" == "1" ]]; then
  CUDA_FLAGS="${CUDA_FLAGS} --use_fast_math"
fi
if [[ -n "${CUDA_FLAGS// /}" ]]; then
  CMAKE_FLAGS+=(-DCMAKE_CUDA_FLAGS="${CUDA_FLAGS}")
fi

echo "Configuring..."
cmake .. "${CMAKE_FLAGS[@]}"

echo "Building..."
make -j"$(nproc)"

popd >/dev/null

# Link outputs
rm -f "$CURR_LINK"
ln -s "$BUILD_DIR" "$CURR_LINK"

mkdir -p "$ROOT/bin"
find "$BUILD_DIR/build/bin" -maxdepth 1 -type f -print0 | while IFS= read -r -d '' f; do
  ln -sf "$f" "$ROOT/bin/$(basename "$f")"
done

echo "Done. Binaries are under $ROOT/bin and $CURR_LINK/build/bin"
