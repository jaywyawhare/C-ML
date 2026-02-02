# Building C-ML with MLIR Support

This guide explains how to build C-ML with MLIR/LLVM integration for JIT compilation.

## Prerequisites

- CMake >= 3.20
- Ninja or Make
- GCC/Clang with C11 support
- Python 3.8+ (for LLVM build)
- 20GB+ free disk space
- 8GB+ RAM

## Step 1: Build LLVM/MLIR from Source

### Clone LLVM Project

```bash
cd /tmp
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout release/18.x  # Or latest stable
```

### Build MLIR

```bash
mkdir build && cd build

cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
    -D CMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_INSTALL_PREFIX=/usr/local/mlir \
    -DLLVM_INSTALL_UTILS=ON

ninja
sudo ninja install
```

This will take 30-60 minutes depending on your hardware.

### Verify Installation

```bash
ls /usr/local/mlir/lib/cmake/mlir  # Should exist
```

## Step 2: Build C-ML with MLIR

### Configure with MLIR

```bash
cd C-ML
mkdir build && cd build

cmake .. \
    -DCML_ENABLE_MLIR=ON \
    -DMLIR_DIR=/usr/local/mlir/lib/cmake/mlir \
    -DLLVM_DIR=/usr/local/mlir/lib/cmake/llvm
```

### Build

```bash
make -j$(nproc)
```

### Verify MLIR is Enabled

```bash
./build/test_mlir_integration
# Should show: "MLIR available: YES"
```

## Step 3: Test JIT Compilation

```c
#include "cml.h"

int main() {
    // Enable JIT mode
    cml_enable_jit(true);

    // Operations now use MLIR JIT
    Tensor* a = tensor_randn((int[]){100, 100}, 2, NULL);
    Tensor* b = tensor_randn((int[]){100, 100}, 2, NULL);
    Tensor* c = tensor_add(a, b);  // JIT compiled!

    printf("JIT compilation successful!\n");
    return 0;
}
```

## Alternative: Use Pre-built MLIR

If available on your system:

```bash
# Ubuntu/Debian
sudo apt install llvm-18 llvm-18-dev mlir-18 mlir-18-dev

# macOS with Homebrew
brew install llvm@18

# Then build C-ML
cmake .. -DCML_ENABLE_MLIR=ON -DMLIR_DIR=/usr/lib/llvm-18/lib/cmake/mlir
```

## Build Without MLIR (Default)

If you don't need JIT:

```bash
cmake .. -DCML_ENABLE_MLIR=OFF
make
```

C-ML will work normally using interpreted mode.

## Troubleshooting

### MLIR Not Found

```
Error: Could not find MLIR
Solution: Set MLIR_DIR to the cmake directory
cmake .. -DMLIR_DIR=/path/to/mlir/lib/cmake/mlir
```

### Linker Errors

```
undefined reference to mlir*
Solution: Ensure LLVM_DIR is also set
cmake .. -DMLIR_DIR=... -DLLVM_DIR=/usr/local/mlir/lib/cmake/llvm
```

### Out of Memory During Build

```
Solution: Reduce parallel jobs
make -j2  # Instead of -j$(nproc)
```

## Docker Image (Coming Soon)

We'll provide a Docker image with MLIR pre-installed:

```bash
docker pull cml/cml-mlir:latest
docker run -it cml/cml-mlir bash
```

## Performance Expectations

With MLIR JIT enabled:

- **Elementwise ops**: 5-10x faster
- **Matrix ops**: 3-5x faster
- **First run**: Slower (compilation overhead)
- **Subsequent runs**: Fast (cached)

## Next Steps

- Read the [MLIR Integration Plan](./MLIR_INTEGRATION_PLAN.md)
- See [Phase 1 Implementation Guide](./mlir/phase1_implementation.md)
- Join the discussion on GitHub Issues
