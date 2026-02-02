# CML Compilation Guide

This guide explains how to build and compile the C-ML library on different platforms and with different configurations.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installing MLIR (Required)](#installing-mlir-required)
3. [Building with Make](#building-with-make)
4. [Building with CMake](#building-with-cmake)
5. [Build Variants](#build-variants)
6. [Compiler Flags and Options](#compiler-flags-and-options)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required

- **C11 Compatible Compiler**: GCC 4.9+, Clang 3.5+, MSVC 2015+, or compatible
- **MLIR 18.x or later**: Required for tensor operations and JIT compilation
- **Make** or **CMake 3.16+**: For building

### Optional

- **CUDA Toolkit**: For NVIDIA GPU support
- **ROCm/HIP**: For AMD GPU support
- **Vulkan SDK**: For Vulkan backend support
- **Metal Framework**: Available on macOS for Apple GPU support
- **SLEEF Library**: For high-accuracy SIMD math (optional, but recommended)
- **Python3**: For visualization tools (optional)

## Installing MLIR (Required)

MLIR is the core execution engine for C-ML. It must be installed before building.

### Linux (Ubuntu/Debian)

```bash
# Add LLVM repository
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 18

# Install MLIR development packages
sudo apt-get update
sudo apt-get install libmlir-18-dev mlir-18-tools llvm-18-dev
```

### Linux (Arch Linux)

```bash
sudo pacman -S llvm mlir
```

### Linux (Fedora/RHEL)

```bash
sudo dnf install llvm-devel mlir-devel
```

### macOS (Homebrew)

```bash
# Install LLVM 18 with MLIR
brew install llvm@18

# Set environment variables (add to ~/.bashrc or ~/.zshrc)
export LLVM_DIR=$(brew --prefix llvm@18)
export LDFLAGS="-L$(brew --prefix llvm@18)/lib"
export CPPFLAGS="-I$(brew --prefix llvm@18)/include"
export LD_LIBRARY_PATH="$(brew --prefix llvm@18)/lib:$LD_LIBRARY_PATH"
```

### macOS (Build from Source)

If you need a specific MLIR version or prefer building from source:

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout llvmorg-18.0.0

mkdir build && cd build
cmake -G Ninja ../llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU"

ninja install
```

### Windows

#### Option 1: Using WSL2 (Recommended)

Follow the Ubuntu instructions inside WSL2.

#### Option 2: Native Windows

1. Download MLIR binaries from [LLVM Releases](https://github.com/llvm/llvm-project/releases)
2. Extract to a known location (e.g., `C:\LLVM`)
3. Set environment variables:
   ```cmd
   set MLIR_DIR=C:\LLVM
   set PATH=C:\LLVM\bin;%PATH%
   ```

#### Option 3: Build from Source

Use Visual Studio Developer Command Prompt:

```cmd
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout llvmorg-18.0.0

mkdir build && cd build
cmake -G "Visual Studio 16 2019" ..\llvm ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DLLVM_ENABLE_PROJECTS="mlir" ^
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU"

cmake --build . --config Release --parallel %NUMBER_OF_PROCESSORS%
cmake --install . --prefix C:\LLVM
```

## Building with Make

The Makefile is the easiest way to build C-ML. It automatically detects MLIR, CUDA, ROCm, and other optional dependencies.

### Basic Build

```bash
cd C-ML
make
```

This creates:
- `build/main` - Main executable
- `build/libcml.a` - Static library
- `build/libcml.so` - Shared library
- `build/examples/*` - Example programs

### Build Variants

```bash
# Debug build with sanitizers and debug symbols
make debug

# Release build with optimizations
make release

# Fast build with CPU-specific optimizations
make fast

# Static analysis build
make analyze

# Visualization frontend only
make frontend

# Start visualization server
make viz
```

### Build Targets

```bash
make all              # Build everything (default)
make lib              # Build libraries only
make examples         # Build example programs
make test             # Build and run tests
make clean            # Remove build artifacts
make help             # Show help
```

### Build Environment Variables

```bash
# Use a specific compiler
CC=gcc CXX=g++ make

CC=clang CXX=clang++ make

# Specify MLIR location (if not auto-detected)
MLIR_CONFIG=/path/to/mlir-config make

# Custom installation prefix
PREFIX=/opt/cml make install
```

### Installing Libraries

```bash
# Install to system directories (requires sudo)
sudo make install PREFIX=/usr/local

# Install to user directory
make install PREFIX=$HOME/.local
```

## Building with CMake

CMake provides more fine-grained control over the build process.

### Basic CMake Build

```bash
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### CMake Build Options

```bash
cd build

# Build with all features enabled
cmake -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_EXAMPLES=ON \
  -DBUILD_TESTS=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DENABLE_MLIR=ON \
  -DENABLE_CUDA=ON \
  -DENABLE_ROCM=ON \
  -DENABLE_VULKAN=ON \
  ..

# Build with specific MLIR path
cmake -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=/path/to/mlir/lib/cmake/mlir \
  ..

# Minimal build (CPU only)
cmake -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=OFF \
  -DENABLE_ROCM=OFF \
  -DENABLE_VULKAN=OFF \
  ..

make -j$(nproc)
```

### CMake Targets

```bash
# Build everything
make -j$(nproc)

# Build specific target
make autograd_example
make test_backends

# Run tests
make test
# or
ctest --output-on-failure

# Build documentation (if Doxygen installed)
cmake -DBUILD_DOCS=ON ..
make docs
```

## Build Variants

### Debug Build

Includes debug symbols, sanitizers, and additional warnings:

```bash
make debug
# or
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

Features:
- `-O0` optimization level
- Debug symbols (`-g3 -ggdb`)
- Address Sanitizer (detects memory errors)
- Undefined Behavior Sanitizer
- Leak Sanitizer
- All warnings enabled

### Release Build

Production build with optimizations:

```bash
make release
# or
cmake -DCMAKE_BUILD_TYPE=Release ..
```

Features:
- `-O3` optimization level
- No debug symbols (smaller binary)
- `NDEBUG` defined (assertions disabled)
- ~50-100x smaller binaries than debug

### Fast Build

Maximum performance build with CPU-specific optimizations:

```bash
make fast
```

Features:
- `-O3` optimization level
- CPU-specific flags: `-march=native -mtune=native`
- `-ffast-math` (less precise floating point, faster)
- Loop unrolling enabled
- Automatic SIMD detection (SSE, AVX, AVX2, AVX-512)

**Note**: Fast builds may be non-portable across different CPU architectures.

## Compiler Flags and Options

### SIMD Support Detection

The Makefile automatically detects and enables:
- SSE/SSE2 (always available on x86_64)
- AVX (if supported)
- AVX2 + FMA (if supported)
- AVX-512 (if supported)

Check what's available:

```bash
# View detected flags
gcc -march=native -Q --help=target | grep "march\|mtune"

# Test specific instruction set
echo | gcc -mavx512f -E - >/dev/null 2>&1 && echo "AVX-512: yes" || echo "AVX-512: no"
```

### SLEEF Math Library

SLEEF provides high-accuracy SIMD math functions:

```bash
# Install SLEEF
sudo apt-get install libsleef-dev  # Debian/Ubuntu

# Build automatically detects SLEEF
make
# Check output for: "SLEEF library found - high-accuracy SIMD math enabled"
```

### Disabling Optional Features

```bash
# Disable MLIR (not recommended - MLIR is required)
cmake -DENABLE_MLIR=OFF ..

# Disable GPU backends
cmake -DENABLE_CUDA=OFF -DENABLE_ROCM=OFF -DENABLE_VULKAN=OFF ..

# Disable shared library
cmake -DBUILD_SHARED_LIBS=OFF ..

# Disable examples
cmake -DBUILD_EXAMPLES=OFF ..

# Disable tests
cmake -BUILD_TESTS=OFF ..
```

## Compiler Compatibility

### GCC

```bash
# GCC 11+
CC=gcc-11 CXX=g++-11 make

# Check version
gcc --version
# Should be GCC 4.9 or later
```

### Clang

```bash
# Clang 15+
CC=clang CXX=clang++ make

# Check version
clang --version
```

### MSVC (Windows)

```cmd
# Use Visual Studio Developer Command Prompt
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release
```

## Cross-Compilation

### Build for ARM64 (on x86_64)

```bash
# Install cross-compilation toolchain
sudo apt-get install g++-aarch64-linux-gnu

# Configure CMake for cross-compilation
cmake -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
      -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
      -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
      -DCMAKE_SYSTEM_NAME=Linux ..

make -j$(nproc)
```

### Build for Raspberry Pi (ARM32)

```bash
# Install toolchain
sudo apt-get install g++-arm-linux-gnueabihf

# Configure CMake
cmake -DCMAKE_C_COMPILER=arm-linux-gnueabihf-gcc \
      -DCMAKE_CXX_COMPILER=arm-linux-gnueabihf-g++ \
      -DCMAKE_SYSTEM_PROCESSOR=arm \
      -DCMAKE_SYSTEM_NAME=Linux ..

make -j$(nproc)
```

## Troubleshooting

### MLIR Not Found

```
Error: MLIR is required to build C-ML
```

Solution:
```bash
# Verify MLIR is installed
which mlir-config
llvm-config --version

# If not in PATH, specify path explicitly
MLIR_CONFIG=/usr/lib/llvm-18/bin/llvm-config make

# Or for CMake
cmake -DMLIR_DIR=/usr/lib/llvm-18/lib/cmake/mlir ..
```

### Linker Errors

```
error: undefined reference to `MLIR...'
```

Solution:
```bash
# Ensure MLIR libraries are in library path
export LD_LIBRARY_PATH=/usr/lib/llvm-18/lib:$LD_LIBRARY_PATH

# Or rebuild with correct MLIR location
make clean
cmake -DMLIR_DIR=/correct/path/to/mlir ..
make
```

### Out of Memory During Build

```bash
# Reduce parallel build jobs
make -j2   # Instead of -j$(nproc)

# Or use CMake
cmake --build . --parallel 2
```

### C++ Compiler Mismatch

```
error: incompatible C++ version
```

Solution:
```bash
# Use compatible C++ compiler
CXX=g++-11 make

# Check C++ standard support
g++ -std=c++20 --version
```

### macOS Specific Issues

```bash
# If using Homebrew LLVM, ensure proper linking
export LDFLAGS="-L$(brew --prefix llvm@18)/lib"
export CPPFLAGS="-I$(brew --prefix llvm@18)/include"
export LD_LIBRARY_PATH="$(brew --prefix llvm@18)/lib:$LD_LIBRARY_PATH"

# Then build
make clean
make
```

### Windows Specific Issues

If CMake fails to find MLIR on Windows:

```cmd
# Set environment variables
set MLIR_DIR=C:\LLVM\lib\cmake\mlir
set LLVM_DIR=C:\LLVM\lib\cmake\llvm

# Then configure CMake
cmake -G "Visual Studio 17 2022" -A x64 ^
  -DMLIR_DIR=C:\LLVM\lib\cmake\mlir ^
  -DLLVM_DIR=C:\LLVM\lib\cmake\llvm ..

cmake --build . --config Release
```

## Build Output

After a successful build, you'll have:

```
build/
├── main                      # Main executable
├── lib/
│   ├── libcml.a             # Static library
│   └── libcml.so            # Shared library
├── examples/
│   ├── autograd_example
│   ├── training_loop_example
│   ├── test_example
│   ├── mnist_example
│   └── ...
└── test_*/                  # Test executables
```

## Verifying the Build

```bash
# Run a simple example
./build/examples/autograd_example

# Run tests (if built)
make test

# Or with CMake
cd build && ctest --output-on-failure
```

## Next Steps

After building successfully, see:
- [RUNNING.md](RUNNING.md) - How to run programs
- [QUICK_START.md](QUICK_START.md) - Quick start guide
- [API_GUIDE.md](API_GUIDE.md) - API reference
