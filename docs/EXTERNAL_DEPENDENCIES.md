# External Dependencies

This document lists all external tools, packages, and libraries used by C-ML.

## Required

### C11 Compiler

GCC 4.9+, Clang 3.5+, or MSVC 2015+ with C11 support.

### CMake 3.16+

Build system generator.

```bash
# Ubuntu/Debian
sudo apt-get install cmake

# Arch Linux
sudo pacman -S cmake

# macOS
brew install cmake
```

### Math Library (libm)

Standard math library, included with all C compilers.

### Dynamic Linker (libdl)

Used for runtime loading of optional libraries (BLAS, SLEEF). Included on Linux/macOS.

______________________________________________________________________

## Optional Libraries (Recommended)

### OpenBLAS / Intel MKL

Optimized BLAS library for matrix operations. C-ML dynamically loads BLAS at runtime, trying multiple libraries in order: MKL, OpenBLAS, ATLAS, reference CBLAS, Accelerate (macOS).

```bash
# Ubuntu/Debian
sudo apt-get install libopenblas-dev

# Arch Linux
sudo pacman -S openblas

# Fedora
sudo dnf install openblas-devel

# macOS (Accelerate framework is built-in)
brew install openblas  # optional, Accelerate is used by default
```

Override with environment variable: `CML_BLAS_LIB=/path/to/libblas.so`

### SLEEF

High-accuracy SIMD math library. Provides vectorized exp, log, sin, cos, etc. C-ML dynamically loads SLEEF at runtime if available.

```bash
# Ubuntu/Debian
sudo apt-get install libsleef-dev

# Arch Linux (build from source, not in repos)
git clone https://github.com/shibatch/sleef.git
cd sleef && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local && make -j$(nproc) && sudo make install

# Build from source (other distros)
git clone https://github.com/shibatch/sleef.git
cd sleef && mkdir build && cd build
cmake .. && make -j$(nproc) && sudo make install
```

### LLVM (for LLVM JIT backend)

LLVM provides JIT compilation for optimized kernel execution. The LLVM backend is auto-detected at CMake configure time.

```bash
# Ubuntu/Debian
sudo apt-get install llvm-18-dev

# Arch Linux
sudo pacman -S llvm

# Fedora
sudo dnf install llvm-devel

# macOS
brew install llvm@18
```

### curl / wget

Used by the dataset hub (`cml_dataset_load()`) to download datasets. At least one must be available in PATH.

```bash
# Usually pre-installed. If not:
sudo apt-get install curl   # or wget
```

______________________________________________________________________

## Optional Libraries (GPU)

### CUDA Toolkit (NVIDIA GPUs)

```bash
# See https://developer.nvidia.com/cuda-downloads
```

Libraries used: `libcudart.so`, `libcuda.so`

### ROCm (AMD GPUs)

```bash
# See https://rocm.docs.amd.com/
```

### Vulkan SDK

```bash
# Ubuntu/Debian
sudo apt-get install libvulkan-dev

# See https://vulkan.lunarg.com/sdk/home
```

______________________________________________________________________

## Summary Table

| Dependency   | Type    | Required | Used For         | Installation          |
| ------------ | ------- | -------- | ---------------- | --------------------- |
| C11 compiler | Tool    | Yes      | Compilation      | GCC/Clang/MSVC        |
| CMake 3.16+  | Tool    | Yes      | Build system     | Package manager       |
| libm         | Library | Yes      | Math functions   | Included              |
| libdl        | Library | Yes      | Dynamic loading  | Included              |
| OpenBLAS/MKL | Library | No       | Fast matrix ops  | Package manager       |
| SLEEF        | Library | No       | SIMD math        | Package manager       |
| LLVM 18+     | Library | No       | JIT compilation  | Package manager       |
| curl/wget    | Tool    | No       | Dataset download | Usually pre-installed |
| CUDA         | Library | No       | NVIDIA GPU       | CUDA Toolkit          |
| ROCm         | Library | No       | AMD GPU          | ROCm installer        |
| Vulkan       | Library | No       | GPU compute      | Vulkan SDK            |

______________________________________________________________________

## Verification

```bash
# Check compiler
gcc --version   # or clang --version

# Check CMake
cmake --version

# Check BLAS
ldconfig -p | grep -i openblas
ldconfig -p | grep -i mkl

# Check SLEEF
ldconfig -p | grep sleef

# Check LLVM
llvm-config --version

# Check download tools
which curl wget
```
