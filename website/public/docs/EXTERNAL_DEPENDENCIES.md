# External Dependencies

## Required

**C11 Compiler** -- GCC 4.9+, Clang 3.5+, or MSVC 2015+.

**CMake 3.16+**

```bash
sudo apt-get install cmake          # Ubuntu/Debian
sudo pacman -S cmake                # Arch
brew install cmake                  # macOS
```

**libm** -- standard math library, included with all C compilers.

**libdl** -- runtime loading of optional libraries. Included on Linux/macOS.

---

## Optional (Recommended)

### OpenBLAS / Intel MKL

Optimized BLAS for matrix ops. C-ML dynamically loads BLAS at runtime, trying: MKL, OpenBLAS, ATLAS, CBLAS, Accelerate (macOS).

```bash
sudo apt-get install libopenblas-dev   # Ubuntu/Debian
sudo pacman -S openblas                # Arch
sudo dnf install openblas-devel        # Fedora
```

macOS uses the built-in Accelerate framework by default.

Override: `CML_BLAS_LIB=/path/to/libblas.so`

### SLEEF

Vectorized SIMD math (exp, log, sin, cos, etc.). Dynamically loaded at runtime if available.

```bash
sudo apt-get install libsleef-dev      # Ubuntu/Debian
sudo dnf install sleef-devel           # Fedora
```

Arch Linux -- not in the official repos. Build from source:

```bash
git clone https://github.com/shibatch/sleef.git
cd sleef && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc) && sudo make install
```

### LLVM (JIT backend)

Auto-detected at CMake configure time.

```bash
sudo apt-get install llvm-18-dev       # Ubuntu/Debian
sudo pacman -S llvm                    # Arch
sudo dnf install llvm-devel            # Fedora
brew install llvm@18                   # macOS
```

### curl / wget

Used by the dataset hub to download datasets. At least one must be in PATH.

---

## Optional (GPU)

### CUDA (NVIDIA)

See https://developer.nvidia.com/cuda-downloads

Uses: `libcudart.so`, `libcuda.so`

### ROCm (AMD)

See https://rocm.docs.amd.com/

### Vulkan

```bash
sudo apt-get install libvulkan-dev     # Ubuntu/Debian
```

See https://vulkan.lunarg.com/sdk/home

---

## Summary

| Dependency   | Required | Used For        | Install               |
| ------------ | -------- | --------------- | --------------------- |
| C11 compiler | Yes      | Compilation     | GCC/Clang/MSVC        |
| CMake 3.16+  | Yes      | Build system    | Package manager       |
| libm         | Yes      | Math functions  | Included              |
| libdl        | Yes      | Dynamic loading | Included              |
| OpenBLAS/MKL | No       | Fast matrix ops | Package manager       |
| SLEEF        | No       | SIMD math       | Source build on Arch  |
| LLVM 18+     | No       | JIT compilation | Package manager       |
| curl/wget    | No       | Dataset download| Usually pre-installed |
| CUDA         | No       | NVIDIA GPU      | CUDA Toolkit          |
| ROCm         | No       | AMD GPU         | ROCm installer        |
| Vulkan       | No       | GPU compute     | Vulkan SDK            |

## Verify

```bash
gcc --version
cmake --version
ldconfig -p | grep -i openblas
ldconfig -p | grep sleef
llvm-config --version
which curl wget
```
