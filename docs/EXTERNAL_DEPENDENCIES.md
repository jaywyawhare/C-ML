# External CLI Tools and Packages Used by C-ML

This document lists all external CLI tools, packages, and dependencies used by C-ML for lowering, optimization, code generation, and execution.

## Required CLI Tools

### 1. **mlir-opt** (Required for MLIR backend)

**Purpose:** MLIR optimization and lowering tool
**Location:** Part of LLVM/MLIR distribution
**Usage:** Lowering MLIR dialects and applying optimization passes

**Used in:**

- `src/ops/ir/mlir/mlir_codegen.c`

**Commands Used:**

```bash
# CPU lowering pipeline
mlir-opt \
  --convert-linalg-to-loops \
  --convert-scf-to-cf \
  --lower-affine \
  --convert-arith-to-llvm \
  --convert-math-to-llvm \
  --convert-func-to-llvm \
  --convert-cf-to-llvm \
  --finalize-memref-to-llvm \
  --reconcile-unrealized-casts \
  input.mlir

# GPU parallel lowering pipeline
mlir-opt \
  --convert-linalg-to-parallel-loops \
  --scf-parallel-loop-tiling="parallel-loop-tile-sizes=16,16,1" \
  --gpu-map-parallel-loops \
  --convert-parallel-loops-to-gpu \
  --gpu-kernel-outlining \
  --lower-affine \
  --convert-scf-to-cf \
  --convert-gpu-to-nvvm \
  input.mlir -o output.mlir
```

**Installation:**

- **Linux (Ubuntu/Debian):** `sudo apt-get install mlir-18-tools`
- **Linux (Arch):** `sudo pacman -S mlir` (includes mlir-opt)
- **Linux (Fedora):** `sudo dnf install mlir-devel` (includes mlir-opt)
- **macOS:** `brew install llvm@18` (includes mlir-opt)
- **Windows:** Install LLVM from [llvm.org](https://llvm.org/builds/)

**Path:** Usually in `/usr/bin/mlir-opt` or `$LLVM_DIR/bin/mlir-opt`

______________________________________________________________________

### 2. **mlir-translate** (Required for MLIR backend)

**Purpose:** Translate MLIR to other IR formats (LLVM IR, SPIR-V, etc.)
**Location:** Part of LLVM/MLIR distribution
**Usage:** Converting MLIR dialect to LLVM IR for compilation

**Used in:**

- `src/ops/ir/mlir/mlir_codegen.c`

**Commands Used:**

```bash
# Convert MLIR to LLVM IR
mlir-translate --mlir-to-llvmir input.mlir > output.ll
```

**Installation:** Same as `mlir-opt` (comes with MLIR tools)

**Path:** Usually in `/usr/bin/mlir-translate` or `$LLVM_DIR/bin/mlir-translate`

______________________________________________________________________

### 3. **llc** (Required for AOT compilation and GPU codegen)

**Purpose:** LLVM static compiler - compiles LLVM IR to object files or assembly
**Location:** Part of LLVM distribution
**Usage:**

- AOT compilation: LLVM IR → object files
- GPU codegen: LLVM IR → PTX (CUDA)

**Used in:**

- `src/ops/ir/mlir/mlir_codegen.c` (GPU PTX generation)
- `src/ops/ir/mlir/mlir_execute.c` (AOT object file generation)

**Commands Used:**

```bash
# Compile LLVM IR to object file (AOT)
llc -filetype=obj -O3 input.ll -o output.o

# Compile LLVM IR to PTX (CUDA GPU)
llc -march=nvptx64 -mcpu=sm_75 -O3 input.ll -o output.ptx
```

**Installation:** Same as `mlir-opt` (comes with LLVM)

**Path:** Usually in `/usr/bin/llc` or `$LLVM_DIR/bin/llc`

______________________________________________________________________

## Optional CLI Tools

### 4. **sed** (Standard Unix tool)

**Purpose:** Stream editor for text processing
**Location:** Pre-installed on most Unix-like systems
**Usage:** Extracting and transforming GPU kernel code from MLIR output

**Used in:**

- `src/ops/ir/mlir/mlir_codegen.c` (GPU kernel extraction)

**Commands Used:**

```bash
# Extract GPU kernel module from MLIR output
sed -n '/^  gpu\.module @main_kernel_0/,/^  }/p' input.mlir | \
  sed '1s/.*/module attributes {...} {/' | \
  sed 's/gpu\.kernel, //' | \
  sed 's/gpu\.known_block_size = array<[^>]*>, //' > output.mlir

# Add target triple to LLVM IR
sed '1i\
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"\
target triple = "nvptx64-nvidia-cuda"' input.ll > output.ll
```

**Installation:**

- **Linux/macOS:** Pre-installed
- **Windows:** Available via Git Bash, WSL, or MSYS2

______________________________________________________________________

### 5. **spirv-as** (Optional - for SPIR-V generation)

**Purpose:** SPIR-V assembler - converts SPIR-V assembly to binary
**Location:** Part of SPIRV-Tools (Vulkan SDK)
**Usage:** Assembling SPIR-V from text format for Vulkan/Metal/WGSL codegen

**Used in:**

- `src/ops/ir/mlir/mlir_codegen.c` (SPIR-V, Metal, WGSL generation)

**Commands Used:**

```bash
# Assemble SPIR-V from assembly text
spirv-as input.spvasm -o output.spv
```

**Installation:**

- **Linux (Ubuntu/Debian):** `sudo apt-get install spirv-tools`
- **Linux (Arch):** `sudo pacman -S spirv-tools`
- **Linux (Fedora):** `sudo dnf install spirv-tools`
- **macOS:** `brew install spirv-tools` or included in Vulkan SDK
- **Windows:** Included in Vulkan SDK

**Path:** Usually in `/usr/bin/spirv-as` or `$VULKAN_SDK/bin/spirv-as`

______________________________________________________________________

### 6. **spirv-cross** (Optional - for Metal codegen)

**Purpose:** SPIR-V cross-compiler - converts SPIR-V to other shading languages
**Location:** Standalone tool from Khronos
**Usage:** Converting SPIR-V to Metal Shading Language (MSL)

**Used in:**

- `src/ops/ir/mlir/mlir_codegen.c` (Metal generation)

**Commands Used:**

```bash
# Convert SPIR-V to Metal
spirv-cross --msl input.spv --output output.metal
```

**Installation:**

- **Linux (Ubuntu/Debian):** `sudo apt-get install spirv-cross`
- **Linux (Arch):** `sudo pacman -S spirv-cross` (AUR: `spirv-cross-git`)
- **Linux (Fedora):** `sudo dnf install spirv-cross` (may need to build from source)
- **macOS:** `brew install spirv-cross`
- **Windows:** Download from [Khronos GitHub](https://github.com/KhronosGroup/SPIRV-Cross/releases)

**Path:** Usually in `/usr/bin/spirv-cross` or `$VULKAN_SDK/bin/spirv-cross`

**Note:** Can also be built from source: https://github.com/KhronosGroup/SPIRV-Cross

______________________________________________________________________

### 7. **llvm-as** (Optional - for SPIR-V via LLVM path)

**Purpose:** LLVM assembler - converts LLVM IR text to bitcode
**Location:** Part of LLVM distribution
**Usage:** Converting LLVM IR to bitcode before SPIR-V conversion

**Used in:**

- `src/ops/ir/mlir/mlir_codegen.c` (SPIR-V generation via LLVM path)

**Commands Used:**

```bash
# Convert LLVM IR to bitcode
llvm-as input.ll -o output.bc
```

**Installation:** Same as `llc` (comes with LLVM)

**Path:** Usually in `/usr/bin/llvm-as` or `$LLVM_DIR/bin/llvm-as`

______________________________________________________________________

### 8. **llvm-spirv** (Optional - for SPIR-V via LLVM path)

**Purpose:** LLVM to SPIR-V translator
**Location:** SPIRV-LLVM-Translator project
**Usage:** Converting LLVM bitcode to SPIR-V binary

**Used in:**

- `src/ops/ir/mlir/mlir_codegen.c` (SPIR-V generation fallback path)

**Commands Used:**

```bash
# Convert LLVM bitcode to SPIR-V
llvm-spirv input.bc -o output.spv
```

**Installation:**

- **Linux (Ubuntu/Debian):** `sudo apt-get install llvm-spirv` (if available)
- **Linux (Arch):** `sudo pacman -S llvm-spirv` (AUR: `llvm-spirv-git`)
- **Linux (Fedora):** May need to build from source
- **macOS:** `brew install llvm-spirv` (if available) or build from source
- **Windows:** Build from source

**Build from source:**

```bash
git clone https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git
cd SPIRV-LLVM-Translator
mkdir build && cd build
cmake .. -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm
make -j$(nproc)
sudo make install
```

**Path:** Usually in `/usr/bin/llvm-spirv` or custom installation path

**Note:** This is optional - C-ML has fallback mechanisms if not available

______________________________________________________________________

### 9. **naga** (Optional - for WGSL codegen)

**Purpose:** WGSL shader compiler from wgpu project
**Location:** Rust-based tool from wgpu
**Usage:** Converting SPIR-V to WebGPU Shading Language (WGSL)

**Used in:**

- `src/ops/ir/mlir/mlir_codegen.c` (WGSL generation)

**Commands Used:**

```bash
# Convert SPIR-V to WGSL
naga input.spv output.wgsl
```

**Installation:**

- **All platforms:** Install via Rust Cargo:
  ```bash
  cargo install naga-cli
  ```
- **Linux/macOS:** May be available in package managers (check AUR for Arch)
- **Windows:** Install Rust from [rustup.rs](https://rustup.rs/), then `cargo install naga-cli`

**Path:** Usually in `~/.cargo/bin/naga` (add to PATH)

**Note:** Alternative to `tint` - C-ML tries both

______________________________________________________________________

### 10. **tint** (Optional - for WGSL codegen)

**Purpose:** Tint shader compiler from Google Dawn project
**Location:** Google Dawn WebGPU implementation
**Usage:** Alternative converter for SPIR-V to WGSL

**Used in:**

- `src/ops/ir/mlir/mlir_codegen.c` (WGSL generation fallback)

**Commands Used:**

```bash
# Convert SPIR-V to WGSL
tint --format wgsl input.spv -o output.wgsl
```

**Installation:**

- **All platforms:** Build from source:
  ```bash
  git clone https://dawn.googlesource.com/tint
  cd tint
  # Follow build instructions in README
  ```
- **Linux/macOS:** May be available via package managers
- **Windows:** Build from source or use pre-built binaries if available

**Path:** Custom installation path

**Note:** Alternative to `naga` - C-ML tries `naga` first, then `tint`

______________________________________________________________________

### 11. **grep** (Standard Unix tool)

**Purpose:** Pattern matching in text files
**Location:** Pre-installed on most Unix-like systems
**Usage:** Analyzing MLIR modules to detect operation types

**Used in:**

- `src/ops/ir/mlir/mlir_codegen.c` (detecting matmul operations, extracting shapes)

**Commands Used:**

```bash
# Check for matmul operations
grep -c 'linalg.matmul' input.mlir

# Extract tensor dimensions
grep -oE 'memref<[0-9]+x[0-9]+xf32>' input.mlir | head -1
```

**Installation:**

- **Linux/macOS:** Pre-installed
- **Windows:** Available via Git Bash, WSL, or MSYS2

______________________________________________________________________

### 5. **python3** (Optional - for visualization)

**Purpose:** Python interpreter for visualization server
**Location:** System Python installation
**Usage:** Running the visualization server (`scripts/viz.py`)

**Used in:**

- `src/cml.c` (launches visualization server)

**Commands Used:**

```bash
python3 scripts/viz.py <executable_path> [args...]
```

**Installation:**

- **Linux (Ubuntu/Debian):** `sudo apt-get install python3`
- **Linux (Arch):** `sudo pacman -S python`
- **Linux (Fedora):** `sudo dnf install python3`
- **macOS:** `brew install python3` or pre-installed
- **Windows:** Download from [python.org](https://www.python.org/downloads/)

**Note:** Only required if using the `VIZ=1` environment variable feature

______________________________________________________________________

## External Libraries (Runtime Dependencies)

### 6. **MLIR/LLVM Libraries** (Required)

**Purpose:** MLIR C API and execution engine
**Location:** LLVM/MLIR installation
**Libraries Used:**

- `MLIR` - Core MLIR library
- `MLIRExecutionEngineShared` - JIT execution engine
- `LLVM` - Core LLVM library
- `MLIRCAPIRegisterEverything` - MLIR C API registration
- `MLIRCAPIIR` - MLIR C API IR
- `MLIRCAPILinalg` - MLIR C API Linalg dialect
- `MLIRCAPIMath` - MLIR C API Math dialect
- `MLIRCAPITransforms` - MLIR C API transforms
- `MLIRCAPIExecutionEngine` - MLIR C API execution engine
- `MLIRCAPIConversion` - MLIR C API conversion
- `MLIRCAPILLVM` - MLIR C API LLVM dialect
- `MLIRCAPISCF` - MLIR C API SCF dialect
- `MLIRCAPIControlFlow` - MLIR C API control flow
- `MLIRCAPIInterfaces` - MLIR C API interfaces
- `MLIRCAPIDebug` - MLIR C API debug

**Installation:** See [MLIR Installation Guide](mlir/BUILDING_WITH_MLIR.md)

______________________________________________________________________

### 7. **CUDA Libraries** (Optional - for NVIDIA GPU)

**Purpose:** CUDA runtime and driver APIs
**Location:** CUDA Toolkit installation
**Libraries Used:**

- `libcudart.so` - CUDA runtime library
- `libcuda.so` - CUDA driver library

**Installation:** See [CUDA Installation](https://developer.nvidia.com/cuda-downloads)

______________________________________________________________________

### 8. **Vulkan Libraries** (Optional - for Vulkan backend)

**Purpose:** Vulkan API for GPU acceleration
**Location:** Vulkan SDK installation
**Libraries Used:**

- `libvulkan.so` - Vulkan loader library

**Installation:** See [Vulkan SDK](https://vulkan.lunarg.com/sdk/home)

______________________________________________________________________

### 9. **OpenBLAS/MKL** (Optional - recommended)

**Purpose:** Optimized BLAS library for matrix operations
**Location:** System installation
**Libraries Used:**

- `libopenblas.so` - OpenBLAS library
- `libmkl_rt.so` - Intel MKL library

**Installation:**

- **Linux (Ubuntu/Debian):** `sudo apt-get install libopenblas-dev`
- **Linux (Arch):** `sudo pacman -S openblas`
- **Linux (Fedora):** `sudo dnf install openblas-devel`
- **macOS:** `brew install openblas`

______________________________________________________________________

### 10. **SLEEF** (Optional - recommended for SIMD math)

**Purpose:** High-accuracy SIMD math library
**Location:** Built from source or system installation
**Libraries Used:**

- `libsleef.so` - SLEEF library

**Installation:** See [SLEEF Repository](https://github.com/shibatch/sleef) or use setup script

______________________________________________________________________

## Summary Table

| Tool/Package       | Type    | Required    | Used For                         | Installation            |
| ------------------ | ------- | ----------- | -------------------------------- | ----------------------- |
| **mlir-opt**       | CLI     | ✅ Yes      | MLIR lowering & optimization     | LLVM/MLIR package       |
| **mlir-translate** | CLI     | ✅ Yes      | MLIR → LLVM IR conversion        | LLVM/MLIR package       |
| **llc**            | CLI     | ✅ Yes      | LLVM IR → object/PTX compilation | LLVM package            |
| **sed**            | CLI     | ⚠️ Optional | Text processing (GPU kernels)    | Pre-installed           |
| **grep**           | CLI     | ⚠️ Optional | Pattern matching (MLIR analysis) | Pre-installed           |
| **python3**        | CLI     | ⚠️ Optional | Visualization server             | System package          |
| **spirv-as**       | CLI     | ⚠️ Optional | SPIR-V assembly                  | SPIRV-Tools/Vulkan SDK  |
| **spirv-cross**    | CLI     | ⚠️ Optional | SPIR-V to Metal converter        | Khronos/package manager |
| **llvm-as**        | CLI     | ⚠️ Optional | LLVM assembler                   | LLVM package            |
| **llvm-spirv**     | CLI     | ⚠️ Optional | LLVM to SPIR-V translator        | Build from source       |
| **naga**           | CLI     | ⚠️ Optional | SPIR-V to WGSL converter         | Rust Cargo              |
| **tint**           | CLI     | ⚠️ Optional | SPIR-V to WGSL converter (alt)   | Build from source       |
| **MLIR/LLVM libs** | Library | ✅ Yes      | Runtime MLIR execution           | LLVM/MLIR package       |
| **CUDA libs**      | Library | ⚠️ Optional | NVIDIA GPU support               | CUDA Toolkit            |
| **Vulkan libs**    | Library | ⚠️ Optional | Vulkan GPU support               | Vulkan SDK              |
| **OpenBLAS/MKL**   | Library | ⚠️ Optional | Matrix operations                | System package          |
| **SLEEF**          | Library | ⚠️ Optional | SIMD math                        | Build from source       |

______________________________________________________________________

## Verification

Check if tools are available:

```bash
# Check MLIR tools
which mlir-opt
which mlir-translate
which llc

# Check versions
mlir-opt --version
mlir-translate --version
llc --version

# Check Python
python3 --version

# Check libraries
ldconfig -p | grep mlir
ldconfig -p | grep cuda
ldconfig -p | grep vulkan
ldconfig -p | grep openblas
```

______________________________________________________________________

## Troubleshooting

### mlir-opt not found

```bash
# Find MLIR installation
find /usr -name "mlir-opt" 2>/dev/null
find /opt -name "mlir-opt" 2>/dev/null

# Add to PATH
export PATH=/path/to/llvm/bin:$PATH
```

### llc not found

```bash
# llc is part of LLVM, same installation as mlir-opt
which llc
# If not found, ensure LLVM bin directory is in PATH
```

### GPU codegen fails

```bash
# Check if mlir-opt has GPU dialect support
mlir-opt --help | grep gpu

# Check CUDA installation
nvcc --version
nvidia-smi
```

### AOT compilation fails

```bash
# Verify llc can generate object files
echo "define void @test() { ret void }" > test.ll
llc -filetype=obj test.ll -o test.o
file test.o  # Should show object file
```

______________________________________________________________________

## Notes

1. **All CLI tools are invoked via `system()` or `popen()`** - they must be in PATH
1. **Temporary files are created in `/tmp/`** - ensure write permissions
1. **Error output is redirected to `/dev/null`** - check logs for detailed errors
1. **Fallback mechanisms exist** - if CLI tools fail, C-ML falls back to interpreter mode
1. **GPU codegen requires specific MLIR passes** - ensure MLIR is built with GPU dialect support

______________________________________________________________________

## Future Improvements

- Replace `system()` calls with direct MLIR C API calls (no CLI dependency)
- Use MLIR's in-process lowering instead of external tools
- Cache compiled kernels to avoid repeated CLI invocations
- Add better error reporting when CLI tools are missing
