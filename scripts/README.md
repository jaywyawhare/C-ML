# C-ML Setup Scripts

This directory contains cross-platform setup scripts to install all dependencies needed for C-ML development.

## Available Scripts

### Linux/macOS: `setup_env.sh`

Interactive bash script that installs:
- **MLIR/LLVM** (required)
- **CUDA Toolkit** (optional, for NVIDIA GPUs)
- **Vulkan SDK** (optional)
- **SLEEF** (optional, recommended for SIMD math)
- **OpenBLAS** (optional, recommended for matrix operations)
- **ROCm** (optional, for AMD GPUs on Linux)
- **SPIRV-Tools** (optional, for SPIR-V generation)
- **SPIRV-Cross** (optional, for Metal codegen)
- **Naga** (optional, for WGSL codegen)
- **Build tools** (GCC/Clang, Make, CMake, Ninja, Git)
- **Node.js and npm** (for frontend development)
- **Python3** (for visualization tools and scripts)
- **React** (installed via npm in viz-ui directory)

**Usage:**
```bash
cd scripts
chmod +x setup_env.sh
./setup_env.sh
```

The script will:
1. Detect your operating system (Ubuntu/Debian, Arch, Fedora, macOS)
2. Detect your GPU (NVIDIA, AMD, Apple)
3. Prompt you to choose what to install
4. Install dependencies using the appropriate package manager
5. Generate an environment setup script at `~/.cml_env.sh`

**After installation:**
```bash
# Source the environment script
source ~/.cml_env.sh

# Or add to your ~/.bashrc or ~/.zshrc
echo "source ~/.cml_env.sh" >> ~/.bashrc
```

### Windows: `setup_env.ps1` (PowerShell)

PowerShell script with similar functionality for Windows.

**Usage:**
```powershell
# Run PowerShell as Administrator
cd scripts
.\setup_env.ps1 -All          # Install everything
.\setup_env.ps1 -MLIR         # Install MLIR only
.\setup_env.ps1 -MLIR -CUDA   # Install MLIR + CUDA
.\setup_env.ps1 -MLIR -Vulkan # Install MLIR + Vulkan
```

**Options:**
- `-All`: Install everything
- `-MLIR`: Install MLIR/LLVM only
- `-CUDA`: Install CUDA Toolkit
- `-Vulkan`: Install Vulkan SDK
- `-SLEEF`: Install SLEEF (requires manual build on Windows)
- `-BLAS`: Install OpenBLAS

### Windows: `setup_env.bat` (Batch)

Simple batch script for Windows users who prefer CMD.

**Usage:**
```cmd
REM Run Command Prompt as Administrator
cd scripts
setup_env.bat
```

Follow the interactive prompts to select what to install.

## What Gets Installed

### Required Dependencies

1. **Build Tools**
   - **GCC/Clang**: C compiler (gcc or clang)
   - **Make**: Build automation tool
   - **CMake**: Cross-platform build system
   - **Git**: Version control
   - Installed via `build-essential` (Ubuntu), `base-devel` (Arch), or Xcode Command Line Tools (macOS)

2. **MLIR/LLVM 18.x**
   - Core execution engine for C-ML
   - Required for tensor operations and JIT compilation
   - Installed via system package manager or Homebrew

3. **Python3**
   - Required for visualization tools and scripts
   - Installed via system package manager

4. **Node.js and npm** (for frontend)
   - Required for building the visualization UI
   - Installed via system package manager or NodeSource repository

### Optional Dependencies

2. **CUDA Toolkit**
   - For NVIDIA GPU acceleration
   - Automatically detected if NVIDIA GPU is present
   - Installs CUDA 12.x by default

3. **Vulkan SDK**
   - For Vulkan backend support
   - Cross-platform GPU acceleration
   - Installs latest stable version

4. **SLEEF**
   - High-accuracy SIMD math library
   - Recommended for performance
   - Built from source and installed to `~/.local`

5. **OpenBLAS**
   - Optimized BLAS library
   - 2-10x faster matrix operations than reference BLAS
   - Installed via system package manager

6. **ROCm**
   - For AMD GPU support (Linux only)
   - Automatically detected if AMD GPU is present

7. **SPIRV-Tools** (spirv-as)
   - SPIR-V assembler for Vulkan/Metal/WGSL codegen
   - Part of Vulkan SDK or available separately

8. **SPIRV-Cross**
   - Converts SPIR-V to Metal Shading Language
   - Required for Metal backend code generation

9. **Naga** (optional)
   - Rust-based WGSL converter
   - Converts SPIR-V to WebGPU Shading Language
   - Installed via Cargo: `cargo install naga-cli`

10. **llvm-spirv** (optional)
    - LLVM to SPIR-V translator
    - Used for SPIR-V generation via LLVM IR path
    - May need to be built from source

11. **React** (for frontend)
    - JavaScript library for building user interfaces
    - Installed automatically via npm in viz-ui directory
    - Required for visualization frontend

## Platform-Specific Notes

### Linux (Ubuntu/Debian)
- Uses `apt-get` package manager
- Adds LLVM repository automatically
- Supports automatic CUDA installation

### Linux (Arch)
- Uses `pacman` package manager
- MLIR available in official repositories
- CUDA available via AUR or manual installation

### Linux (Fedora/RHEL)
- Uses `dnf` package manager
- MLIR available in official repositories
- CUDA requires manual repository setup

### macOS
- Requires Homebrew (`brew`)
- Installs LLVM 18 via Homebrew
- Metal framework available by default (no installation needed)
- CUDA not supported (use Metal instead)

### Windows
- Requires Chocolatey package manager (installed automatically)
- Visual Studio with C++ tools recommended
- SLEEF requires manual build from source
- WSL2 recommended for easier development

## Environment Variables

After installation, the following environment variables may be set:

### MLIR/LLVM
```bash
export MLIR_DIR=/path/to/mlir/lib/cmake/mlir
export LLVM_DIR=/path/to/llvm/lib/cmake/llvm
export PATH=/path/to/llvm/bin:$PATH
```

### CUDA
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Vulkan
```bash
export VULKAN_SDK=/opt/vulkan
export PATH=$VULKAN_SDK/bin:$PATH
export LD_LIBRARY_PATH=$VULKAN_SDK/lib:$LD_LIBRARY_PATH
```

### SLEEF
```bash
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig:$PKG_CONFIG_PATH
```

### ROCm
```bash
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

### Rust/Cargo (for Naga)
```bash
export PATH=$HOME/.cargo/bin:$PATH
```

### Node.js/npm
```bash
# Usually already in PATH after installation
# Verify with: which node && which npm
```

## Troubleshooting

### MLIR Not Found

If CMake can't find MLIR after installation:

```bash
# Find MLIR installation
find /usr -name "MLIRConfig.cmake" 2>/dev/null
find /opt -name "MLIRConfig.cmake" 2>/dev/null

# Set MLIR_DIR explicitly
export MLIR_DIR=/path/to/mlir/lib/cmake/mlir
cmake .. -DMLIR_DIR=$MLIR_DIR
```

### CUDA Not Detected

```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Verify CUDA paths
ls /usr/local/cuda/bin/nvcc
```

### Vulkan Not Found

```bash
# Check Vulkan installation
vulkaninfo --summary

# Verify Vulkan SDK path
echo $VULKAN_SDK
```

### SLEEF Build Fails

SLEEF requires:
- CMake 3.16+
- C compiler (GCC/Clang)
- Git

If build fails, check:
1. All dependencies are installed
2. Sufficient disk space (~500MB)
3. Write permissions to `~/.local`

## Manual Installation

If automatic installation fails, see:
- [MLIR Installation Guide](../docs/mlir/BUILDING_WITH_MLIR.md)
- [CUDA Installation](https://developer.nvidia.com/cuda-downloads)
- [Vulkan SDK](https://vulkan.lunarg.com/sdk/home)
- [SLEEF Repository](https://github.com/shibatch/sleef)

## Next Steps

After running the setup script:

1. **Verify installation:**
   ```bash
   source ~/.cml_env.sh  # Linux/macOS
   # or
   . ~\.cml_env.ps1     # Windows PowerShell
   ```

2. **Build C-ML:**
   ```bash
   cd /path/to/C-ML
   mkdir build && cd build
   cmake .. -DMLIR_DIR=$MLIR_DIR
   make -j$(nproc)
   ```

3. **Build Frontend (optional):**
   ```bash
   cd viz-ui
   npm install  # Install React and other dependencies
   npm run build
   ```

3. **Run tests:**
   ```bash
   ./test_dispatch
   ./test_backends
   ```

## Support

For issues or questions:
- Check [Troubleshooting](#troubleshooting) section
- See [C-ML Documentation](../docs/)
- Open an issue on GitHub
