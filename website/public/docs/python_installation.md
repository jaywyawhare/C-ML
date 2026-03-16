# CML Python Bindings - Installation Guide

Complete guide to installing CML Python bindings.

## Quick Start (5 minutes)

### Prerequisites Check

```bash
# Check Python version (3.8+)
python3 --version

# Check if pip is available
python3 -m pip --version
```

### Installation Steps

```bash
# 1. Ensure C-ML library is built
cd ..  # Go to C-ML root
make   # Build C library

# 2. Install Python bindings
cd python
pip3 install cffi
python3 setup.py install

# 3. Verify
python3 -c "import cml; cml.init(); print('Success!'); cml.cleanup()"
```

## Detailed Installation

### Step 1: Build C-ML Library

The Python bindings require the compiled C library.

**Using Make:**

```bash
cd ..
make
```

This creates:
- `build/lib/libcml.a` (static)
- `build/lib/libcml.so` (shared, if built)

**Using CMake:**

```bash
cd ..
mkdir -p build && cd build
cmake -DBUILD_SHARED_LIBS=ON ..
make -j$(nproc)
cd ..
```

### Step 2: Install Dependencies

```bash
# Install CFFI (required for building bindings)
pip3 install cffi

# Optional: install development tools
pip3 install setuptools wheel build
```

### Step 3: Build CFFI Bindings

Option A: Automatic (recommended)

```bash
cd python
python3 setup.py build_ext --inplace
```

Option B: Manual

```bash
cd python
python3 cml/build_cffi.py
```

### Step 4: Install Package

**System-wide installation:**

```bash
python3 setup.py install
```

**User installation (no sudo):**

```bash
python3 setup.py install --user
```

**Development mode (editable):**

```bash
pip3 install -e .
```

Development mode is recommended for development work.

### Step 5: Verify Installation

Test basic functionality:

```bash
python3 << 'EOF'
import cml

cml.init()
print("CML initialized")

x = cml.randn([10, 10])
print(f"Created tensor with {x.size} elements")

y = cml.zeros([10, 10])
z = x + y
print("Tensor operations work")

cml.cleanup()
print("CML Python bindings working!")
EOF
```

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install python3-dev python3-pip

# Install CFFI
pip3 install cffi

# Build and install
cd python
python3 setup.py install
```

### Linux (Arch)

```bash
# Install dependencies
sudo pacman -S python python-pip

# Install CFFI
pip install cffi

# Build and install
cd python
python setup.py install
```

### macOS

```bash
# Install dependencies (if using Homebrew)
brew install python3

# Install CFFI
pip3 install cffi

# Build and install
cd python
python3 setup.py install
```

If using LLVM from Homebrew:

```bash
export LDFLAGS="-L$(brew --prefix llvm@18)/lib"
export CPPFLAGS="-I$(brew --prefix llvm@18)/include"
python3 setup.py build_ext --inplace
```

### Windows (WSL2 Recommended)

Use WSL2 with Ubuntu and follow Linux instructions.

**Native Windows (advanced):**

```cmd
# Install Python from python.org or Microsoft Store

# Install dependencies
pip install cffi

# Build C library with CMake
cd ..
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release

# Build Python bindings
cd ..\python
python setup.py build_ext --inplace
python setup.py install
```

## Troubleshooting

### Problem: "cffi not found"

```
ModuleNotFoundError: No module named 'cffi'
```

**Solution:**

```bash
pip3 install cffi
```

### Problem: "libcml.a not found"

```
OSError: cannot open shared object file
```

**Solution:**

```bash
# Rebuild C library
cd ..
make clean
make

# Then rebuild bindings
cd python
python3 setup.py build_ext --inplace
```

### Problem: "MLIR not found during build"

```
ERROR: MLIR not found
```

**Solution:**

```bash
# Set MLIR path explicitly
export MLIR_DIR=/path/to/mlir/lib/cmake/mlir
cd python
python3 setup.py build_ext --inplace
```

### Problem: Library path issues at runtime

```
ImportError: libcml.so.18: cannot open shared object file
```

**Solution:**

```bash
# Set library path
export LD_LIBRARY_PATH=$(pwd)/../build/lib:$LD_LIBRARY_PATH

# Or create symbolic link
ln -s ../build/lib/libcml.so ./

# Then test
python3 -c "import cml"
```

### Problem: Permission denied during installation

```
PermissionError: [Errno 13] Permission denied
```

**Solution:**

Install as user instead:

```bash
python3 setup.py install --user

# Or use pip with user flag
pip3 install --user -e .
```

### Problem: Python version mismatch

```
ERROR: Python version X.X does not match requirement >=3.8
```

**Solution:**

Ensure Python 3.8+ is installed and used:

```bash
# Check version
python3 --version

# Use explicit Python path
/usr/bin/python3.10 setup.py install

# Or create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install cffi
python setup.py install
```

## Virtual Environment Setup

Recommended for development:

```bash
# Create virtual environment
python3 -m venv cml_env

# Activate it
source cml_env/bin/activate  # On Windows: cml_env\Scripts\activate

# Install dependencies
pip install cffi

# Build and install
cd python
python setup.py develop

# Test
python -c "import cml; print('OK')"

# Deactivate when done
deactivate
```

## Building for Distribution

### Create Distribution Package

```bash
cd python

# Build wheel
pip install wheel
python -m build

# Output in dist/
ls dist/
```

### Install from Wheel

```bash
pip install dist/cml-*.whl
```

## Uninstallation

### If installed via pip

```bash
pip uninstall cml
```

### If installed via setup.py

```bash
cd python
python setup.py install --record files.txt
xargs rm -f < files.txt
```

## Verification

### Test Script

```bash
python3 << 'EOF'
import sys
print(f"Python: {sys.version}")

try:
    import cml
    print(f"CML: OK")

    # Test basic operations
    cml.init()
    x = cml.randn([10, 10])
    cml.cleanup()

    print("All tests passed!")
    sys.exit(0)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
```

### Run Examples

```bash
cd examples
python3 01_hello_cml.py
python3 02_neural_network.py
```

## Next Steps

- See [API Reference](api_reference.md) for API documentation
- See [Examples](examples.md) for usage examples
- See [Documentation](index.md) for all CML docs

## Support

- **Build issues**: Check build logs, ensure dependencies are installed
- **Runtime issues**: Set `export LD_LIBRARY_PATH=../../build/lib:$LD_LIBRARY_PATH`
- **API questions**: See API_GUIDE.md
- **Examples**: Check examples/ folder

## Version Information

- **CML Python Bindings**: v0.0.3
- **Minimum Python**: 3.8
- **CFFI Requirement**: 1.14.0+
- **C Standard**: C11
- **MLIR Requirement**: 18.x+
