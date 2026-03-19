# Installation

## Quick Start

```bash
# 1. Build C library
cd C-ML
mkdir -p build && cd build && cmake .. && make -j$(nproc)

# 2. Build Python bindings
cd ../python
pip install cffi
python3 cml/build_cffi.py

# 3. Install
pip install -e .

# 4. Verify
python3 -c "import cml; cml.init(); print('ok'); cml.cleanup()"
```

## Requirements

- Python 3.8+
- `cffi` package
- C-ML library (built via CMake)
- LLVM 18+ (optional, for JIT backend)

## Platform Notes

**Linux**: `sudo apt install python3-dev` (Debian/Ubuntu) or `sudo pacman -S python` (Arch).

**macOS**: `brew install python3`. If using Homebrew LLVM: `export LDFLAGS="-L$(brew --prefix llvm)/lib"`.

**Windows**: Use WSL2. Native builds require Visual Studio + CMake.

## Library Path

If you get `OSError: cannot find libcml.so`:

```bash
export LD_LIBRARY_PATH=/path/to/C-ML/build/lib:$LD_LIBRARY_PATH
```

## Development Mode

```bash
pip install -e ".[dev]"    # editable install
pytest tests/              # run tests
```
