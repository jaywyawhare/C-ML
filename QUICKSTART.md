# Quick Start Guide - Running C-ML

## Complete Setup & Run Instructions

### 1. Build Everything (Recommended)

```bash
# Build C library + frontend in one command
./build.sh all

# Or build manually:
make                    # Build C library
cd viz-ui && npm install && npm run build && cd ..  # Build frontend
```

### 2. Run Examples

After building, run any example:

```bash
# Basic examples
./build/examples/hello_cml
./build/examples/simple_xor
./build/examples/training_loop_example

# With visualization dashboard (opens http://localhost:8001)
VIZ=1 ./build/examples/training_loop_example
VIZ=1 ./build/examples/mnist_example
```

### 3. Run with Visualization Dashboard

The visualization server automatically starts when `VIZ=1` is set:

```bash
# Start example with visualization
VIZ=1 ./build/examples/training_loop_example

# Or manually start the server
python3 scripts/fastapi_server.py

# Then run your program (it will connect automatically)
./build/examples/training_loop_example
```

**Visualization Dashboard**: Opens at `http://localhost:8001` (or port specified by `VIZ_PORT`)

### 4. Run Tests

```bash
# Run all tests
make test

# Or use the build script
./build.sh test
```

### 5. Complete Development Workflow

```bash
# 1. Clean previous builds
make clean

# 2. Build everything
./build.sh all

# 3. Run example with visualization
VIZ=1 ./build/examples/training_loop_example

# 4. Open browser to http://localhost:8001
# You'll see:
#   - Computational Graph visualization
#   - Training metrics (loss, accuracy)
#   - Kernel code generation
```

## Environment Variables

- `VIZ=1` - Enable visualization dashboard
- `VIZ_PORT=8001` - Change visualization port (default: 8001)
- `PORT=8001` - Backend server port
- `HOST=0.0.0.0` - Backend server host
- `WORKERS=1` - Number of uvicorn workers
- `USE_UVLOOP=1` - Use uvloop for better async performance

## Troubleshooting

### MLIR Not Found
```bash
# Install MLIR (Arch Linux)
sudo pacman -S llvm mlir

# Or check if mlir-config exists
which mlir-config
```

### Frontend Not Building
```bash
cd viz-ui
npm install
npm run build
```

### Library Not Found
```bash
# Set library path
export LD_LIBRARY_PATH=$PWD/build:$LD_LIBRARY_PATH

# Or install system-wide
sudo ./build.sh install
```

## Quick Commands Reference

```bash
# Build
make                    # Build C library
make release            # Optimized build
make debug              # Debug build
./build.sh all          # Build everything

# Run
./build/examples/hello_cml
VIZ=1 ./build/examples/training_loop_example

# Test
make test

# Clean
make clean
./build.sh clean

# Install
sudo ./build.sh install
```
