# Contributing to C-ML

Thank you for your interest in contributing to C-ML! This guide will help you
get started.

## Build Instructions

### Prerequisites

- C compiler with C11 support (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.16 or later
- (Optional) LLVM 19+ for JIT backend
- (Optional) CUDA Toolkit for NVIDIA GPU support
- (Optional) ROCm/HIP for AMD GPU support
- (Optional) OpenCL for portable GPU compute

### Building

```bash
# Configure and build
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)

# Run tests
cd build && ctest --output-on-failure

# Install
cmake --install build --prefix /usr/local
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_TESTS` | ON | Build test suite |
| `BUILD_EXAMPLES` | ON | Build example programs |
| `BUILD_SHARED_LIBS` | ON | Build shared library |
| `ENABLE_LLVM_BACKEND` | ON | Enable LLVM JIT backend |
| `ENABLE_CUDA` | ON | Enable CUDA backend |
| `ENABLE_ROCM` | ON | Enable ROCm/HIP backend |
| `ENABLE_OPENCL` | ON | Enable OpenCL backend |
| `ENABLE_DISTRIBUTED` | OFF | Enable distributed training |

## Code Style

### C Code

- **Standard**: C11
- **Indentation**: 4 spaces (no tabs)
- **Line length**: 100 characters soft limit
- **Braces**: K&R style (opening brace on same line)
- **Naming**:
  - Functions: `snake_case` (e.g., `tensor_matmul`, `nn_linear`)
  - Types: `PascalCase` (e.g., `Tensor`, `Linear`, `AutogradEngine`)
  - Macros: `UPPER_SNAKE_CASE` (e.g., `CML_VERSION_MAJOR`)
  - Private/static: `g_` prefix for globals, `_` suffix for internal helpers

### File Organization

- Headers in `include/` mirror source layout in `src/`
- Each layer gets its own header (`include/nn/layers/foo.h`) and source (`src/nn/layers/foo.c`)
- Use include guards: `#ifndef CML_<PATH>_<NAME>_H`

### Example

```c
/**
 * @brief Create a new FooBar layer
 * @param input_size Input dimension
 * @param output_size Output dimension
 * @return New FooBar, or NULL on failure
 */
FooBar* nn_foobar(int input_size, int output_size, DType dtype, DeviceType device) {
    FooBar* fb = calloc(1, sizeof(FooBar));
    if (!fb) return NULL;

    module_init(&fb->base, "FooBar", foobar_forward, foobar_free);
    fb->input_size = input_size;
    fb->output_size = output_size;
    // ... initialize weights ...

    return fb;
}
```

## Testing

### Requirements

- All new layers must have gradient checks in `tests/grad_check.c`
- All bug fixes should include a regression test
- Run the full test suite before submitting:

```bash
cd build && ctest --output-on-failure
```

### Test Structure

Tests are auto-discovered by CMake from `tests/*.c`. Each test file should:

1. Include `cml.h`
2. Have a `main()` function that returns 0 on success, non-zero on failure
3. Print clear pass/fail messages

### Gradient Checks

Use `numerical_grad_check()` to verify analytical gradients:

```c
// Central finite differences: (f(x+eps) - f(x-eps)) / (2*eps)
bool ok = numerical_grad_check(param_tensor, loss_fn, /*eps=*/1e-3, /*tol=*/1e-2);
```

## Pull Request Process

1. **Fork** the repository and create a feature branch from `master`
2. **Implement** your changes following the code style guide
3. **Test** thoroughly — all existing tests must pass
4. **Document** new public APIs in the appropriate header files
5. **Submit** a pull request with a clear description

### Commit Message Format

Use concise, descriptive commit messages:

```
<type>: <short summary>

<optional body with more details>
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `build`, `ci`

Examples:
- `feat: Add ConvTranspose3d layer`
- `fix: Prevent NULL dereference in tensor_clone`
- `test: Add gradient checks for all normalization layers`

## Architecture Overview

```
C-ML/
├── include/           # Public headers
│   ├── cml.h         # Main entry point — includes everything
│   ├── core/         # Config, logging, training, serialization
│   ├── tensor/       # Tensor struct and operations
│   ├── autograd/     # Automatic differentiation
│   ├── nn/           # Neural network module system
│   │   └── layers/   # Individual layer implementations
│   ├── optim/        # Optimizers
│   ├── ops/          # IR operations, SIMD, GPU backends
│   ├── backend/      # Device management, thread pool, OpenCL
│   └── alloc/        # Memory management and pools
├── src/              # Implementation (mirrors include/)
├── tests/            # Test suite (auto-discovered by CMake)
├── examples/         # Tutorials, benchmarks, demos
├── docs/             # MkDocs documentation
└── python/           # Python bindings
```

### Key Design Decisions

- **Lazy execution**: Operations build an IR graph; data is materialized on demand
- **Module system**: Layers inherit from `Module` base with parameter management
- **Multi-backend**: GPU ops dispatched via backend abstraction; CUDA/ROCm use dlopen
- **Autograd**: IR-based backward pass with gradient accumulation

## Questions?

Open an issue on GitHub or check the [documentation](https://jaywyawhare.github.io/C-ML/).
