# Getting Started with C-ML

A quick guide to get you started with C-ML in minutes.

## Installation

### Prerequisites

- C11 compatible compiler (GCC 4.9+, Clang 3.5+, MSVC 2015+)
- CMake 3.16+ (optional) or Make
- Math library (libm, usually included)

### Build from Source

```bash
# Clone repository
git clone https://github.com/jaywyawhare/C-ML.git
cd C-ML
```

#### Option A: Use the helper script (recommended)

```bash
./build.sh all     # Builds the C core and the viz frontend
./build.sh lib     # Library only
./build.sh test    # Build + run tests
```

#### Option B: Use Make or CMake directly

```bash
# Make
make clean && make

# CMake
mkdir -p build && cd build
cmake -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### Link Your Program

```bash
# Static library
gcc your_program.c -I./include -L./build/lib -lcml -lm -o your_program

# Shared library
gcc your_program.c -I./include -L./build/lib -lcml -lm -o your_program
export LD_LIBRARY_PATH=./build/lib:$LD_LIBRARY_PATH
```

## Your First Program

### Step 1: Include the Header

```c
#include "cml.h"
```

### Step 2: Initialize the Library

```c
int main(void) {
    cml_init();
    cml_seed(42);  // For reproducible results

    // Your code here

    cml_cleanup();
    return 0;
}
```

### Step 3: Create Tensors

```c
// Create tensors
int shape[] = {2, 3};
TensorConfig config = tensor_config_default();
Tensor* a = cml_zeros(shape, 2, &config);
Tensor* b = cml_ones(shape, 2, &config);
```

### Step 4: Perform Operations

```c
// Elementwise operations
Tensor* sum = cml_add(a, b);
Tensor* product = cml_mul(a, b);

// Cleanup
tensor_free(sum);
tensor_free(product);
tensor_free(a);
tensor_free(b);
```

## Complete Example

```c
#include "cml.h"
#include <stdio.h>

int main(void) {
    // Initialize
    cml_init();
    cml_seed(42);

    // Create tensors
    int shape[] = {2, 2};
    TensorConfig config = tensor_config_default();
    Tensor* a = cml_ones(shape, 2, &config);
    Tensor* b = cml_ones(shape, 2, &config);

    // Perform operation
    Tensor* result = cml_add(a, b);

    // Print result
    float* data = (float*)tensor_data_ptr(result);
    if (data) {
        printf("Result:\n");
        printf("  [%.1f, %.1f]\n", data[0], data[1]);
        printf("  [%.1f, %.1f]\n", data[2], data[3]);
    }

    // Cleanup
    tensor_free(result);
    tensor_free(a);
    tensor_free(b);
    cml_cleanup();

    return 0;
}
```

## Next Steps

1. **Learn Tensor Operations**: See [API Reference](api_reference.md#tensor-operations)
1. **Build Neural Networks**: See [Neural Network Layers](nn_layers.md)
1. **Train Models**: See [Training Guide](training.md)
1. **Understand Autograd**: See [Autograd System](autograd.md)

## Common Patterns

### Pattern 1: Model Creation

```c
Sequential* model = cml_nn_sequential();
DeviceType device = cml_get_default_device();
DType dtype = cml_get_default_dtype();

model = sequential_add_chain(model,
    (Module*)cml_nn_linear(784, 128, dtype, device, true),
    (Module*)cml_nn_relu(false),
    (Module*)cml_nn_linear(128, 10, dtype, device, true),
    NULL
);
```

### Pattern 2: Training Loop

```c
for (int epoch = 0; epoch < num_epochs; epoch++) {
    cml_optim_zero_grad(optimizer);
    Tensor* outputs = cml_nn_module_forward((Module*)model, inputs);
    Tensor* loss = cml_nn_mse_loss(outputs, targets);
    cml_backward(loss, NULL, false, false);
    cml_optim_step(optimizer);

    tensor_free(loss);
    tensor_free(outputs);
}
```

### Pattern 3: Error Handling

```c
Tensor* t = tensor_empty(shape, ndim, NULL);
if (!t) {
    if (CML_HAS_ERRORS()) {
        printf("Error: %s\n", CML_LAST_ERROR());
        error_stack_print_all();
    }
    return -1;
}
```

## Troubleshooting

### Build Issues

**Problem**: Compiler not found

- **Solution**: Install GCC or Clang: `sudo apt-get install gcc` (Linux) or `brew install gcc` (macOS)

**Problem**: CMake version too old

- **Solution**: Update CMake or use Make instead

### Runtime Issues

**Problem**: Library not found

- **Solution**: Set `LD_LIBRARY_PATH` or install library system-wide

**Problem**: Segmentation fault

- **Solution**: Ensure `cml_init()` is called before any operations

## Python Bindings

CML provides Python bindings via CFFI for using the library from Python:

```bash
# Build the C library first
cd build && cmake .. && make -j4

# Install Python bindings
cd ../python
pip install cffi
python setup.py install

# Test
python -c "import cml; cml.init(); print('OK'); cml.cleanup()"
```

See `python/README.md` for full documentation including:

- 63 comprehensive CFFI tests
- Neural network examples
- Memory management details
- Lazy evaluation support

## Resources

- [Quick Start commands](../QUICKSTART.md) - OS-specific build script usage
- Run tests: `./build.sh test` or `make test`
- Run examples: `./build/examples/hello_cml` (after building)
- Visualization: `VIZ=1 ./build/examples/training_loop_example`
- [API Reference](api_reference.md) - Complete API documentation
- [Examples](../examples/) - Working code examples
- [Python Bindings](../python/README.md) - Python CFFI bindings
- [GitHub Issues](https://github.com/jaywyawhare/C-ML/issues) - Report bugs or ask questions
