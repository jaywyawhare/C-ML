# CML Quick Start Guide

Get started with C-ML in 5 minutes!

## Prerequisites

- C11 compatible compiler (GCC, Clang, or MSVC)
- MLIR 18.x installed (see [COMPILATION.md](COMPILATION.md#installing-mlir-required))
- Make or CMake

## Step 1: Clone and Build

```bash
# Clone repository
git clone https://github.com/jaywyawhare/C-ML.git
cd C-ML

# Build (creates build/ directory)
make

# Verify build
ls build/main build/examples/
```

## Step 2: Run Your First Example

```bash
# Run a simple example
./build/examples/autograd_example

# Expected output:
# Testing autograd functionality...
# Forward pass: result = [...]
# Backward pass: gradients computed
```

## Step 3: Create Your First Program

Create `my_first_ml.c`:

```c
#include "cml.h"
#include <stdio.h>

int main(void) {
    // Initialize CML
    cml_init();
    cml_seed(42);  // Reproducible results

    printf("=== CML Quick Start ===\n\n");

    // Get default device and data type
    DeviceType device = cml_get_default_device();
    DType dtype = cml_get_default_dtype();

    // 1. Create a simple neural network
    printf("Step 1: Building neural network...\n");
    Sequential* model = cml_nn_sequential();

    // Input layer: 4 neurons
    // Hidden layer: 8 neurons with ReLU
    // Output layer: 2 neurons
    model = cml_nn_sequential_add(model,
        (Module*)cml_nn_linear(4, 8, dtype, device, true));
    model = cml_nn_sequential_add(model,
        (Module*)cml_nn_relu(false));
    model = cml_nn_sequential_add(model,
        (Module*)cml_nn_linear(8, 2, dtype, device, true));

    // 2. Create sample data
    printf("Step 2: Creating sample data...\n");
    Tensor* input = tensor_randn((int[]){10, 4}, 2, NULL);
    Tensor* target = tensor_randn((int[]){10, 2}, 2, NULL);

    // 3. Create optimizer
    printf("Step 3: Setting up optimizer...\n");
    Optimizer* optimizer = cml_optim_adam_for_model(
        (Module*)model,
        0.001f,    // learning rate
        0.0f,      // weight decay
        0.9f,      // beta1
        0.999f,    // beta2
        1e-8f      // epsilon
    );

    // 4. Training loop
    printf("Step 4: Training for 10 epochs...\n\n");
    cml_nn_module_set_training((Module*)model, true);

    for (int epoch = 0; epoch < 10; epoch++) {
        // Zero gradients
        cml_optim_zero_grad(optimizer);

        // Forward pass
        Tensor* output = cml_nn_module_forward((Module*)model, input);

        // Compute loss (MSE)
        Tensor* loss = cml_nn_mse_loss(output, target);

        // Get loss value for printing
        float* loss_data = (float*)tensor_data_ptr(loss);
        float loss_value = loss_data ? loss_data[0] : 0.0f;

        // Backward pass (compute gradients)
        cml_backward(loss, NULL, false, false);

        // Update parameters
        cml_optim_step(optimizer);

        if ((epoch + 1) % 2 == 0) {
            printf("Epoch %2d: Loss = %.6f\n", epoch + 1, loss_value);
        }

        // Cleanup tensors from this iteration
        tensor_free(loss);
        tensor_free(output);
    }

    printf("\nStep 5: Making inference...\n");
    cml_nn_module_set_training((Module*)model, false);

    Tensor* pred = cml_nn_module_forward((Module*)model, input);
    float* pred_data = (float*)tensor_data_ptr(pred);

    printf("First prediction: [%.4f, %.4f]\n",
           pred_data[0], pred_data[1]);

    // Cleanup
    printf("\nCleaning up...\n");
    tensor_free(input);
    tensor_free(target);
    tensor_free(pred);
    optimizer_free(optimizer);
    module_free((Module*)model);
    cml_cleanup();

    printf("Done!\n");
    return 0;
}
```

## Step 4: Compile and Run

```bash
# Compile your program
gcc -std=c11 -O2 my_first_ml.c \
    -I./include \
    -L./build/lib \
    -lcml \
    -lm \
    -o my_first_ml

# Run it
export LD_LIBRARY_PATH=./build/lib:$LD_LIBRARY_PATH
./my_first_ml
```

Expected output:
```
=== CML Quick Start ===

Step 1: Building neural network...
Step 2: Creating sample data...
Step 3: Setting up optimizer...
Step 4: Training for 10 epochs...

Epoch  2: Loss = 0.475532
Epoch  4: Loss = 0.421123
Epoch  6: Loss = 0.389456
Epoch  8: Loss = 0.362789
Epoch 10: Loss = 0.341234

Step 5: Making inference...
First prediction: [-0.1234, 0.5678]

Cleaning up...
Done!
```

## Common Next Steps

### Train on Your Own Data

```c
// Load your data
Tensor* X = load_data("data.csv");  // Your loading function
Tensor* y = load_labels("labels.csv");

// Ensure correct shapes
printf("Data shape: [%d, %d]\n", X->shape[0], X->shape[1]);

// Use in training loop
Tensor* output = cml_nn_module_forward((Module*)model, X);
```

### Use Different Activation Functions

```c
// Available activations:
model = cml_nn_sequential_add(model, (Module*)cml_nn_relu(false));
model = cml_nn_sequential_add(model, (Module*)cml_nn_sigmoid(false));
model = cml_nn_sequential_add(model, (Module*)cml_nn_tanh(false));
model = cml_nn_sequential_add(model, (Module*)cml_nn_gelu(false));
```

### Use Different Loss Functions

```c
// MSE Loss (Regression)
Tensor* loss = cml_nn_mse_loss(predictions, targets);

// Cross Entropy (Classification)
Tensor* loss = cml_nn_cross_entropy_loss(logits, labels);

// Binary Cross Entropy
Tensor* loss = cml_nn_bce_loss(predictions, targets);

// MAE Loss (Regression)
Tensor* loss = cml_nn_mae_loss(predictions, targets);
```

### Use Different Optimizers

```c
// SGD with momentum
Optimizer* opt = cml_optim_sgd_for_model(
    (Module*)model,
    0.01f,   // learning rate
    0.9f,    // momentum
    0.0f     // weight decay
);

// RMSprop
Optimizer* opt = cml_optim_rmsprop_for_model(
    (Module*)model,
    0.001f,  // learning rate
    0.99f,   // alpha (decay rate)
    1e-8f,   // epsilon
    0.0f     // weight decay
);

// AdaGrad
Optimizer* opt = cml_optim_adagrad_for_model(
    (Module*)model,
    0.01f,   // learning rate
    1e-10f   // epsilon
);
```

### Add Regularization (Dropout)

```c
// Add dropout layer in model
model = cml_nn_sequential_add(model,
    (Module*)cml_nn_dropout(0.5f, false));  // 50% dropout
```

### Use GPU

```bash
# Compile (same as before)
gcc -std=c11 -O2 my_program.c -I./include -L./build/lib -lcml -lm -o my_program

# Run on GPU
CML_DEVICE=cuda ./my_program  # NVIDIA GPU
CML_DEVICE=metal ./my_program # Apple Silicon
CML_DEVICE=rocm ./my_program  # AMD GPU
```

### Enable Visualization

```bash
# Build visualization frontend (requires Node.js/npm)
cd viz-ui && npm install && npm run build && cd ..

# Run with visualization
VIZ=1 ./my_program

# Access dashboard at http://localhost:8001
```

## Example: MNIST Digit Recognition

Adapt the quick start for MNIST:

```c
// After building model...

// 1. Load MNIST data (you need to implement or find data)
Dataset* dataset = load_mnist("data/mnist");
Tensor* X_train = dataset->X;
Tensor* y_train = dataset->y;

// 2. Define model for 28x28 images → 10 digits
Sequential* model = cml_nn_sequential();
model = cml_nn_sequential_add(model,
    (Module*)cml_nn_linear(784, 128, dtype, device, true));  // Flatten + FC
model = cml_nn_sequential_add(model,
    (Module*)cml_nn_relu(false));
model = cml_nn_sequential_add(model,
    (Module*)cml_nn_dropout(0.2f, false));
model = cml_nn_sequential_add(model,
    (Module*)cml_nn_linear(128, 10, dtype, device, true));

// 3. Create optimizer and train...
Optimizer* optimizer = cml_optim_adam_for_model(
    (Module*)model, 0.001f, 0.0001f, 0.9f, 0.999f, 1e-8f);

// 4. Training loop with cross entropy loss
for (int epoch = 0; epoch < 10; epoch++) {
    cml_optim_zero_grad(optimizer);

    Tensor* logits = cml_nn_module_forward((Module*)model, X_train);
    Tensor* loss = cml_nn_cross_entropy_loss(logits, y_train);

    cml_backward(loss, NULL, false, false);
    cml_optim_step(optimizer);

    printf("Epoch %d: Loss = %.4f\n", epoch, get_loss_value(loss));

    tensor_free(loss);
    tensor_free(logits);
}
```

## Troubleshooting

### Compilation Error: "cml.h not found"

```bash
# Make sure to use correct include path
gcc -I./include my_program.c ...

# Or use full path
gcc -I/home/user/C-ML/include my_program.c ...
```

### Linker Error: "undefined reference to `cml_init`"

```bash
# Make sure to link the library
gcc my_program.c -L./build/lib -lcml -lm ...

# And set library path at runtime
export LD_LIBRARY_PATH=./build/lib:$LD_LIBRARY_PATH
./my_program
```

### Runtime Error: "MLIR not initialized"

```bash
# Make sure to call cml_init() at program start
int main(void) {
    cml_init();  // ← Add this!
    // ... rest of code ...
    cml_cleanup();
}
```

### NaN or Inf in Results

This usually means:
1. Learning rate too high - reduce it (try 0.0001)
2. Data not normalized - normalize to [-1, 1] or [0, 1] range
3. Bad initialization - set seed with `cml_seed(42)`

```c
// Normalize data to [0, 1]
void normalize_tensor(Tensor* t) {
    // Implement min-max normalization
    // result = (x - min) / (max - min)
}
```

## More Examples

See the `examples/` directory:

```bash
./build/examples/autograd_example        # Autograd demonstrations
./build/examples/training_loop_example   # Training loop patterns
./build/examples/test_example            # Comprehensive tests
./build/examples/mnist_example           # MNIST recognition (if data available)
```

## Full Documentation

- [API Guide](API_GUIDE.md) - Complete API reference
- [Compilation Guide](COMPILATION.md) - Build options and troubleshooting
- [Running Guide](RUNNING.md) - How to run programs
- [Architecture](ARCHITECTURE.md) - Internal design
- [Overview](OVERVIEW.md) - Features and capabilities

## Tips & Tricks

1. **Reproducible Results**: Use `cml_seed(42)` at startup
2. **Debug Output**: Set `CML_LOG_LEVEL=4` for verbose logging
3. **Memory Efficiency**: Use `CleanupContext` for automatic cleanup
4. **Performance**: Use `make fast` for CPU-optimized build
5. **GPU Support**: Build with `-DENABLE_CUDA=ON` for NVIDIA GPUs

## Getting Help

- Check [Running Guide](RUNNING.md#troubleshooting) for common issues
- Review [API Guide](API_GUIDE.md) for function signatures
- Run examples in `examples/` directory as templates
- Enable debug logging: `CML_LOG_LEVEL=4 ./my_program`

Happy learning! 🚀
