# C-ML Examples

This directory contains example programs showing how to use the C-ML library.

## Quick Start

After building and installing C-ML:

```bash
# Compile an example
gcc simple_xor.c -I/usr/local/include/cml -lcml -lm -o simple_xor

# Run it
./simple_xor

# Run with visualization
VIZ=1 ./simple_xor
```

## Examples

### `simple_xor.c` - XOR Neural Network
**Difficulty:** Beginner
**Topics:** Basic neural network, training loop, predictions

A complete example showing how to:
- Create a simple feedforward network
- Train on the XOR problem
- Make predictions
- Evaluate accuracy

**Compile:**
```bash
gcc simple_xor.c -lcml -lm -o simple_xor
```

**Expected output:**
```
=== C-ML Simple XOR Example ===

Training data created: 4 samples

Model architecture:
...

Starting training...
Epoch           Loss
-----           ----
0               0.250000
100             0.123456
...
1000            0.001234

✓ Training complete!

Testing predictions:
Input           Prediction      Expected        Correct?
-----           ----------      --------        --------
[0, 0]          0.0123          0               ✓
[0, 1]          0.9876          1               ✓
[1, 0]          0.9845          1               ✓
[1, 1]          0.0234          0               ✓

Accuracy: 4/4 (100%)
```

### Other Examples

The following examples are also available in this directory:

- **`autograd_example.c`** - Demonstrates automatic differentiation
- **`training_loop_example.c`** - Advanced training loop with metrics
- **`bench_gemm.c`** - Matrix multiplication benchmarks
- **`export_graph.c`** - Export computation graph for visualization
- **`early_stopping_lr_scheduler.c`** - Training with early stopping and learning rate scheduling

## Building All Examples

You can build all examples at once:

```bash
# From the C-ML root directory
make all

# Examples will be in build/examples/
ls build/examples/
```

Or build them individually:

```bash
# From the examples directory
gcc simple_xor.c -I../include -L../build -lcml -lm -o simple_xor
```

## Using Examples as Templates

These examples are designed to be used as templates for your own projects:

1. **Copy the example** to your project directory
2. **Modify** the network architecture and training data
3. **Compile** using the same command
4. **Run** and iterate

## Common Patterns

### Creating a Model
```c
Sequential* model = cml_nn_sequential();
cml_nn_sequential_add(model, (Module*)cml_nn_linear(input_size, hidden_size, DTYPE_FLOAT32, DEVICE_CPU, true));
cml_nn_sequential_add(model, (Module*)cml_nn_relu(false));
cml_nn_sequential_add(model, (Module*)cml_nn_linear(hidden_size, output_size, DTYPE_FLOAT32, DEVICE_CPU, true));
```

### Training Loop
```c
for (int epoch = 0; epoch < num_epochs; epoch++) {
    // Forward
    Tensor* output = cml_nn_sequential_forward(model, X);
    Tensor* loss = cml_nn_mse_loss(output, y);

    // Backward
    cml_optim_zero_grad(optimizer);
    cml_backward(loss, NULL, false, false);
    cml_optim_step(optimizer);
}
```

### Creating Tensors
```c
float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
int shape[] = {2, 2};
Tensor* t = cml_tensor(data, shape, 2, NULL);
```

## Troubleshooting

### "cannot find -lcml"
Make sure C-ML is installed or specify the library path:
```bash
gcc simple_xor.c -I../include -L../build -lcml -lm -o simple_xor
```

### "error while loading shared libraries"
Add the library path to LD_LIBRARY_PATH:
```bash
export LD_LIBRARY_PATH=../build:$LD_LIBRARY_PATH
./simple_xor
```

Or use the static library:
```bash
gcc simple_xor.c -I../include ../build/libcml.a -lm -o simple_xor
```

## Next Steps

1. Try running `simple_xor.c` first
2. Modify it to solve a different problem
3. Check out the more advanced examples
4. Read the [USAGE.md](../USAGE.md) guide for more details

## Contributing

Have a cool example? Submit a pull request! Good examples:
- Solve a specific problem clearly
- Are well-commented
- Include compilation instructions
- Show best practices
