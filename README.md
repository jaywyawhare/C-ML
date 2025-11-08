<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/dark-mode.svg">
    <img alt="C-ML" src="docs/img.svg" height="96">
  </picture>
</p>

<h1 align="center">C-ML: C Machine Learning Library</h1>

<p align="center">
  A comprehensive machine learning library written in pure C, providing automatic differentiation, neural network layers, optimizers, and training utilities with an interactive visualization UI.
</p>

______________________________________________________________________

## 🚀 Features

### Core Library

- **Automatic Differentiation (Autograd)**: Dynamic computation graphs with automatic gradient computation
- **Neural Network Layers**: Complete set of layers including Linear, Conv2d, BatchNorm2d, Pooling, Activations, Dropout, LayerNorm, and more
- **Optimizers**: SGD and Adam optimizers with momentum, weight decay, and learning rate scheduling
- **Tensor Operations**: Comprehensive tensor operations with broadcasting support
- **Loss Functions**: MSE, MAE, BCE, Cross Entropy, and more
- **Memory Management**: Safe memory management with automatic cleanup
- **Training Metrics**: Built-in training metrics tracking with automatic epoch timing, gradient norms, loss reduction rates, early stopping support, and learning rate scheduling visualization
- **Logging System**: Configurable logging levels for debugging

### Visualization UI

- **Interactive Training Dashboard**: Real-time visualization of training metrics, loss curves, and accuracy plots
- **Computational Graph Visualization**: Visual representation of the computation graph with ops topology
- **Model Architecture View**: Interactive model architecture visualization
- **Bias-Variance Analysis**: Plot training, validation, and test metrics for comprehensive model analysis
- **Early Stopping Visualization**: Visual indicators for early stopping with actual vs expected epochs
- **Learning Rate Scheduling**: Display LR scheduler type and parameters in metrics panel
- **Modern Web Interface**: Sleek, responsive UI built with React and Vite

______________________________________________________________________

## 📋 Requirements

### Core Library

- C11 compatible compiler (GCC, Clang)
- CMake 3.16+ (for CMake build)
- Make (for Makefile build)

### Visualization UI (Optional)

- Python 3.8+
- Node.js 16+ and npm
- FastAPI and Uvicorn (install via `pip install -r requirements.txt`)

______________________________________________________________________

## 🛠️ Installation

### Building the Library

```bash
# Using Make (builds main and examples)
make clean && make

# Using CMake (enable examples)
mkdir build && cd build
cmake -DBUILD_EXAMPLES=ON ..
make -j$(nproc)
```

The build will create:

- `build/main` - Example executable (Makefile)
- `build/examples/*` - Example binaries (Makefile)
- `build/bin/*` - Binaries when using CMake
- `build/libcml.a` - Static library
- `build/libcml.so` - Shared library (if enabled)

### Setting Up the Visualization UI

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies
cd viz-ui
npm install
cd ..
```

______________________________________________________________________

## 🎯 Quick Start

### How VIZ=1 Works

When you set `VIZ=1`, C-ML automatically:

1. **Detects before main()** - A constructor function runs before your program starts
1. **Launches visualization** - Starts FastAPI server (port 8001) and React frontend (port 5173)
1. **Runs your program** - Executes your program with `CML_VIZ=1` set
1. **Exports automatically** - Graph and metrics are exported to JSON files during training
1. **Opens browser** - Automatically opens `http://localhost:5173` for visualization

The visualization UI updates in real-time as your training progresses!

### Basic Training Example

```c
#include "cml.h"
#include <stdio.h>

int main(void) {
    cml_init();

    // Create a simple neural network
    Sequential *model = nn_sequential();
    sequential_add(model, (Module*)nn_linear(2, 4, DTYPE_FLOAT32, DEVICE_CPU, true));
    sequential_add(model, (Module*)nn_relu(false));
    sequential_add(model, (Module*)nn_linear(4, 1, DTYPE_FLOAT32, DEVICE_CPU, true));
    sequential_add(model, (Module*)nn_sigmoid());

    summary((Module*)model);

    Parameter **params;
    int num_params;
    module_collect_parameters((Module*)model, &params, &num_params, true);

    Optimizer *optimizer = optim_adam(params, num_params, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);

    for (int epoch = 0; epoch < 100; epoch++) {
        optimizer_zero_grad(optimizer);

        Tensor *outputs = module_forward((Module*)model, inputs);
        Tensor *loss = tensor_mse_loss(outputs, targets);
        tensor_backward(loss, NULL, false, false);
        optimizer_step(optimizer);

        tensor_free(loss);
        tensor_free(outputs);
    }

    optimizer_free(optimizer);
    CM_FREE(params);
    module_free((Module*)model);
    cml_cleanup();

    return 0;
}
```

### Training with Automatic Metrics Tracking

C-ML automatically captures training metrics without any manual tracking code. Simply use `cml_init()` and `cml_cleanup()`:

```c
#include "cml.h"
#include "Core/cleanup.h"
#include <stdio.h>

int main(void) {
    CleanupContext *cleanup = cleanup_context_create();
    if (!cleanup) return 1;

    cml_init(); // Automatically initializes global metrics tracking

    // Create model and optimizer
    Sequential *model = nn_sequential();
    // ... add layers ...

    cleanup_register_model(cleanup, (Module*)model);
    training_metrics_register_model((Module*)model); // Register for architecture export

    Parameter **params;
    int num_params;
    module_collect_parameters((Module*)model, &params, &num_params, true);
    cleanup_register_params(cleanup, params);

    Optimizer *optimizer = optim_adam(params, num_params, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);
    cleanup_register_optimizer(cleanup, optimizer);

    // Set expected epochs for UI
    training_metrics_set_expected_epochs(100);

    // Training loop - metrics are automatically captured!
    for (int epoch = 0; epoch < 100; epoch++) {
        optimizer_zero_grad(optimizer); // Automatically detects new epoch

        Tensor *outputs = module_forward((Module*)model, X);
        Tensor *loss = tensor_mse_loss(outputs, y);
        tensor_backward(loss, NULL, false, false); // Automatically captures loss
        optimizer_step(optimizer); // Automatically captures LR and gradient norm

        // Capture training accuracy (optional)
        float accuracy = calculate_accuracy(outputs, y);
        training_metrics_auto_capture_train_accuracy(accuracy);

        tensor_free(loss);
        tensor_free(outputs);
    }

    // Metrics are automatically exported to training.json
    // Real-time updates happen during training (when VIZ=1 or CML_VIZ=1)
    // Final export happens on cml_cleanup()

cleanup:
    cleanup_context_free(cleanup); // Centralized cleanup - frees all registered resources
    cml_cleanup(); // Automatically exports final metrics
    return 0;
}
```

**Key Points:**

- **No manual tracking needed** -\*\* Metrics are captured automatically
- **Real-time export** - `training.json` is updated continuously during training (when `VIZ=1` or `CML_VIZ=1`)
- **Centralized cleanup** - `CleanupContext` manages all resources with a single call
- **Automatic epoch detection** - Uses `optimizer_zero_grad()` to detect new epochs

### Launching the Visualization UI

There are two ways to launch the visualization UI:

#### Method 1: Automatic Launch (Recommended)

Set the `VIZ=1` environment variable when running your program:

```bash
# The visualization UI will automatically launch before your program runs
VIZ=1 ./build/main

# Or with examples
VIZ=1 ./build/examples/test
```

This will:

1. Automatically detect `VIZ=1` before your program starts
1. Launch `scripts/viz.py` which starts:
   - FastAPI backend server (port 8001)
   - React frontend dev server (port 5173)
1. Run your program with `CML_VIZ=1` set (enables automatic graph/metrics export)
1. Open your browser to `http://localhost:5173`

#### Method 2: Manual Launch

```bash
# From the project root
python scripts/viz.py <executable> [args...]

# Example
python scripts/viz.py ./build/main
python scripts/viz.py ./build/examples/test
```

The UI will automatically load training metrics from `training.json` and graph data from `graph.json` as they are exported during training.

______________________________________________________________________

## 📊 Visualization UI Features

### Training Results Tab

- **Real-time Metrics**: View training loss, accuracy, learning rate, gradient norms, and more
- **Interactive Charts**: Zoom, pan, and toggle metrics on/off with dynamic x-axis (supports early stopping)
- **Bias-Variance Analysis**: Compare training, validation, and test metrics on the same plots
- **Epoch Timing**: Track training time per epoch and total time
- **Early Stopping Indicators**: Visual badges and icons showing early stopping status (actual vs expected epochs)
- **Learning Rate Scheduling**: Display LR scheduler type (e.g., "StepLR (step_size=30,gamma=0.5)") and parameters in metrics panel
- **Conditional Metrics**: Validation and test metrics only appear when data is available

### Computational Blueprint Tab

- **Ops Topology**: Visualize the computation graph with all operations in a clean vertical layout
- **Model Architecture**: Interactive view of your neural network structure using Cytoscape
- **Vertical Layout**: Clean, readable graph layout optimized for exploration
- **Left-aligned Graphs**: Easy to read and navigate with manual scrolling

### Kernel Studio Tab

- **Coming Soon**: Interactive workspace for kernel optimization and deployment
- **Hanging Slate Animation**: Sleek "coming soon" overlay with animated slate

______________________________________________________________________

## 📁 Project Structure

```
C-ML/
├── include/              # Header files
│   ├── autograd/        # Automatic differentiation
│   ├── Core/            # Core utilities (metrics, logging, etc.)
│   ├── nn/              # Neural network layers
│   ├── optim/           # Optimizers
│   └── tensor/          # Tensor operations
├── src/                 # Implementation files
│   ├── autograd/
│   ├── Core/
│   ├── nn/
│   └── optim/
├── examples/            # Example programs
├── tests/               # Python test suite
├── viz-ui/              # React visualization frontend
│   ├── src/
│   │   ├── components/  # React components
│   │   └── App.jsx      # Main app
│   └── public/          # Static assets
├── scripts/             # Utility scripts
│   ├── viz.py           # Visualization launcher
│   └── fastapi_server.py # Backend API server
├── docs/                # Documentation
├── main.c               # Simple training example
└── Makefile             # Build configuration
```

______________________________________________________________________

## 📚 Documentation

- **[Autograd System](docs/AUTOGRAD.md)** - Complete guide to automatic differentiation
- **[Neural Network Layers](docs/NN_LAYERS.md)** - Available layers and their usage
- **[Training Guide](docs/TRAINING.md)** - Comprehensive guide to training neural networks
- **[Autograd Implementation](docs/AUTOGRAD_IMPLEMENTATION.md)** - Technical implementation details
- **[Layers Implementation](docs/LAYERS_COMPLETE.md)** - Layer implementation details
- **[Integration Summary](docs/INTEGRATION_SUMMARY.md)** - Library integration overview
- **[Documentation Index](docs/INDEX.md)** - Complete documentation index

______________________________________________________________________

## 🧪 Examples

Example programs are available in the `examples/` directory:

- **`main.c`** - Simple XOR classification example
- **`examples/test.c`** - Comprehensive training with train/val/test splits and automatic metrics
- **`examples/early_stopping_lr_scheduler.c`** - Early stopping and learning rate scheduling example
- **`examples/autograd_example.c`** - Autograd system demonstration
- **`examples/training_loop_example.c`** - Full training loop example
- **`examples/export_graph.c`** - Graph export for visualization

Run examples:

```bash
# Build and run main example
make && ./build/main

# Build and run test example
make && ./build/examples/test

# Build and run early stopping example
make && ./build/examples/early_stopping_lr_scheduler

# Build and run specific example
make build/examples/autograd_example
./build/examples/autograd_example

# Run with visualization (automatic launch)
VIZ=1 ./build/main
VIZ=1 ./build/examples/test
VIZ=1 ./build/examples/early_stopping_lr_scheduler
```

______________________________________________________________________

## 🧩 Key Components

### Training Metrics

The `TrainingMetrics` system automatically tracks (no manual code needed):

- **Epoch times** - Automatic timing per epoch and total training time
- **Training/validation/test losses and accuracies** - Tracked per epoch
- **Learning rates** - Current LR per epoch with scheduler information
- **Gradient norms** - L2 norm of gradients for health monitoring
- **Loss reduction rates** - Percentage reduction in loss
- **Loss stability metrics** - Standard deviation of recent losses
- **Early stopping status** - Actual vs expected epochs when early stopping occurs
- **Learning rate history** - Full LR history per epoch for scheduler visualization

All metrics are automatically exported to `training.json` for real-time visualization. The export happens continuously during training when `VIZ=1` or `CML_VIZ=1` is set.

### Centralized Cleanup

The `CleanupContext` system provides centralized resource management:

- **Register resources** - Models, optimizers, tensors, datasets, and raw memory
- **Single cleanup call** - `cleanup_context_free()` frees all registered resources
- **Reduces boilerplate** - No need for individual `CM_FREE()` calls everywhere
- **Prevents memory leaks** - Ensures all resources are properly freed

### Dataset Utilities

- **`dataset_split_three()`** - Split dataset into train/validation/test sets with specified ratios
- **`training_metrics_evaluate_dataset()`** - Automatically evaluate model on dataset and record metrics
- **`dataset_from_arrays()`** - Create dataset from input and target arrays

### Optimizers

- **SGD**: Stochastic Gradient Descent with momentum and weight decay
- **Adam**: Adaptive Moment Estimation with configurable hyperparameters

### Neural Network Layers

- **Linear**: Fully connected layers
- **Conv2d**: 2D Convolutional layers
- **BatchNorm2d**: Batch normalization
- **Pooling**: Max and Average pooling
- **Activations**: ReLU, Sigmoid, Tanh, GELU, Swish
- **Dropout**: Regularization layer
- **LayerNorm**: Layer normalization
- **Sequential**: Container for stacking layers

______________________________________________________________________

## 🧪 Testing

Run the Python test suite:

```bash
pytest tests/
```

______________________________________________________________________

## 📝 License

See [LICENCE.md](LICENCE.md) for license information.

______________________________________________________________________

## 🤝 Contributing

Contributions are welcome! Please ensure your code:

- Follows the existing code style
- Includes appropriate documentation
- Passes all existing tests
- Updates relevant documentation

______________________________________________________________________

## 🐛 Support

For issues, questions, or contributions, please refer to the project repository.

______________________________________________________________________

## 🎯 Roadmap

- [ ] Kernel Studio: Interactive kernel optimization workspace
- [ ] Operator scheduling previews
- [ ] Backend-specific optimizations (CUDA, Metal, etc.)
- [ ] One-click deployment bundles
- [ ] Additional optimizers (RMSprop, AdaGrad, etc.)
- [ ] More layer types (RNN, LSTM, Transformer blocks)
- [ ] Distributed training support

______________________________________________________________________

<p align="center">
  Built with ❤️ in C
</p>
