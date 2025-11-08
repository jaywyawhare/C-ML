# C-ML Documentation Index

Welcome to the C-ML documentation. This index provides an overview of all available documentation.

## Getting Started

- **[README](../README.md)** - Project overview and quick start
- **[Training Guide](TRAINING.md)** - Start here to learn how to train neural networks with automatic metrics tracking
- **[Visualization UI](../README.md#visualization-ui-features)** - Interactive training dashboard and model visualization

## Core Documentation

### Automatic Differentiation

- **[Autograd System](AUTOGRAD.md)** - Complete guide to automatic differentiation

  - Overview of the autograd system
  - API reference for gradient computation
  - Usage examples and best practices
  - Technical implementation details

- **[Autograd Implementation](AUTOGRAD_IMPLEMENTATION.md)** - Technical deep dive

  - Detailed implementation notes
  - Design decisions and architecture
  - Performance considerations
  - Advanced features

### Neural Network Layers

- **[Neural Network Layers](NN_LAYERS.md)** - Complete layer reference

  - Available layers (Linear, Conv2d, BatchNorm2d, LayerNorm, Pooling, etc.)
  - Layer usage examples
  - API reference
  - Implementation status

- **[Layers Implementation](LAYERS_COMPLETE.md)** - Implementation details

  - Layer implementation status
  - Testing recommendations
  - Feature completeness

### Training

- **[Training Guide](TRAINING.md)** - Comprehensive training guide

  - Model definition
  - Parameter collection
  - Optimizer usage
  - Loss functions
  - Complete training loop examples
  - **Training Metrics** - Automatic tracking of training, validation, and test metrics
  - **Visualization** - Export metrics to JSON for interactive visualization
  - Best practices

### Training Metrics

The C-ML library includes built-in training metrics tracking that works automatically:

- **Automatic Epoch Timing** - Tracks time per epoch and total training time (no manual code needed)
- **Loss and Accuracy Tracking** - Automatically records training, validation, and test metrics
- **Gradient Norm Monitoring** - Tracks gradient health during training
- **Learning Rate Tracking** - Monitors learning rate changes and scheduler information
- **Loss Reduction Rate** - Computes percentage reduction in loss
- **Loss Stability** - Calculates standard deviation of recent losses
- **Early Stopping Support** - Tracks early stopping status (actual vs expected epochs)
- **LR Scheduler Visualization** - Displays scheduler type and parameters in UI
- **Real-time JSON Export** - Continuously exports metrics to `training.json` for visualization (when `VIZ=1` or `CML_VIZ=1`)

All metrics are captured automatically when using `cml_init()` and `cml_cleanup()`. See [Training Guide](TRAINING.md#training-metrics) for detailed usage.

### Visualization UI

C-ML includes an interactive web-based visualization UI:

- **Training Results Dashboard** - Real-time visualization of training metrics
- **Computational Graph Visualization** - Visual representation of ops topology
- **Model Architecture View** - Interactive model structure visualization using Cytoscape
- **Bias-Variance Analysis** - Plot training, validation, and test metrics together
- **Early Stopping Visualization** - Visual indicators for early stopping with actual vs expected epochs
- **LR Scheduler Display** - Show scheduler type and parameters in metrics panel
- **Automatic Launch** - Set `VIZ=1` to automatically launch before program runs

See [README](../README.md#visualization-ui-features) for setup and usage.

### Integration

- **[Integration Summary](INTEGRATION_SUMMARY.md)** - Library integration overview
  - Build system integration
  - Dependency management
  - Component integration
  - Training metrics integration

### Development

- **[Implementation Status](IMPLEMENTATION_STATUS.md)** - Current implementation status
  - Completed features
  - Recently added features
  - Implementation notes
- **[TODO Implementations](TODO_IMPLEMENTATIONS.md)** - Planned features and improvements
  - High priority items
  - Medium priority items
  - Low priority items
  - Performance optimizations

## Documentation Structure

```
docs/
├── INDEX.md                      # This file
├── AUTOGRAD.md                   # Autograd system guide
├── AUTOGRAD_IMPLEMENTATION.md   # Autograd technical details
├── NN_LAYERS.md                  # Neural network layers reference
├── LAYERS_COMPLETE.md            # Layer implementation details
├── TRAINING.md                   # Training guide (with metrics)
├── INTEGRATION_SUMMARY.md       # Integration overview
├── IMPLEMENTATION_STATUS.md     # Implementation status
└── TODO_IMPLEMENTATIONS.md       # Future work
```

## Quick Links

### For Beginners

1. Start with [README](../README.md) for an overview
1. Read [Training Guide](TRAINING.md) for a complete example with metrics
1. Explore [Neural Network Layers](NN_LAYERS.md) for available layers
1. Try the [Visualization UI](../README.md#launching-the-visualization-ui) to see your training progress

### For Advanced Users

1. Review [Autograd Implementation](AUTOGRAD_IMPLEMENTATION.md) for technical details
1. Check [Layers Implementation](LAYERS_COMPLETE.md) for implementation status
1. See [Implementation Status](IMPLEMENTATION_STATUS.md) for current features
1. Use [Training Metrics](../README.md#training-with-metrics-tracking) for comprehensive monitoring

### For Contributors

1. Review [Integration Summary](INTEGRATION_SUMMARY.md) for build system details
1. Check [TODO Implementations](TODO_IMPLEMENTATIONS.md) for planned work
1. See [Implementation Status](IMPLEMENTATION_STATUS.md) for current state
1. Follow contribution guidelines in [README](../README.md#contributing)

## Examples

Example code is available in the `examples/` directory and root:

- `main.c` - Simple XOR classification example
- `examples/test.c` - Comprehensive training with train/val/test splits and automatic metrics
- `examples/early_stopping_lr_scheduler.c` - Early stopping and learning rate scheduling example
- `examples/autograd_example.c` - Autograd system examples
- `examples/training_loop_example.c` - Training loop pattern demonstration
- `examples/export_graph.c` - Graph export for visualization

## API Reference

For detailed API documentation, see:

- `include/cml.h` - Main library header with inline documentation
- `include/tensor/` - Tensor operations
- `include/autograd/` - Automatic differentiation
- `include/nn/` - Neural network components
- `include/optim/` - Optimizers
- `include/Core/training_metrics.h` - Training metrics API

All header files include comprehensive inline documentation using Doxygen-style comments.

## Key Features

### Core Library

- Automatic differentiation (autograd)
- Neural network layers
- Optimizers (SGD, Adam)
- Loss functions
- Tensor operations

### Training Utilities

- **Training Metrics** - Built-in automatic metrics tracking (no manual code needed)
- **Automatic Timing** - Epoch time calculation
- **Gradient Monitoring** - Gradient norm tracking
- **Early Stopping** - Track early stopping status (actual vs expected epochs)
- **LR Scheduling** - Display scheduler type and parameters in UI
- **Real-time JSON Export** - Continuously export metrics for visualization (when `VIZ=1` or `CML_VIZ=1`)
- **Centralized Cleanup** - `CleanupContext` for resource management
- **Dataset Splitting** - `dataset_split_three()` for train/val/test splits
- **Automatic Evaluation** - `training_metrics_evaluate_dataset()` for validation/test evaluation

### Visualization

- **Interactive Dashboard** - Real-time training visualization
- **Graph Visualization** - Computational graph and model architecture
- **Bias-Variance Analysis** - Training/validation/test metrics comparison

## Support

For issues, questions, or contributions, please refer to the project repository.
