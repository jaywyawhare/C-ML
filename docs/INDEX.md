# C-ML Documentation Index

Welcome to the C-ML documentation. This index provides an overview of all available documentation.

## Getting Started

- **[README](index.md)** - Project overview and quick start
- **[Training Guide](TRAINING.md)** - Start here to learn how to train neural networks

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

  - Available layers (Linear, Conv2d, BatchNorm2d, etc.)
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
  - Best practices

### Integration

- **[Integration Summary](INTEGRATION_SUMMARY.md)** - Library integration overview
  - Build system integration
  - Dependency management
  - Component integration

### Development

- **[Implementation Status](IMPLEMENTATION_STATUS.md)** - Current implementation status
- **[TODO Implementations](TODO_IMPLEMENTATIONS.md)** - Planned features and improvements

## Documentation Structure

```
docs/
├── INDEX.md                      # This file
├── AUTOGRAD.md                   # Autograd system guide
├── AUTOGRAD_IMPLEMENTATION.md    # Autograd technical details
├── NN_LAYERS.md                  # Neural network layers reference
├── LAYERS_COMPLETE.md            # Layer implementation details
├── TRAINING.md                   # Training guide
├── INTEGRATION_SUMMARY.md        # Integration overview
├── IMPLEMENTATION_STATUS.md      # Implementation status
└── TODO_IMPLEMENTATIONS.md       # Future work
```

## Quick Links

### For Beginners

1. Start with [README](index.md) for an overview
1. Read [Training Guide](TRAINING.md) for a complete example
1. Explore [Neural Network Layers](NN_LAYERS.md) for available layers

### For Advanced Users

1. Review [Autograd Implementation](AUTOGRAD_IMPLEMENTATION.md) for technical details
1. Check [Layers Implementation](LAYERS_COMPLETE.md) for implementation status
1. See [Implementation Status](IMPLEMENTATION_STATUS.md) for current features

### For Contributors

1. Review [Integration Summary](INTEGRATION_SUMMARY.md) for build system details
1. Check [TODO Implementations](TODO_IMPLEMENTATIONS.md) for planned work
1. See [Implementation Status](IMPLEMENTATION_STATUS.md) for current state

## Examples

Example code is available in the `examples/` directory:

- `main.c` - Complete training example
- `training_example.c` - Training pattern demonstration
- `autograd_example.c` - Autograd system examples

## API Reference

For detailed API documentation, see:

- `include/cml.h` - Main library header with inline documentation
- `include/tensor/` - Tensor operations
- `include/autograd/` - Automatic differentiation
- `include/nn/` - Neural network components
- `include/optim/` - Optimizers

All header files include comprehensive inline documentation using Doxygen-style comments.
