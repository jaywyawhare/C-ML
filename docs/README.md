# C-ML Documentation

Welcome to the comprehensive documentation for **C-ML**, a lightweight machine learning library written in C. This document provides detailed insights into the library's architecture, components, usage, and development practices.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Directory Structure](#directory-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Modules](#modules)
   - [Layers](#layers)
   - [Activations](#activations)
   - [Loss Functions](#loss-functions)
   - [Optimizers](#optimizers)
   - [Preprocessing](#preprocessing)
   - [Regularizers](#regularizers)
7. [Testing](#testing)
8. [Development Guidelines](#development-guidelines)
9. [Contributing](#contributing)
10. [License](#license)

---

## Introduction

C-ML is a lightweight and modular machine learning library designed for educational purposes and lightweight applications. It provides essential components for building and training neural networks in C, focusing on simplicity, performance, and extensibility.

---

## Features

- **Layers**: Dense, Dropout
- **Activations**: ReLU, Sigmoid, Tanh, Softmax, ELU, Leaky ReLU, Linear
- **Loss Functions**: Mean Squared Error, Binary Cross-Entropy, Focal Loss, etc.
- **Optimizers**: SGD, Adam, RMSprop
- **Preprocessing**: Label Encoding, One-Hot Encoding, Standard Scaler, Min-Max Scaler
- **Regularizers**: L1, L2, Combined L1-L2
- **Test Coverage**: Comprehensive unit tests for all modules.

---

## Directory Structure

```
C-ML/
├── src/                  # Source files
│   ├── Activations/      # Activation functions
│   ├── Layers/           # Layer implementations
│   ├── Loss_Functions/   # Loss functions
│   ├── Optimizers/       # Optimizer implementations
│   ├── Preprocessing/    # Preprocessing utilities
│   ├── Regularizers/     # Regularization techniques
├── test/                 # Test files
│   ├── Activations/      # Tests for activation functions
│   ├── Layers/           # Tests for layers
│   ├── Loss_Functions/   # Tests for loss functions
│   ├── Optimizers/       # Tests for optimizers
│   ├── Preprocessing/    # Tests for preprocessing utilities
│   ├── Regularizers/     # Tests for regularization techniques
├── docs/                 # Documentation
├── main.c                # Example usage of the library
├── Makefile              # Build system
└── README.md             # Project overview
```

---

## Installation

### Prerequisites

- GCC (GNU Compiler Collection)
- `make` build tool

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/jaywyawhare/C-ML.git
   cd C-ML
   ```

2. Build the project:
   ```bash
   make
   ```

3. Run the example program:
   ```bash
   ./bin/main
   ```

4. Run the tests:
   ```bash
   make test
   ```

---

## Usage

The `main.c` file demonstrates how to use the library to create a simple neural network. Below is an example:

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../src/Layers/dense.h"
#include "../src/Activations/relu.h"
#include "../src/Loss_Functions/meanSquaredError.h"

int main()
{
    float input[] = {1.0, 2.0, 3.0};
    int input_size = 3;

    float target[] = {0.0, 1.0};
    int output_size = 2;

    DenseLayer dense_layer = {NULL, NULL, 0, 0};
    initializeDense(&dense_layer, input_size, output_size);

    float dense_output[2];
    forwardDense(&dense_layer, input, dense_output);

    for (int i = 0; i < output_size; i++)
    {
        dense_output[i] = relu(dense_output[i]);
    }

    float loss = meanSquaredError(target, dense_output, output_size);
    printf("Loss: %f\n", loss);

    freeDense(&dense_layer);
    return 0;
}
```

---

## Modules

### Layers

#### Dense Layer
- Fully connected layer.
- **Functions**:
  - [`initializeDense`](../src/Layers/dense.c): Initializes the layer with random weights and biases.
  - [`forwardDense`](../src/Layers/dense.c): Performs forward propagation.
  - [`backwardDense`](../src/Layers/dense.c): Computes gradients during backpropagation.
  - [`updateDense`](../src/Layers/dense.c): Updates weights and biases using gradients.
  - [`freeDense`](../src/Layers/dense.c): Frees allocated memory.

#### Dropout Layer
- Regularization technique to prevent overfitting.
- **Functions**:
  - [`initializeDropout`](../src/Layers/dropout.c): Initializes the dropout layer.
  - [`forwardDropout`](../src/Layers/dropout.c): Applies dropout during forward propagation.
  - [`backwardDropout`](../src/Layers/dropout.c): Computes gradients during backpropagation.

---

### Activations

#### Supported Functions
- **ReLU**: [`relu(float x)`](../src/Activations/relu.c)
- **Sigmoid**: [`sigmoid(float x)`](../src/Activations/sigmoid.c)
- **Tanh**: [`tanH(float x)`](../src/Activations/tanh.c)
- **Softmax**: [`softmax(float *z, int n)`](../src/Activations/softmax.c)
- **ELU**: [`elu(float x, float alpha)`](../src/Activations/elu.c)
- **Leaky ReLU**: [`leakyRelu(float x)`](../src/Activations/leakyRelu.c)
- **Linear**: [`linear(float x)`](../src/Activations/linear.c)

---

### Loss Functions

#### Supported Losses
- **Mean Squared Error**: [`meanSquaredError(float *y, float *yHat, int n)`](../src/Loss_Functions/meanSquaredError.h)
- **Binary Cross-Entropy**: [`binaryCrossEntropyLoss(float *yHat, float *y, int size)`](../src/Loss_Functions/binaryCrossEntropyLoss.h)
- **Focal Loss**: [`focalLoss(float *y, float *yHat, int n, float gamma)`](../src/Loss_Functions/focalLoss.h)

---

### Optimizers

#### Supported Optimizers
- **SGD**: [`SGD(float x, float y, float lr, float *w, float *b)`](../src/Optimizers/sgd.c)
- **Adam**: [`Adam(float x, float y, float lr, float *w, float *b, ...)`](../src/Optimizers/Adam.c)
- **RMSprop**: [`RMSprop(float x, float y, float lr, float *w, float *b, ...)`](../src/Optimizers/RMSprop.c)

---

### Preprocessing

#### Utilities
- **Standard Scaler**: [`standardScaler(float *x, int size)`](../src/Preprocessing/standardScaler.c)
- **Min-Max Scaler**: [`minMaxScaler(float *x, int size)`](../src/Preprocessing/minMaxScaler.c)
- **Label Encoder**: [`labelEncoder(char *x, int size, ...)`](../src/Preprocessing/labelEncoder.c)
- **One-Hot Encoder**: [`oneHotEncoding(char *x, int size, ...)`](../src/Preprocessing/oneHotEncoder.c)

---

### Regularizers

#### Supported Regularizers
- **L1**: [`l1(float x, float y, ...)`](../src/Regularizers/l1.c)
- **L2**: [`l2(float x, float y, ...)`](../src/Regularizers/l2.c)
- **L1-L2**: [`l1_l2(float *w, float *dw, float l1, float l2, int n)`](../src/Regularizers/l1_l2.c)

---

## Testing

Run the tests using the `make test` command:
```bash
make test
```

Each module has a corresponding test file in the `test/` directory. The tests validate the correctness of the implementation and ensure robustness.

---

## Development Guidelines

### Code Style
- Follow consistent indentation and naming conventions.
- Use meaningful variable names.
- Add comments to explain complex logic.

### Error Handling
- Check for null pointers and invalid inputs.
- Use `fprintf` to log errors and exit gracefully.

### Memory Management
- Allocate memory dynamically where necessary.
- Free allocated memory to prevent leaks.

### Testing
- Write unit tests for all functions.
- Use assertions to validate expected behavior.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes with clear messages.
4. Submit a pull request.

---

## License

This project is licensed under the DBaJ-NC-CFL License. See the [LICENSE](../LICENCE.md) file for details.