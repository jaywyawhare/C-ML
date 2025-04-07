# C-ML Documentation

Welcome to the comprehensive documentation for **C-ML**, a lightweight machine learning library written in C. This document provides detailed insights into the library's architecture, components, usage, and development practices.

---

## Table of Contents

- [C-ML Documentation](#c-ml-documentation)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Features](#features)
  - [Directory Structure](#directory-structure)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Steps](#steps)
  - [Usage](#usage)
  - [Modules](#modules)
    - [Layers](#layers)
      - [Dense Layer](#dense-layer)
      - [Dropout Layer](#dropout-layer)
    - [Activations](#activations)
      - [Supported Functions](#supported-functions)
    - [Loss Functions](#loss-functions)
      - [Supported Losses](#supported-losses)
    - [Optimizers](#optimizers)
      - [Supported Optimizers](#supported-optimizers)
    - [Preprocessing](#preprocessing)
      - [Utilities](#utilities)
    - [Regularizers](#regularizers)
      - [Supported Regularizers](#supported-regularizers)
  - [Testing](#testing)
  - [Development Guidelines](#development-guidelines)
    - [Code Style](#code-style)
    - [Error Handling](#error-handling)
    - [Memory Management](#memory-management)
    - [Testing](#testing-1)
  - [Contributing](#contributing)
  - [License](#license)

---

## Introduction

C-ML is a lightweight and modular machine learning library designed for educational purposes and lightweight applications. It provides essential components for building and training neural networks in C, focusing on simplicity, performance, and extensibility.

---

## Features

- **Layers**: Dense, Dropout
- **Activations**: ReLU, Sigmoid, Tanh, Softmax, ELU, Leaky ReLU, Linear
- **Loss Functions**: Mean Squared Error, Binary Cross-Entropy, Focal Loss, Huber Loss, KLD Loss, Log-Cosh Loss, Poisson Loss, Smooth L1 Loss, Tversky Loss, Cosine Similarity Loss, etc.
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

The `main.c` file demonstrates how to use the library to create a simple neural network with a dense layer, ReLU activation, and mean squared error loss.

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "include/Core/training.h"
#include "include/Core/logging.h"

int main()
{
    srand(time(NULL));
    NeuralNetwork *network = create_neural_network(2);

    build_network(network, OPTIMIZER_ADAM, 0.01f, LOSS_MSE, 0.0f, 0.0f);
    model_add(network, LAYER_DENSE, ACTIVATION_RELU, 2, 4, 0.0f, 0, 0);
    model_add(network, LAYER_DENSE, ACTIVATION_TANH, 4, 4, 0.0f, 0, 0);
    model_add(network, LAYER_DENSE, ACTIVATION_SIGMOID, 4, 1, 0.0f, 0, 0);

    int num_samples = 4;
    float **X_train = (float **)cm_safe_malloc(num_samples * sizeof(float *), __FILE__, __LINE__);
    float **y_train = (float **)cm_safe_malloc(num_samples * sizeof(float *), __FILE__, __LINE__);

    for (int i = 0; i < num_samples; i++)
    {
        X_train[i] = (float *)cm_safe_malloc(2 * sizeof(float), __FILE__, __LINE__);
        y_train[i] = (float *)cm_safe_malloc(1 * sizeof(float), __FILE__, __LINE__);
    }

    X_train[0][0] = 0.0f;
    X_train[0][1] = 0.0f;
    y_train[0][0] = 0.0f;
    X_train[1][0] = 0.0f;
    X_train[1][1] = 1.0f;
    y_train[1][0] = 1.0f;

    X_train[2][0] = 1.0f;
    X_train[2][1] = 0.0f;
    y_train[2][0] = 1.0f;

    X_train[3][0] = 1.0f;
    X_train[3][1] = 1.0f;
    y_train[3][0] = 1.0f;

    summary(network);
    train_network(network, X_train, y_train, num_samples, 2, 1, 1, 300);

    MetricType metrics[] = {METRIC_R2_SCORE};

    int num_metrics = sizeof(metrics) / sizeof(metrics[0]);
    float results[num_metrics];

    test_network(network, X_train, y_train, num_samples, 2, 1, (int *)metrics, num_metrics, results);
    LOG_INFO("R2 Score: %.2f", results[0]);

    for (int i = 0; i < num_samples; i++)
    {
        float prediction = 0.0f;
        forward_pass(network, X_train[i], &prediction, 2, 1, 0);
        LOG_INFO(" Input: [%.0f, %.0f], Expected: %.0f, Predicted: %.4f\n",
               X_train[i][0], X_train[i][1], y_train[i][0], prediction);
    }

    free_neural_network(network);

    for (int i = 0; i < num_samples; i++)
    {
        cm_safe_free((void **)&X_train[i]);
        cm_safe_free((void **)&y_train[i]);
    }
    cm_safe_free((void **)&X_train);
    cm_safe_free((void **)&y_train);

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

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## License

This project is licensed under the DBaJ-NC-CFL [License](./LICENCE.md).