# C-ML: A Lightweight Machine Learning Library in C

C-ML is a lightweight machine learning library written in C. It provides implementations for layers, activations, optimizers, preprocessing, and loss functions, enabling you to build and train simple neural networks.

## Features

- **Layers**: Dense, Dropout
- **Activations**: ReLU, Sigmoid, Tanh, Softmax, ELU, Leaky ReLU, Linear
- **Loss Functions**: Mean Squared Error, Binary Cross-Entropy, Focal Loss, etc.
- **Optimizers**: SGD, Adam, RMSprop
- **Preprocessing**: Label Encoding, One-Hot Encoding, Standard Scaler, Min-Max Scaler
- **Regularizers**: L1, L2, Combined L1-L2

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
│   ├── my_functions.h    # Header file with function declarations
├── test/                 # Test files
│   ├── Activations/      # Tests for activation functions
│   ├── Layers/           # Tests for layers
│   ├── Loss_Functions/   # Tests for loss functions
│   ├── Optimizers/       # Tests for optimizers
│   ├── Preprocessing/    # Tests for preprocessing utilities
│   ├── Regularizers/     # Tests for regularization techniques
├── main.c                # Example usage of the library
├── Makefile              # Build system
└── README.md             # Documentation
```

## Prerequisites

- GCC (GNU Compiler Collection)
- `make` build tool

## Build Instructions

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

5. Clean the build artifacts:
   ```bash
   make clean
   ```

## Example Usage

The `main.c` file demonstrates how to use the library to create a simple neural network with a dense layer, ReLU activation, and mean squared error loss.

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "src/Layers/dense.h"
#include "src/Activations/relu.h"
#include "src/Loss_Functions/meanSquaredError.h"

int main()
{
    printf("Starting program...\n");
    float input[] = {1.0, 2.0, 3.0};
    int input_size = 3;

    float target[] = {0.0, 1.0};
    int output_size = 2;

    DenseLayer dense_layer = {NULL, NULL, 0, 0};
    initializeDense(&dense_layer, input_size, output_size);

    float dense_output[2];
    forwardDense(&dense_layer, input, dense_output);
    printf("Dense Layer Output: [%f, %f]\n", dense_output[0], dense_output[1]);

    for (int i = 0; i < output_size; i++)
    {
        dense_output[i] = relu(dense_output[i]);
    }
    printf("Activated Output: [%f, %f]\n", dense_output[0], dense_output[1]);

    float loss = meanSquaredError(target, dense_output, output_size);
    printf("Loss: %f\n", loss);

    float d_output[2] = {dense_output[0] - target[0], dense_output[1] - target[1]};
    float d_input[3] = {0};
    float d_weights[6] = {0};
    float d_biases[2] = {0};
    backwardDense(&dense_layer, input, dense_output, d_output, d_input, d_weights, d_biases);

    float learning_rate = 0.01;
    updateDense(&dense_layer, d_weights, d_biases, learning_rate);

    freeDense(&dense_layer);

    printf("Program completed successfully.\n");
    return 0;
}
```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the DBaJ-NC-CFL [License](./LICENCE.md).