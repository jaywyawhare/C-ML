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
├── main.c                # Example usage of the library
└── Makefile              # Build system
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

4. Clean the build artifacts:
   ```bash
   make clean
   ```


## Example Usage

The `main.c` file demonstrates how to use the library to create a simple neural network with a dense layer, ReLU activation, and mean squared error loss.

```c
#include <stdio.h>
#include <stdlib.h>
#include "src/my_functions.h"

int main() {
    float input[] = {1.0, 2.0, 3.0};
    float target[] = {0.0, 1.0};
    DenseLayer dense_layer = {NULL, NULL, 0, 0};

    initializeDense(&dense_layer, 3, 2);
    float output[2];
    forwardDense(&dense_layer, input, output);

    printf("Output: [%f, %f]\n", output[0], output[1]);
    freeDense(&dense_layer);

    return 0;
}
```


## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.


## License

This project is licensed under the DBaJ-NC-CFL [License](./LICENCE.md).