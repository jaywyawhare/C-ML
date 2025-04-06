<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="./docs/dark-mode.svg">
  <img alt="C-ML logo" src="./docs/light-mode.svg" width="50%" height="50%">
</picture>

</div>

---

C-ML is a lightweight machine learning library written in C. It provides implementations for various neural network components.

## Features

- **Layers**: Dense, Dropout, Flatten, Pooling, Max-Pooling
- **Activations**: ReLU, Sigmoid, Tanh, Softmax, ELU, Leaky ReLU, Linear
- **Loss Functions**: Mean Squared Error, Binary Cross-Entropy, Focal Loss, etc.
- **Metrics**: Accuracy, Precision, Recall, F1 Score, etc.
- **Optimizers**: SGD, Adam, RMSprop
- **Preprocessing**: Label Encoding, One-Hot Encoding, Standard Scaler, Min-Max Scaler
- **Regularizers**: L1, L2, Combined L1-L2


## Directory Structure

```
C-ML/
├── docs/                 # Documentation files
├── examples/             # Example code and usage
├── include/              # Header files
├── src/                  # Source files
│   ├── Core/             # Core library files
│   ├── Activations/      # Activation functions
│   ├── Layers/           # Layer implementations
│   ├── Loss_Functions/   # Loss functions
│   ├── Optimizers/       # Optimizer implementations
│   ├── Preprocessing/    # Preprocessing utilities
│   ├── Metrics/          # Metric functions
│   └── Regularizers/     # Regularization techniques
├── test/                 # Test files
│   ├── Activations/      # Tests for activation functions
│   ├── Layers/           # Tests for layers
│   ├── Loss_Functions/   # Tests for loss functions
│   ├── Optimizers/       # Tests for optimizers
│   ├── Preprocessing/    # Tests for preprocessing utilities
│   ├── Metrics/          # Tests for metrics
│   └── Regularizers/     # Tests for regularization techniques
├── mkdocs.yml            # Documentation configuration
├── LICENSE.md            # License information
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

5. Run the examples:
   ```bash
   make nn_example
   ```

6. Clean the build artifacts:
   ```bash
   make clean
   ```


## Example Usage

The `main.c` file demonstrates how to use the library to create a simple neural network with a dense layer, ReLU activation, and mean squared error loss.

```c
#include <stdio.h>
#include <stdlib.h>
#include "../include/Core/training.h"
#include "../include/Core/dataset.h"

int main()
{
    NeuralNetwork *network = create_neural_network(2);
    build_network(network, OPTIMIZER_ADAM, 0.1f, LOSS_MSE, 0.0f, 0.0f);
    model_add(network, LAYER_DENSE, ACTIVATION_RELU, 2, 4, 0.0f, 0, 0);
    model_add(network, LAYER_DENSE, ACTIVATION_TANH, 4, 4, 0.0f, 0, 0);
    model_add(network, LAYER_DENSE, ACTIVATION_SIGMOID, 4, 1, 0.0f, 0, 0);

    float X_data[4][2] = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}};

    float y_data[4][1] = {
        {0.0f},
        {1.0f},
        {1.0f},
        {1.0f}};

    Dataset *dataset = dataset_create();
    dataset_load_arrays(dataset, (float *)X_data, (float *)y_data, 4, 2, 1);

    summary(network);

    train_network(network, dataset, 30);
    test_network(network, dataset->X, dataset->y, dataset->num_samples, NULL);

    dataset_free(dataset);
    free_neural_network(network);

    return 0;
}
```


## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.


## License

This project is licensed under the DBaJ-NC-CFL [License](./LICENCE.md).