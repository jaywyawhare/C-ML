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
#include <time.h>
#include "include/Core/training.h"

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
    printf("R2 Score: %.2f\n", results[0]);

    for (int i = 0; i < num_samples; i++)
    {
        float prediction = 0.0f;
        forward_pass(network, X_train[i], &prediction, 2, 1, 0);
        printf("Input: [%.0f, %.0f], Expected: %.0f, Predicted: %.4f\n",
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


## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.


## License

This project is licensed under the DBaJ-NC-CFL [License](./LICENCE.md).