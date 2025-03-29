# Core Modules

This section documents the core modules of the C-ML library, which provide essential functionalities for building and training neural networks.

## Training Module

- **Description**: Provides functions for creating, building, training, and evaluating neural networks.
- **Functions**:
    - `NeuralNetwork *create_neural_network(int input_size)`: Creates a new neural network.
    - `CM_Error build_network(NeuralNetwork *network, OptimizerType optimizer_type, float learning_rate, int loss_function, float l1_lambda, float l2_lambda)`: Builds the neural network by setting the optimizer, loss function, and regularization parameters.
    - `CM_Error add_layer(NeuralNetwork *network, LayerConfig config)`: Adds a layer to the neural network.
    - `CM_Error model_add(NeuralNetwork *network, LayerType type, ActivationType activation, int input_size, int output_size, float rate, int kernel_size, int stride)`: Adds a layer to the neural network using a simplified interface.
    - `CM_Error forward_pass(NeuralNetwork *network, float *input, float *output, int input_size, int output_size, int is_training)`: Performs a forward pass through the network.
    - `float calculate_loss(float *predicted, float *actual, int size, LossType loss_type)`: Calculates the loss between predicted and actual values.
    - `void calculate_loss_gradient(float *predicted, float *actual, float *gradient, int size, LossType loss_type)`: Calculates the gradient of the loss function.
    - `CM_Error train_network(NeuralNetwork *network, float **X_train, float **y_train, int num_samples, int input_size, int output_size, int batch_size, int epochs)`: Trains the neural network.
    - `CM_Error evaluate_network(NeuralNetwork *network, float **X_test, float **y_test, int num_samples, int input_size, int output_size, int *metrics, int num_metrics, float *results)`: Evaluates the neural network on a given dataset.
    - `CM_Error test_network(NeuralNetwork *network, float **X_test, float **y_test, int num_samples, int input_size, int output_size, int *metrics, int num_metrics, float *results)`: Tests the neural network (alias for evaluate_network).
    - `CM_Error free_neural_network(NeuralNetwork *network)`: Frees memory allocated for the neural network.
    - `void summary(NeuralNetwork *network)`: Prints a summary of the neural network architecture.
- **File**: [`training.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Core/training.c)

## Memory Management Module

- **Description**: Provides safe memory allocation and deallocation functions.
- **Functions**:
    - `void *cm_safe_malloc(size_t size, const char *file, int line)`: Allocates memory safely and logs the file and line number in case of failure.
    - `void cm_safe_free(void **ptr)`: Frees allocated memory safely and sets the pointer to NULL.
- **File**: [`memory_management.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Core/memory_management.c)
