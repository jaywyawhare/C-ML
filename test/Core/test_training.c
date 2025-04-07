#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "../../include/Core/memory_management.h"
#include "../../include/Core/training.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"

void test_large_layer_sizes()
{
    set_log_level(LOG_LEVEL_DEBUG);

    NeuralNetwork *network = create_neural_network(1000);

    model_add(network, LAYER_DENSE, ACTIVATION_RELU, 1000, 2000, 0.0f, 0, 0);
    model_add(network, LAYER_DENSE, ACTIVATION_RELU, 2000, 500, 0.0f, 0, 0);

    float *input = (float *)cm_safe_malloc(1000 * sizeof(float), __FILE__, __LINE__);
    float *output = (float *)cm_safe_malloc(500 * sizeof(float), __FILE__, __LINE__);

    for (int i = 0; i < 1000; i++)
        input[i] = 0.1f;

    CM_Error error = forward_pass(network, input, output, 1000, 500, 0);

    assert(error == CM_SUCCESS);

    free_neural_network(network);
    cm_safe_free((void **)&input);
    cm_safe_free((void **)&output);
}

void test_mismatched_layer_sizes()
{

    NeuralNetwork *network = create_neural_network(10);

    model_add(network, LAYER_DENSE, ACTIVATION_RELU, 10, 20, 0.0f, 0, 0);
    model_add(network, LAYER_DENSE, ACTIVATION_RELU, 20, 5, 0.0f, 0, 0);

    float input[15] = {0};
    float output[5] = {0};

    CM_Error error = forward_pass(network, input, output, 15, 5, 0);

    assert(error == CM_INVALID_LAYER_DIMENSIONS_ERROR);

    free_neural_network(network);
}

void test_optimizer_memory()
{

    NeuralNetwork *network = create_neural_network(10);

    model_add(network, LAYER_DENSE, ACTIVATION_RELU, 10, 100, 0.0f, 0, 0);
    model_add(network, LAYER_DENSE, ACTIVATION_RELU, 100, 50, 0.0f, 0, 0);

    CM_Error error = build_network(network, OPTIMIZER_ADAM, 0.01f, LOSS_MSE, 0.0f, 0.0f);
    assert(error == CM_SUCCESS);

    error = initialize_optimizer_params(network);
    assert(error == CM_SUCCESS);

    free_neural_network(network);
}

int main()
{
    printf("Testing Neural Network Training\n");
    test_large_layer_sizes();
    test_mismatched_layer_sizes();
    test_optimizer_memory();
    return CM_SUCCESS;
}
