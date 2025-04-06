#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "../../include/Core/memory_management.h"
#include "../../include/Core/training.h"
#include "../../include/Core/error_codes.h"

void test_large_layer_sizes() {
    // Create a network with very large layer sizes
    NeuralNetwork *network = create_neural_network(1000);
    
    // Add layers with increasing sizes to test buffer allocation
    model_add(network, LAYER_DENSE, ACTIVATION_RELU, 1000, 2000, 0.0f, 0, 0);
    model_add(network, LAYER_DENSE, ACTIVATION_RELU, 2000, 500, 0.0f, 0, 0);
    
    // Test forward pass with large input
    float *input = (float *)cm_safe_malloc(1000 * sizeof(float), __FILE__, __LINE__);
    float *output = (float *)cm_safe_malloc(500 * sizeof(float), __FILE__, __LINE__);
    
    // Fill input with test data
    for (int i = 0; i < 1000; i++) input[i] = 0.1f;
    
    // This would segfault without your fixes
    CM_Error error = forward_pass(network, input, output, 1000, 500, 0);
    
    // Assert no error
    assert(error == CM_SUCCESS);
    
    // Clean up
    free_neural_network(network);
    cm_safe_free((void **)&input);
    cm_safe_free((void **)&output);
}

void test_mismatched_layer_sizes() {
    // Create network with input size 10
    NeuralNetwork *network = create_neural_network(10);
    
    // Add layers with different input/output sizes
    model_add(network, LAYER_DENSE, ACTIVATION_RELU, 10, 20, 0.0f, 0, 0);
    model_add(network, LAYER_DENSE, ACTIVATION_RELU, 20, 5, 0.0f, 0, 0);
    
    // Try to run forward pass with input size 15 (mismatch)
    float input[15] = {0};
    float output[5] = {0};
    
    // This should return an error code, not segfault
    CM_Error error = forward_pass(network, input, output, 15, 5, 0);
    
    // Assert proper error handling
    assert(error == CM_INVALID_LAYER_DIMENSIONS_ERROR);
    
    free_neural_network(network);
}

void test_optimizer_memory() {
    // Create network
    NeuralNetwork *network = create_neural_network(10);
    
    // Add layers
    model_add(network, LAYER_DENSE, ACTIVATION_RELU, 10, 100, 0.0f, 0, 0);
    model_add(network, LAYER_DENSE, ACTIVATION_RELU, 100, 50, 0.0f, 0, 0);
    
    // Build with Adam optimizer (requires memory allocation)
    CM_Error error = build_network(network, OPTIMIZER_ADAM, 0.01f, LOSS_MSE, 0.0f, 0.0f);
    assert(error == CM_SUCCESS);
    
    // Initialize optimizer params (would segfault without your fixes)
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
