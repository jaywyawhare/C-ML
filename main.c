#include <stdlib.h>
#include <math.h>
#include "include/Layers/dense.h"
#include "include/Layers/flatten.h"
#include "include/Activations/relu.h"
#include "include/Loss_Functions/mean_squared_error.h"
#include "include/Core/error_codes.h"

int main()
{
    float input[] = {1.0, 2.0, 3.0};
    int input_size = 3;

    float target[] = {0.0, 1.0};
    int output_size = 2;

    FlattenLayer flatten_layer = {0, 0};
    if (initialize_flatten(&flatten_layer, input_size) != CM_SUCCESS)
    {
        fprintf(stderr, "Failed to initialize Flatten Layer\n");
        return CM_LAYER_NOT_INITIALIZED_ERROR;
    }

    float flattened_output[3];
    if (forward_flatten(&flatten_layer, input, flattened_output) != CM_SUCCESS)
    {
        fprintf(stderr, "Failed to perform forward pass for Flatten Layer\n");
        return CM_INVALID_LAYER_DIMENSIONS_ERROR;
    }

    DenseLayer dense_layer = {NULL, NULL, 0, 0};
    if (initialize_dense(&dense_layer, input_size, output_size) != CM_SUCCESS)
    {
        fprintf(stderr, "Failed to initialize Dense Layer\n");
        return CM_LAYER_NOT_INITIALIZED_ERROR;
    }

    float dense_output[2];
    if (forward_dense(&dense_layer, flattened_output, dense_output) != CM_SUCCESS)
    {
        fprintf(stderr, "Failed to perform forward pass for Dense Layer\n");
        return CM_INVALID_LAYER_DIMENSIONS_ERROR;
    }

    for (int i = 0; i < output_size; i++)
    {
        dense_output[i] = relu(dense_output[i]);
    }

    float loss = mean_squared_error(target, dense_output, output_size);
    if (loss == CM_INVALID_INPUT_ERROR)
    {
        fprintf(stderr, "Failed to compute Mean Squared Error\n");
        return CM_INVALID_INPUT_ERROR;
    }

    float d_output[2] = {dense_output[0] - target[0], dense_output[1] - target[1]};
    float d_input[3] = {0};
    float d_weights[6] = {0};
    float d_biases[2] = {0};
    if (backward_dense(&dense_layer, flattened_output, dense_output, d_output, d_input, d_weights, d_biases) != CM_SUCCESS)
    {
        fprintf(stderr, "Failed to perform backward pass for Dense Layer\n");
        return CM_INVALID_LAYER_DIMENSIONS_ERROR;
    }

    float learning_rate = 0.01;
    if (update_dense(&dense_layer, d_weights, d_biases, learning_rate) != CM_SUCCESS)
    {
        fprintf(stderr, "Failed to update Dense Layer\n");
        return CM_INVALID_LAYER_DIMENSIONS_ERROR;
    }

    free_dense(&dense_layer);
    free_flatten(&flatten_layer);
    printf("Program completed successfully.\n");
    return CM_SUCCESS;
}