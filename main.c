#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "src/my_functions.h"

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
