#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/Layers/dense.h"
#include "../include/Activations/sigmoid.h"
#include "../include/Loss_Functions/mean_squared_error.h"
#include "../include/Core/error_codes.h"

#define INPUT_SIZE 2
#define HIDDEN_SIZE 1
#define OUTPUT_SIZE 1
#define LEARNING_RATE 0.1
#define EPOCHS 1000
#define TRAINING_SAMPLES 4

int main() {
    float training_data[TRAINING_SAMPLES][INPUT_SIZE] = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    float training_labels[TRAINING_SAMPLES][OUTPUT_SIZE] = {
        {0.0},
        {1.0},
        {1.0},
        {1.0}
    };

    DenseLayer hidden_layer = {NULL, NULL, 0, 0};
    initialize_dense(&hidden_layer, INPUT_SIZE, HIDDEN_SIZE);

    DenseLayer output_layer = {NULL, NULL, 0, 0};
    initialize_dense(&output_layer, HIDDEN_SIZE, OUTPUT_SIZE);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0;

        for (int i = 0; i < TRAINING_SAMPLES; i++) {
            float hidden_layer_input[INPUT_SIZE] = {training_data[i][0], training_data[i][1]};
            float hidden_layer_output[HIDDEN_SIZE];
            forward_dense(&hidden_layer, hidden_layer_input, hidden_layer_output);

            for (int j = 0; j < HIDDEN_SIZE; j++) {
                hidden_layer_output[j] = sigmoid(hidden_layer_output[j]);
            }

            float output_layer_output[OUTPUT_SIZE];
            forward_dense(&output_layer, hidden_layer_output, output_layer_output);

            for (int j = 0; j < OUTPUT_SIZE; j++) {
                output_layer_output[j] = sigmoid(output_layer_output[j]);
            }

            float loss = mean_squared_error(training_labels[i], output_layer_output, OUTPUT_SIZE);
            total_loss += loss;

            float d_output[OUTPUT_SIZE] = {output_layer_output[0] - training_labels[i][0]};
            float d_hidden[HIDDEN_SIZE] = {0.0};
            float d_output_weights[HIDDEN_SIZE * OUTPUT_SIZE] = {0.0};
            float d_output_biases[OUTPUT_SIZE] = {0.0};

            backward_dense(&output_layer, hidden_layer_output, output_layer_output, d_output, d_hidden, d_output_weights, d_output_biases);

            float d_hidden_input[INPUT_SIZE] = {0.0};
            float d_hidden_weights[INPUT_SIZE * HIDDEN_SIZE] = {0.0};
            float d_hidden_biases[HIDDEN_SIZE] = {0.0};

            for (int j = 0; j < HIDDEN_SIZE; j++) {
                float sigmoid_derivative = hidden_layer_output[j] * (1 - hidden_layer_output[j]);
                d_hidden[j] *= sigmoid_derivative;
            }

            backward_dense(&hidden_layer, hidden_layer_input, hidden_layer_output, d_hidden, d_hidden_input, d_hidden_weights, d_hidden_biases);

            update_dense(&output_layer, d_output_weights, d_output_biases, LEARNING_RATE);
            update_dense(&hidden_layer, d_hidden_weights, d_hidden_biases, LEARNING_RATE);
        }

        printf("Epoch %d, Loss: %f\n", epoch, total_loss / TRAINING_SAMPLES);
    }

    printf("\nTesting the trained network:\n");
    for (int i = 0; i < TRAINING_SAMPLES; i++) {
        float input[INPUT_SIZE] = {training_data[i][0], training_data[i][1]};

        float hidden_output[HIDDEN_SIZE];
        forward_dense(&hidden_layer, input, hidden_output);
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            hidden_output[j] = sigmoid(hidden_output[j]);
        }

        float output[OUTPUT_SIZE];
        forward_dense(&output_layer, hidden_output, output);
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            output[j] = sigmoid(output[j]);
        }

        printf("Input: %f %f, Output: %f, Expected: %f\n", input[0], input[1], output[0], training_labels[i][0]);
    }

    free_dense(&hidden_layer);
    free_dense(&output_layer);

    return CM_SUCCESS;
}
