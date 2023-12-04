#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "my_functions.h"

void trainRegressionModel(float **X_train, float *y_train, int num_samples, int num_features) {
    DenseLayer denseLayer;
    initializeDense(&denseLayer, num_features, 1);

    float learning_rate = 0.01;
    int epochs = 100;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float loss = 0.0;

        for (int i = 0; i < num_samples; ++i) {
            float output;
            forwardDense(&denseLayer, X_train[i], &output);

            float error = output - y_train[i];
            loss += (error * error);

            float d_output = 2.0 * error;
            float d_input[num_features];
            float d_weights[num_features];
            float d_biases[1];

            backwardDense(&denseLayer, X_train[i], &output, &d_output, d_input, d_weights, d_biases);

            updateDense(&denseLayer, d_weights, d_biases, learning_rate);
        }

        loss /= num_samples;
        printf("Epoch %d - Loss: %f\n", epoch + 1, loss);
    }

    freeDense(&denseLayer);
}

void createDummyDataset(float **X, float *y, int num_samples, int num_features) {
    srand(time(NULL));

    for (int i = 0; i < num_samples; ++i) {
        X[i] = (float *)malloc(num_features * sizeof(float));

        for (int j = 0; j < num_features; ++j) {
            X[i][j] = (float)(rand() % 100) / 100.0;
        }

        y[i] = 2 * X[i][0] + 1 * X[i][1] - 0.5 * X[i][2] + 0.7 * X[i][3] + 3.0;
    }
}

void freeDataset(float **X, int num_samples) {
    for (int i = 0; i < num_samples; ++i) {
        free(X[i]);
    }
    free(X);
}

int main() {
    int num_samples = 100;
    int num_features = 5;

    float **X = (float **)malloc(num_samples * sizeof(float *));
    float *y = (float *)malloc(num_samples * sizeof(float));

    createDummyDataset(X, y, num_samples, num_features);

    printf("Generated Dataset:\n");
    for (int i = 0; i < num_samples; ++i) {
        printf("Sample %d: [", i + 1);
        for (int j = 0; j < num_features; ++j) {
            printf("%.2f", X[i][j]);
            if (j != num_features - 1) {
                printf(", ");
            }
        }
        printf("] -> Target: %.2f\n", y[i]);
    }

    trainRegressionModel(X, y, num_samples, num_features);

    freeDataset(X, num_samples);
    free(y);

    return 0;
}
