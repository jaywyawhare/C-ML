#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../include/Core/training.h"

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
    printf("[main] Info: R2 Score: %.2f\n", results[0]);

    for (int i = 0; i < num_samples; i++)
    {
        float prediction = 0.0f;
        forward_pass(network, X_train[i], &prediction, 2, 1, 0);
        printf("[main] Info: Input: [%.0f, %.0f], Expected: %.0f, Predicted: %.4f\n",
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
