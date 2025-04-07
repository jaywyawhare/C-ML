#include <stdio.h>
#include <stdlib.h>
#include "include/Core/training.h"
#include "include/Core/dataset.h"
#include "include/Core/logging.h"

int main()
{
    set_log_level(LOG_LEVEL_INFO);
    
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