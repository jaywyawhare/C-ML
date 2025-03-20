#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "../../src/Layers/dense.h"

void test_dense_layer()
{
    DenseLayer layer = {NULL, NULL, 0, 0};

    initializeDense(&layer, 3, 2);

    assert(layer.weights != NULL);
    assert(layer.biases != NULL);

    float input[] = {1.0, 2.0, 3.0};
    float output[2];
    forwardDense(&layer, input, output);

    assert(output[0] != 0.0f);
    assert(output[1] != 0.0f);

    float d_output[] = {0.1, 0.2};
    float d_input[3] = {0};
    float d_weights[6] = {0};
    float d_biases[2] = {0};
    backwardDense(&layer, input, output, d_output, d_input, d_weights, d_biases);

    for (int i = 0; i < 6; i++)
    {
        assert(d_weights[i] != 0.0f);
    }
    for (int i = 0; i < 2; i++)
    {
        assert(d_biases[i] != 0.0f);
    }

    updateDense(&layer, d_weights, d_biases, 0.01);

    freeDense(&layer);
    printf("Dense layer test passed\n");
}

int main()
{
    printf("Testing DenseLayer\n");
    test_dense_layer();
    return 0;
}
