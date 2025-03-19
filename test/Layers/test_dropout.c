#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "../../src/my_functions.h"

void test_dropout_layer()
{
    DropoutLayer layer;
    initializeDropout(&layer, 0.5);

    assert(layer.dropout_rate == 0.5);

    float input[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float output[5];
    forwardDropout(&layer, input, output, 5);
    printf("Forward pass output: ");
    for (int i = 0; i < 5; i++)
    {
        printf("%f ", output[i]);
    }
    printf("\n");

    float d_output[] = {0.1, 0.2, 0.3, 0.4, 0.5};
    float d_input[5] = {0};
    backwardDropout(&layer, input, output, d_output, d_input, 5);
    printf("Backward pass gradients: ");
    for (int i = 0; i < 5; i++)
    {
        printf("%f ", d_input[i]);
    }
    printf("\n");

    printf("Dropout layer test passed\n");
}

int main()
{
    printf("Testing DropoutLayer\n");
    test_dropout_layer();
    return 0;
}
