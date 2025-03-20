#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "../../src/Layers/dropout.h"

void test_dropout_layer()
{
    DropoutLayer layer;
    initializeDropout(&layer, 0.5);

    assert(layer.dropout_rate == 0.5);

    float input[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float output[5];
    forwardDropout(&layer, input, output, 5);

    for (int i = 0; i < 5; i++)
    {
        assert(output[i] == 0.0f || output[i] == input[i] / (1 - layer.dropout_rate));
    }

    float d_output[] = {0.1, 0.2, 0.3, 0.4, 0.5};
    float d_input[5] = {0};
    backwardDropout(&layer, input, output, d_output, d_input, 5);

    for (int i = 0; i < 5; i++)
    {
        if (output[i] == 0.0f)
        {
            assert(d_input[i] == 0.0f);
        }
        else
        {
            assert(fabs(d_input[i] - d_output[i] / (1 - layer.dropout_rate)) < 1e-6);
        }
    }

    printf("Dropout layer test passed\n");
}

int main()
{
    printf("Testing DropoutLayer\n");
    test_dropout_layer();
    return 0;
}
