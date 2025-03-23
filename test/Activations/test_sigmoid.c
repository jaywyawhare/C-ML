#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include "../../include/Activations/sigmoid.h"
#include "../../include/Core/error_codes.h"

void test_sigmoid()
{
    float input[11] = {0.0, 1.0, -1.0, 2.0, -2.0, 1e6, -1e6, FLT_MAX, -FLT_MAX, NAN, INFINITY};
    float expected_output[11] = {0.5f, 0.73105858f, 0.26894142f, 0.88079708f, 0.11920292f, 1.0f, 0.0f, 1.0f, 0.0f, NAN, 1.0f};
    float tolerance = 1e-6;
    for (int i = 0; i < 11; i++)
    {
        if (isnan(input[i]) || isinf(input[i]))
        {
            assert(CM_INVALID_INPUT_ERROR == sigmoid(input[i]));
        }
        else
        {
            assert(fabs(sigmoid(input[i]) - expected_output[i]) < tolerance);
        }
    }
    printf("sigmoid activation function test passed\n");
}

int main()
{
    printf("Testing sigmoid activation function\n");
    test_sigmoid();
    return 0;
}
