#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include "../../include/Activations/leaky_relu.h"
#include "../../include/Core/error_codes.h"

void test_leakyRelu()
{
    float input[11] = {0.0, 1.0, -1.0, 2.0, -2.0, 1e6, -1e6, FLT_MAX, -FLT_MAX, NAN, INFINITY};
    float expected_output[11] = {0.0f, 1.0f, -0.01f, 2.0f, -0.02f, 1e6f, -1e4f, FLT_MAX, -FLT_MAX * 0.01f, CM_INVALID_INPUT_ERROR, CM_INVALID_INPUT_ERROR};
    float tolerance = 1e-6;

    for (int i = 0; i < 11; i++)
    {
        if (isnan(input[i]) || isinf(input[i]))
        {
            assert(CM_INVALID_INPUT_ERROR == leaky_relu(input[i]));
        }
        else
        {
            assert(fabs(leaky_relu(input[i]) - expected_output[i]) < tolerance);
        }
    }
    printf("leakyRelu activation function test passed\n");
}

int main()
{
    printf("Testing leakyRelu activation function\n");
    test_leakyRelu();
    return 0;
}