#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include "../../include/Activations/linear.h"
#include "../../include/Core/error_codes.h"

void test_linear_activation()
{
    float input[13] = {0.0, 1.0, -1.0, 2.0, -2.0, 1e6, -1e6, FLT_MAX, -FLT_MAX, NAN, INFINITY, -INFINITY, 0.5};
    float expected_output[13] = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 1e6f, -1e6f, FLT_MAX, -FLT_MAX, NAN, CM_INVALID_INPUT_ERROR, CM_INVALID_INPUT_ERROR, 0.5f};
    float tolerance = 1e-6;

    for (int i = 0; i < 13; i++)
    {
        if (isnan(input[i]) || isinf(input[i]))
        {
            assert(CM_INVALID_INPUT_ERROR == linear(input[i]));
        }
        else
        {
            assert(fabs(linear(input[i]) - expected_output[i]) < tolerance);
        }
    }
    printf("linear activation function test passed\n");
}

int main()
{
    printf("Testing linear activation function\n");
    test_linear_activation();
    return CM_SUCCESS;
}