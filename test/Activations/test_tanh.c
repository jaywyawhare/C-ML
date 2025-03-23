#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include "../../include/Activations/tanh.h"
#include "../../include/Core/error_codes.h"

void test_tanh()
{
    float input[11] = {0.0, 1.0, -1.0, 2.0, -2.0, 1e6, -1e6, FLT_MAX, -FLT_MAX, NAN, INFINITY};
    float expected_output[11] = {0.0f, 0.76159416f, -0.76159416f, 0.96402758f, -0.96402758f, 1.0f, -1.0f, 1.0f, -1.0f, CM_INVALID_INPUT_ERROR, CM_INVALID_INPUT_ERROR};
    float tolerance = 1e-6;

    for (int i = 0; i < 11; i++)
    {
        if (isnan(input[i]) || isinf(input[i]))
        {
            assert(CM_INVALID_INPUT_ERROR == tanH(input[i]));
        }
        else
        {
            assert(fabs(tanH(input[i]) - expected_output[i]) < tolerance);
        }
    }
    printf("tanh activation function test passed\n");
}

int main()
{
    printf("Testing tanh activation function\n");
    test_tanh();
    return 0;
}
