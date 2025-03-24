#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../../include/Activations/gelu.h"
#include "../../include/Core/error_codes.h"

void test_gelu()
{
    float input[11] = {0.0, 1.0, -1.0, 2.0, -2.0, 1e-6, -1e-6, 1e6, -1e6, NAN, INFINITY};
    float expected_output[11] = {0.0f, 0.841192f, -0.158808f, 1.954597f, -0.045402f, 1e-6f, -1e-6f, 1e6f, 0.0f, CM_INVALID_INPUT_ERROR, CM_INVALID_INPUT_ERROR};
    float tolerance = 1e-6;

    for (int i = 0; i < 11; i++)
    {
        if (isnan(input[i]) || isinf(input[i]))
        {
            assert(CM_INVALID_INPUT_ERROR == gelu(input[i]));
        }
        else
        {
            float output = gelu(input[i]);
            assert(fabs(output - expected_output[i]) < tolerance);
        }
    }

    printf("GELU activation function test passed\n");
}

int main()
{
    printf("Testing GELU activation function\n");
    test_gelu();
    return CM_SUCCESS;
}
