#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include "../../include/Activations/elu.h"
#include "../../include/Core/error_codes.h"

void test_elu()
{
    float input[11] = {0.0, 1.0, -1.0, 2.0, -2.0, 1e6, -1e6, FLT_MAX, -FLT_MAX, NAN, INFINITY};
    float alpha_values[6] = {0.0, 1.0, FLT_MAX, -FLT_MAX, NAN, INFINITY};
    float tolerance = 1e-6;

    for (int a = 0; a < 6; a++)
    {
        float alpha = alpha_values[a];

        for (int i = 0; i < 11; i++)
        {
            if (isnan(input[i]) || isnan(alpha) || isinf(input[i]) || isinf(alpha))
            {
                assert(CM_INVALID_INPUT_ERROR == elu(input[i], alpha));
            }
            else
            {
                float expected_output;
                if (input[i] >= 0)
                {
                    expected_output = input[i];
                }
                else
                {
                    expected_output = alpha * (expf(input[i]) - 1);
                }

                float output = elu(input[i], alpha);
                assert(fabs(output - expected_output) < tolerance);
            }
        }
    }
    printf("ELU activation function test passed\n");
}

int main()
{
    printf("Testing ELU activation function\n");
    test_elu();
    return CM_SUCCESS;
}