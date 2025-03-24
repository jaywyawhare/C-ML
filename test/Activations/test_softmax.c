#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include "../../include/Activations/softmax.h"
#include "../../include/Core/error_codes.h"

void test_softmax()
{
    float input[5][3] = {
        {0.0, 0.0, 0.0},
        {1.0, 2.0, 3.0},
        {-1.0, -2.0, -3.0},
        {1000.0, 1001.0, 1002.0},
        {INFINITY, 0.0, -INFINITY}};
    float expected_output[5][3] = {
        {0.33333333f, 0.33333333f, 0.33333333f},
        {0.09003057f, 0.24472847f, 0.66524096f},
        {0.66524096f, 0.24472847f, 0.09003057f},
        {0.09003057f, 0.24472847f, 0.66524096f},
        {CM_INVALID_INPUT_ERROR, CM_INVALID_INPUT_ERROR, CM_INVALID_INPUT_ERROR}};
    float tolerance = 1e-6;

    for (int i = 0; i < 5; i++)
    {
        float *output = softmax(input[i], 3);
        if (isinf(input[i][0]) || isinf(input[i][1]) || isinf(input[i][2]) ||
            isnan(input[i][0]) || isnan(input[i][1]) || isnan(input[i][2]))
        {
            assert(output == (float *)CM_INVALID_INPUT_ERROR);
        }
        else
        {
            for (int j = 0; j < 3; j++)
            {
                assert(fabs(output[j] - expected_output[i][j]) < tolerance);
            }
            freeSoftmax(&output); 
        }
    }
    printf("softmax activation function test passed\n");
}

int main()
{
    printf("Testing softmax activation function\n");
    test_softmax();
    return 0;
}
