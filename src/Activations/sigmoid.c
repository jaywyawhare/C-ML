#include <math.h>
#include <float.h>
#include <stdio.h>
#include "../../include/Activations/sigmoid.h"
#include "../../include/Core/error_codes.h"

#define DEBUG_LOGGING 0

/**
 * @brief Applies the sigmoid activation function.
 *
 * The sigmoid activation function is defined as:
 * - f(x) = 1 / (1 + exp(-x))
 *
 * @param x The input value.
 * @return The result of the sigmoid activation function.
 */

float sigmoid(float x)
{
    if (isnan(x) || isinf(x) || x == -INFINITY)
    {
        fprintf(stderr, "[sigmoid] Error: Invalid input (NaN or Inf)\n");
        return CM_INVALID_INPUT_ERROR;
    }

    float result;
    if (x >= 0)
    {
        float exp_neg_x = expf(-x);
        result = 1 / (1 + exp_neg_x);
    }
    else
    {
        float exp_pos_x = expf(x);
        result = exp_pos_x / (1 + exp_pos_x);
    }
#if DEBUG_LOGGING
    printf("[sigmoid] Input: x=%f, Output: %f\n", x, result);
#endif
    return result;
}