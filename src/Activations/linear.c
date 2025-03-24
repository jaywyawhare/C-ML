#include <math.h>
#include <float.h>
#include <stdio.h>
#include "../../include/Core/error_codes.h"
#include "../../include/Activations/linear.h"

#define DEBUG_LOGGING 0

/**
 * @brief Applies the Linear (identity) activation function.
 *
 * The Linear activation function is defined as:
 * - f(x) = x
 *
 * @param x The input value.
 * @return The result of the Linear activation function.
 */
float linear(float x)
{
    if (isnan(x) || isinf(x) || x == -INFINITY)
    {
        fprintf(stderr, "[linear] Error: Invalid input (NaN or Inf)\n");
        return CM_INVALID_INPUT_ERROR;
    }

    float result = x;
#if DEBUG_LOGGING
    printf("[linear] Input: x=%f, Output: %f\n", x, result);
#endif
    return result;
}