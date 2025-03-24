#include <math.h>
#include <float.h>
#include <stdio.h>
#include "../../include/Core/error_codes.h"
#include "../../include/Activations/relu.h"

#define DEBUG_LOGGING 0

/**
 * @brief Applies the Rectified Linear Unit (ReLU) activation function.
 *
 * The ReLU activation function is defined as:
 * - f(x) = x, if x > 0
 * - f(x) = 0, if x <= 0
 *
 * @param x The input value.
 * @return The result of the ReLU activation function.
 */
float relu(float x)
{
    if (isnan(x) || isinf(x) || x == -INFINITY)
    {
        fprintf(stderr, "[relu] Error: Invalid input (NaN or Inf)\n");
        return CM_INVALID_INPUT_ERROR;
    }

    float result = x > 0 ? x : 0;
#if DEBUG_LOGGING
    printf("[relu] Input: x=%f, Output: %f\n", x, result);
#endif
    return result;
}
