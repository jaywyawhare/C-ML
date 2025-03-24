#include <math.h>
#include <float.h>
#include <stdio.h>
#include "../../include/Activations/gelu.h"
#include "../../include/Core/error_codes.h"

#define DEBUG_LOGGING 0

/**
 * @brief Applies the Gaussian Error Linear Unit (GELU) activation function.
 *
 * The GELU activation function is defined as:
 * - f(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *
 * @param x The input value.
 * @return The result of the GELU activation function.
 */
float gelu(float x)
{
    if (isnan(x) || isinf(x) || x == -INFINITY)
    {
        fprintf(stderr, "[gelu] Error: Invalid input (NaN or Inf)\n");
        return CM_INVALID_INPUT_ERROR;
    }

    const float sqrt_2_over_pi = 0.7978845608f;
    float result = 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
#if DEBUG_LOGGING
    printf("[gelu] Input: x=%f, Output: %f\n", x, result);
#endif
    return result;
}