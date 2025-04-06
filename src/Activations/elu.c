#include <math.h>
#include <float.h>
#include <stdio.h>
#include "../../include/Activations/elu.h"
#include "../../include/Core/error_codes.h"

#define EPSILON 1e-6f
#ifndef DEBUG_LOGGING
#define DEBUG_LOGGING 0
#endif

/**
 * @brief Applies the Exponential Linear Unit (ELU) activation function.
 *
 * The ELU activation function is defined as:
 * - f(x) = x, if x >= 0
 * - f(x) = alpha * (exp(x) - 1), if x < 0
 *
 * @param x The input value.
 * @param alpha The scaling factor for negative values.
 * @return The result of the ELU activation function, or an error code.
 */
float elu(float x, float alpha)
{
    if (isnan(x) || isnan(alpha) || isinf(x) || isinf(alpha) || x == -INFINITY || alpha == -INFINITY)
    {
        fprintf(stderr, "[elu] Error: Invalid input (NaN or Inf)\n");
        return CM_INVALID_INPUT_ERROR;
    }

    float result = x >= 0 ? x : alpha * (expf(x) - 1);
#if DEBUG_LOGGING
    printf("[elu] Input: x=%f, alpha=%f, Output: %f\n", x, alpha, result);
#endif
    return result;
}

/**
 * @brief Computes the derivative of the ELU activation function.
 *
 * The derivative of ELU is:
 * - f'(x) = 1, if x >= 0
 * - f'(x) = f(x) + alpha, if x < 0
 *
 * @param x The input value.
 * @param alpha The scaling factor for negative values.
 * @return The derivative of the ELU function.
 */
float elu_derivative(float x, float alpha)
{
    if (isnan(x) || isnan(alpha) || isinf(x) || isinf(alpha))
    {
        fprintf(stderr, "[elu_derivative] Error: Invalid input (NaN or Inf)\n");
        return CM_INVALID_INPUT_ERROR;
    }

    if (x >= 0)
    {
        return 1.0f;
    }
    else
    {
        return alpha * expf(x);
    }
}