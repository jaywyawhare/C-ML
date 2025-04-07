#include <math.h>
#include <float.h>
#include <stdio.h>
#include "../../include/Core/error_codes.h"
#include "../../include/Activations/linear.h"
#include "../../include/Core/logging.h"

#ifndef DEBUG_LOGGING
#define DEBUG_LOGGING 0
#endif

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
        LOG_ERROR("Invalid input (NaN or Inf)");
        return CM_INVALID_INPUT_ERROR;
    }

    float result = x;
#if DEBUG_LOGGING
    printf("[linear] Debug: Input: x=%f, Output: %f\n", x, result);
#endif
    return result;
}

/**
 * @brief Computes the derivative of the Linear activation function.
 *
 * The derivative of Linear is:
 * - f'(x) = 1
 *
 * @param x The input value.
 * @return The derivative of the Linear function.
 */
float linear_derivative(float x)
{
    if (isnan(x) || isinf(x))
    {
        LOG_ERROR("Invalid input (NaN or Inf)");
        return CM_INVALID_INPUT_ERROR;
    }

    return 1.0f;
}