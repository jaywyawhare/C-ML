#include <math.h>
#include <float.h>
#include <stdio.h>
#include "../../include/Core/error_codes.h"
#include "../../include/Activations/relu.h"
#include "../../include/Core/logging.h"

#ifndef DEBUG_LOGGING
#define DEBUG_LOGGING 0
#endif

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
        LOG_ERROR("Invalid input (NaN or Inf)");
        return CM_INVALID_INPUT_ERROR;
    }

    float result = x > 0 ? x : 0;
#if DEBUG_LOGGING
    LOG_DEBUG("Input: x=%f, Output: %f", x, result);
#endif
    return result;
}

/**
 * @brief Computes the derivative of the ReLU activation function.
 *
 * The derivative of ReLU is:
 * - f'(x) = 1, if x > 0
 * - f'(x) = 0, if x <= 0
 *
 * @param x The input value.
 * @return The derivative of the ReLU function.
 */
float relu_derivative(float x)
{
    if (isnan(x) || isinf(x))
    {
        LOG_ERROR("Invalid input (NaN or Inf)");
        return CM_INVALID_INPUT_ERROR;
    }

    return x > 0 ? 1.0f : 0.0f;
}
