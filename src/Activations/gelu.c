#include <math.h>
#include <float.h>
#include <stdio.h>
#include "../../include/Activations/gelu.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"

#ifndef DEBUG_LOGGING
#define DEBUG_LOGGING 0
#endif

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
        LOG_ERROR("Invalid input (NaN or Inf)");
        return CM_INVALID_INPUT_ERROR;
    }

    const float sqrt_2_over_pi = 0.7978845608f;
    float result = 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
    LOG_DEBUG("Input: x=%f, Output: %f", x, result);
    return result;
}

/**
 * @brief Computes the derivative of the GELU activation function.
 *
 * The derivative of GELU is approximated as:
 * - f'(x) = 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *           + 0.5 * x * (1 - tanh^2(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *             * sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
 *
 * @param x The input value.
 * @return The derivative of the GELU function.
 */
float gelu_derivative(float x)
{
    if (isnan(x) || isinf(x))
    {
        LOG_ERROR("Invalid input (NaN or Inf)");
        return CM_INVALID_INPUT_ERROR;
    }

    const float sqrt_2_over_pi = 0.7978845608f;
    float tanh_arg = sqrt_2_over_pi * (x + 0.044715f * x * x * x);
    float tanh_val = tanhf(tanh_arg);
    float sech2 = 1.0f - tanh_val * tanh_val;

    return 0.5f * (1.0f + tanh_val) + 0.5f * x * sech2 * sqrt_2_over_pi * (1.0f + 3.0f * 0.044715f * x * x);
}