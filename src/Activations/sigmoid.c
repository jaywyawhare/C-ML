#include <math.h>
#include <float.h>
#include <stdio.h>
#include "../../include/Activations/sigmoid.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"

#ifndef DEBUG_LOGGING
#define DEBUG_LOGGING 0
#endif

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
        LOG_ERROR("Invalid input (NaN or Inf)");
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
    printf("[sigmoid] Debug: Input: x=%f, Output: %f\n", x, result);
#endif
    return result;
}

/**
 * @brief Computes the derivative of the sigmoid activation function.
 *
 * The derivative of sigmoid is:
 * - f'(x) = f(x) * (1 - f(x))
 *
 * @param sigmoid_output The output of the sigmoid function (f(x)).
 * @return The derivative of the sigmoid function.
 */
float sigmoid_derivative(float sigmoid_output)
{
    if (isnan(sigmoid_output) || isinf(sigmoid_output) || sigmoid_output < 0.0f || sigmoid_output > 1.0f)
    {
        LOG_ERROR("Invalid sigmoid output (NaN, Inf, or out of range)");
        return CM_INVALID_INPUT_ERROR;
    }

    return sigmoid_output * (1.0f - sigmoid_output);
}