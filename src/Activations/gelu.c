#include <math.h>
#include <float.h>
#include "../../include/Activations/gelu.h"
#include "../../include/Core/error_codes.h"

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
        return CM_INVALID_INPUT_ERROR;
    }

    const float sqrt_2_over_pi = 0.7978845608f;
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
}