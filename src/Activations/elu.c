#include <math.h>
#include <float.h>
#include "../../include/Activations/elu.h"
#include "../../include/Core/error_codes.h"

#define EPSILON 1e-6f

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
        return CM_INVALID_INPUT_ERROR;
    }

    return x >= 0 ? x : alpha * (expf(x) - 1);
}
