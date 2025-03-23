#include <math.h>
#include <float.h>
#include "../../include/Core/error_codes.h"
#include "../../include/Activations/linear.h"

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
        return CM_INVALID_INPUT_ERROR;
    }
    return x;
}