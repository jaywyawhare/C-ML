#include <math.h>
#include <float.h>
#include "../../include/Activations/tanh.h"
#include "../../include/Core/error_codes.h"

#define TANH_THRESHOLD 20.0f

/**
 * @brief Applies the hyperbolic tangent (tanh) activation function.
 *
 * The tanh activation function is defined as:
 * - f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
 *
 * For numerical stability:
 * - Returns 1.0 if x > 20.0
 * - Returns -1.0 if x < -20.0
 *
 * @param x The input value.
 * @return The result of the tanh activation function. Clipped to -1.0 or 1.0
 *         for extreme inputs, ensuring numerical stability.
 */
float tanH(float x)
{
    if (isnan(x) || isinf(x) || x == -INFINITY)
    {
        return CM_INVALID_INPUT_ERROR;
    }
    if (x > TANH_THRESHOLD)
    {
        return 1.0f;
    }
    else if (x < -TANH_THRESHOLD)
    {
        return -1.0f;
    }
    else
    {
        float e_pos = expf(x);
        float e_neg = expf(-x);
        return (e_pos - e_neg) / (e_pos + e_neg);
    }
}
