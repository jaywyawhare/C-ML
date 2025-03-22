#include <math.h>
#include <float.h>
#include "../../include/Activations/tanh.h"

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
 * @return The result of the tanh activation function.
 */
float tanH(float x)
{
    if (x > 20.0f)
    {
        return 1.0f;
    }
    else if (x < -20.0f)
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