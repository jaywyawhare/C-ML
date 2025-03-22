#include "../../include/Activations/gelu.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
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
    const float sqrt_2_over_pi = 0.7978845608f;
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
}