#include "../../include/Activations/elu.h"

/**
 * @brief Applies the Exponential Linear Unit (ELU) activation function.
 *
 * The ELU activation function is defined as:
 * - f(x) = x, if x >= 0
 * - f(x) = alpha * (exp(x) - 1), if x < 0
 *
 * @param x The input value.
 * @param alpha The scaling factor for negative values.
 * @return The result of the ELU activation function.
 */
float elu(float x, float alpha)
{
    return x >= 0 ? x : alpha * (expf(x) - 1);
}
