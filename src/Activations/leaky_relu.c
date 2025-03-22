#include <math.h>
#include "../../include/Activations/leaky_relu.h"

/**
 * @brief Applies the Leaky Rectified Linear Unit (Leaky ReLU) activation function.
 *
 * The Leaky ReLU activation function is defined as:
 * - f(x) = x, if x > 0
 * - f(x) = 0.01 * x, if x <= 0
 *
 * @param x The input value.
 * @return The result of the Leaky ReLU activation function.
 */
float leakyRelu(float x)
{
   return x > 0 ? x : 0.01 * x;
}