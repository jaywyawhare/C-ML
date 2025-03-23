#include <math.h>
#include <float.h>
#include "../../include/Core/error_codes.h"
#include "../../include/Activations/leaky_relu.h"

#define LEAKY_RELU_ALPHA 0.01f

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
   if (isnan(x) || isinf(x) || x == -INFINITY)
   {
      return CM_INVALID_INPUT_ERROR;
   }

   return x > 0 ? x : LEAKY_RELU_ALPHA * x;
}