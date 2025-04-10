#include <math.h>
#include <float.h>
#include <stdio.h>
#include "../../include/Core/error_codes.h"
#include "../../include/Activations/leaky_relu.h"
#include "../../include/Core/logging.h"

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
float leaky_relu(float x)
{
   if (isnan(x) || isinf(x) || x == -INFINITY)
   {
      LOG_ERROR("Invalid input (NaN or Inf)");
      return CM_INVALID_INPUT_ERROR;
   }

   float result = x > 0 ? x : LEAKY_RELU_ALPHA * x;
   LOG_DEBUG("Input: x=%f, Output: %f", x, result);
   return result;
}

/**
 * @brief Computes the derivative of the Leaky ReLU activation function.
 *
 * The derivative of Leaky ReLU is:
 * - f'(x) = 1, if x > 0
 * - f'(x) = alpha, if x <= 0
 *
 * @param x The input value.
 * @return The derivative of the Leaky ReLU function.
 */
float leaky_relu_derivative(float x)
{
   if (isnan(x) || isinf(x))
   {
      LOG_ERROR("Invalid input (NaN or Inf)");
      return CM_INVALID_INPUT_ERROR;
   }

   return x > 0 ? 1.0f : LEAKY_RELU_ALPHA;
}