#include <math.h>
#include <float.h>
#include <stdio.h>
#include "../../include/Core/error_codes.h"
#include "../../include/Activations/leaky_relu.h"

#define LEAKY_RELU_ALPHA 0.01f
#define DEBUG_LOGGING 0

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
      fprintf(stderr, "[leaky_relu] Error: Invalid input (NaN or Inf)\n");
      return CM_INVALID_INPUT_ERROR;
   }

   float result = x > 0 ? x : LEAKY_RELU_ALPHA * x;
#if DEBUG_LOGGING
   printf("[leaky_relu] Input: x=%f, Output: %f\n", x, result);
#endif
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
      fprintf(stderr, "[leaky_relu_derivative] Error: Invalid input (NaN or Inf)\n");
      return CM_INVALID_INPUT_ERROR;
   }

   return x > 0 ? 1.0f : LEAKY_RELU_ALPHA;
}