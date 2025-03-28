#ifndef C_ML_GELU_H
#define C_ML_GELU_H

/**
 * @brief Applies the Gaussian Error Linear Unit (GELU) activation function.
 *
 * @param x The input value.
 * @return The result of the GELU activation function.
 */
float gelu(float x);

/**
 * @brief Computes the derivative of the GELU activation function.
 *
 * @param x The input value.
 * @return The derivative of the GELU function.
 */
float gelu_derivative(float x);
#endif
