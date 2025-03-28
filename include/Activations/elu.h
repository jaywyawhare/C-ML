#ifndef ELU_H
#define ELU_H

/**
 * @brief Applies the Exponential Linear Unit (ELU) activation function.
 *
 * @param x The input value.
 * @param alpha The scaling factor for negative values.
 * @return The result of the ELU activation function.
 */
float elu(float x, float alpha);

/**
 * @brief Computes the derivative of the ELU activation function.
 *
 * @param x The input value.
 * @param alpha The scaling factor for negative values.
 * @return The derivative of the ELU function.
 */
float elu_derivative(float x, float alpha);

#endif
