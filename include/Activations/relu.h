#ifndef RELU_H
#define RELU_H

/**
 * @brief Applies the Rectified Linear Unit (ReLU) activation function.
 *
 * @param x The input value.
 * @return The result of the ReLU activation function.
 */
float relu(float x);

/**
 * @brief Computes the derivative of the ReLU activation function.
 *
 * @param x The input value.
 * @return The derivative of the ReLU function.
 */
float relu_derivative(float x);

#endif
