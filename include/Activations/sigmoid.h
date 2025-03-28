#ifndef SIGMOID_H
#define SIGMOID_H

/**
 * @brief Applies the sigmoid activation function.
 *
 * @param x The input value.
 * @return The result of the sigmoid activation function.
 */
float sigmoid(float x);

/**
 * @brief Computes the derivative of the sigmoid activation function.
 *
 * @param sigmoid_output The output of the sigmoid function (f(x)).
 * @return The derivative of the sigmoid function.
 */
float sigmoid_derivative(float sigmoid_output);

#endif
