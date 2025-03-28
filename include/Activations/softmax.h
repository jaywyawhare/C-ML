#ifndef SOFTMAX_H
#define SOFTMAX_H

/**
 * @brief Applies the softmax activation function.
 *
 * @param z Pointer to the input array.
 * @param n The number of elements in the input array.
 * @return Pointer to the output array containing the softmax values.
 */
float *softmax(float *z, int n);

/**
 * @brief Frees the memory allocated for the softmax output.
 *
 * @param output Pointer to the pointer of the output array to be freed.
 */
void free_softmax(float **output);

/**
 * @brief Computes the derivative of the softmax activation function.
 *
 * @param softmax_output Pointer to the softmax output array.
 * @param n The number of elements in the output array.
 * @return Pointer to the Jacobian matrix (n x n) or error code.
 */
float *softmax_derivative(float *softmax_output, int n);

/**
 * @brief Frees the memory allocated for the softmax derivative Jacobian.
 *
 * @param jacobian Pointer to the Jacobian matrix to be freed.
 */
void free_softmax_derivative(float **jacobian);

#endif
