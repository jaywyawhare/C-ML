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
void freeSoftmax(float **output);

#endif
