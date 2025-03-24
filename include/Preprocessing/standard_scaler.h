#ifndef STANDARD_SCALER_H
#define STANDARD_SCALER_H

/**
 * @brief Scales an array of floats to have a mean of 0 and a standard deviation of 1.
 *
 * @param x The input array of floats.
 * @param size The size of the input array.
 * @return A pointer to the scaled array, or NULL if an error occurs.
 */
float *standardScaler(float *x, int size);

/**
 * @brief Scales an array of floats to have a mean of 0 and a standard deviation of 1.
 *
 * @param x The input array of floats.
 * @param size The size of the input array.
 * @return A pointer to the scaled array, or NULL if an error occurs.
 */
float *standard_scaler(float *x, int size);

#endif
