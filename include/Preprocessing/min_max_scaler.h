#ifndef MIN_MAX_SCALER_H
#define MIN_MAX_SCALER_H

/**
 * @brief Scales an array of floats to a range of [0, 1] using min-max scaling.
 *
 * @param x The input array of floats.
 * @param size The size of the input array.
 * @return A pointer to the scaled array, or NULL if an error occurs.
 */
float *minMaxScaler(float *x, int size);

/**
 * @brief Scales an array of floats to a range of [0, 1] using min-max scaling.
 *
 * @param x The input array of floats.
 * @param size The size of the input array.
 * @return A pointer to the scaled array, or NULL if an error occurs.
 */
float *min_max_scaler(float *x, int size);

#endif
