#ifndef MEAN_SQUARED_ERROR_H
#define MEAN_SQUARED_ERROR_H

/**
 * @brief Computes the Mean Squared Error (MSE).
 *
 * @param y Pointer to the ground truth values.
 * @param yHat Pointer to the predicted values.
 * @param n The number of elements in y and yHat.
 * @return The computed error, or an error code if inputs are invalid.
 */
float mean_squared_error(float *y, float *yHat, int n);

#endif
