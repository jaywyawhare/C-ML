#ifndef MEAN_ABSOLUTE_PERCENTAGE_ERROR_H
#define MEAN_ABSOLUTE_PERCENTAGE_ERROR_H

/**
 * @brief Computes the Mean Absolute Percentage Error (MAPE).
 *
 * @param y Pointer to the ground truth values.
 * @param yHat Pointer to the predicted values.
 * @param n The number of elements in y and yHat.
 * @return The computed error as a percentage, or an error code if inputs are invalid.
 */
float mean_absolute_percentage_error(float *y, float *yHat, int n);

#endif
