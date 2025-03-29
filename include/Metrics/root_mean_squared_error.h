#ifndef ROOT_MEAN_SQUARED_ERROR_H
#define ROOT_MEAN_SQUARED_ERROR_H

/**
 * @brief Computes the Root Mean Squared Error (RMSE).
 *
 * @param y Pointer to the ground truth values.
 * @param yHat Pointer to the predicted values.
 * @param n The number of elements.
 * @return The computed RMSE, or an error code.
 */
float root_mean_squared_error(float *y, float *yHat, int n);

#endif
