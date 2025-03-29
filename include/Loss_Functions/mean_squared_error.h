#ifndef MEAN_SQUARED_ERROR_H
#define MEAN_SQUARED_ERROR_H

/**
 * @brief Computes the Mean Squared Error (MSE).
 *
 * @param y Pointer to the ground truth values.
 * @param yHat Pointer to the predicted values.
 * @param n The number of elements in y and yHat.
 * @return The computed MSE, or an error code.
 */
float mean_squared_error(float *y, float *yHat, int n);

/**
 * @brief Computes the derivative of Mean Squared Error (MSE) for a single sample.
 *
 * @param predicted Predicted value.
 * @param actual Actual value.
 * @param n Number of elements in the sample.
 * @return The derivative value.
 */
float mean_squared_error_derivative(float predicted, float actual, int n);

#endif
