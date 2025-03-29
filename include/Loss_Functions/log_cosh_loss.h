#ifndef LOG_COSH_LOSS_H
#define LOG_COSH_LOSS_H

/**
 * @brief Computes the Log-Cosh Loss.
 *
 * @param y Pointer to the ground truth values.
 * @param yHat Pointer to the predicted values.
 * @param n The number of elements in y and yHat.
 * @return The computed loss, or an error code if inputs are invalid.
 */
float log_cosh_loss(float *y, float *yHat, int n);

/**
 * @brief Computes the derivative of the Log-Cosh Loss.
 *
 * @param y Ground truth value.
 * @param yHat Predicted value.
 * @return The derivative value.
 */
float log_cosh_loss_derivative(float y, float yHat);

#endif
