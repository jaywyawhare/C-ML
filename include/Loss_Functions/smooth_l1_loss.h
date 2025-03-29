#ifndef SMOOTH_L1_LOSS_H
#define SMOOTH_L1_LOSS_H

/**
 * @brief Computes the Smooth L1 Loss.
 *
 * @param y Pointer to the ground truth values.
 * @param yHat Pointer to the predicted values.
 * @param n The number of elements in y and yHat.
 * @return The computed loss, or an error code if inputs are invalid.
 */
float smooth_l1_loss(float *y, float *yHat, int n);

/**
 * @brief Computes the derivative of the Smooth L1 Loss.
 *
 * @param y Ground truth value.
 * @param yHat Predicted value.
 * @return The derivative value.
 */
float smooth_l1_loss_derivative(float y, float yHat);

#endif
