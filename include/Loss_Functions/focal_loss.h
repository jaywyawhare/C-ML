#ifndef FOCAL_LOSS_H
#define FOCAL_LOSS_H

/**
 * @brief Computes the Focal Loss.
 *
 * @param y Pointer to the ground truth values.
 * @param yHat Pointer to the predicted values.
 * @param n The number of elements in y and yHat.
 * @param gamma The focusing parameter to adjust the rate at which easy examples are down-weighted.
 * @return The computed loss, or an error code if inputs are invalid.
 */
float focal_loss(float *y, float *yHat, int n, float gamma);

/**
 * @brief Computes the derivative of the Focal Loss.
 *
 * @param y Ground truth value.
 * @param yHat Predicted value.
 * @param gamma The focusing parameter.
 * @return The derivative value.
 */
float focal_loss_derivative(float y, float yHat, float gamma);

#endif
