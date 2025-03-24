#ifndef FOCAL_LOSS_H
#define FOCAL_LOSS_H

/**
 * @brief Computes the Focal Loss.
 *
 * @param y Pointer to the ground truth labels.
 * @param yHat Pointer to the predicted probabilities.
 * @param n The number of elements in y and yHat.
 * @param gamma The focusing parameter to adjust the rate at which easy examples are down-weighted.
 * @return The computed loss, or an error code if inputs are invalid.
 */
float focal_loss(float *y, float *yHat, int n, float gamma);

#endif
