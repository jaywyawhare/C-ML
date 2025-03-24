#ifndef BINARY_CROSS_ENTROPY_LOSS_H
#define BINARY_CROSS_ENTROPY_LOSS_H

/**
 * @brief Computes the Binary Cross-Entropy Loss.
 *
 * @param yHat Pointer to the predicted probabilities.
 * @param y Pointer to the ground truth labels.
 * @param size The number of elements in y and yHat.
 * @return The computed loss, or an error code if inputs are invalid.
 */
float binary_cross_entropy_loss(float *yHat, float *y, int size);

#endif
