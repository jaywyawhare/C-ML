#ifndef COSINE_SIMILARITY_LOSS_H
#define COSINE_SIMILARITY_LOSS_H

/**
 * @brief Computes the Cosine Similarity Loss.
 *
 * @param y Pointer to the ground truth values.
 * @param yHat Pointer to the predicted values.
 * @param n The number of elements in y and yHat.
 * @return The computed loss, or an error code if inputs are invalid.
 */
float cosine_similarity_loss(float *y, float *yHat, int n);

/**
 * @brief Computes the derivative of the Cosine Similarity Loss.
 *
 * @param y Pointer to the ground truth values.
 * @param yHat Pointer to the predicted values.
 * @param n The number of elements in y and yHat.
 * @return The derivative value, or an error code if inputs are invalid.
 */
float cosine_similarity_loss_derivative(float *y, float *yHat, int n);

#endif
