#ifndef RECALL_H
#define RECALL_H

/**
 * @brief Computes the Recall metric.
 *
 * @param y Pointer to the ground truth labels.
 * @param yHat Pointer to the predicted labels.
 * @param n The number of elements in y and yHat.
 * @param threshold The threshold for binary classification.
 * @return The computed Recall, or an error code if inputs are invalid.
 */
float recall(float *y, float *yHat, int n, float threshold);

#endif
