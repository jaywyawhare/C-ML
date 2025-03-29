#ifndef BALANCED_ACCURACY_H
#define BALANCED_ACCURACY_H

/**
 * @brief Computes the Balanced Accuracy metric.
 *
 * @param y Pointer to the ground truth labels.
 * @param yHat Pointer to the predicted labels.
 * @param n The number of elements in y and yHat.
 * @param threshold The threshold for binary classification.
 * @return The computed balanced accuracy, or an error code if inputs are invalid.
 */
float balanced_accuracy(float *y, float *yHat, int n, float threshold);

#endif
