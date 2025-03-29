#ifndef ACCURACY_H
#define ACCURACY_H

/**
 * @brief Computes the accuracy of predictions.
 *
 * @param y Pointer to the ground truth labels.
 * @param yHat Pointer to the predicted labels.
 * @param n The number of elements in y and yHat.
 * @param threshold The threshold for binary classification.
 * @return The computed accuracy, or an error code if inputs are invalid.
 */
float accuracy(float *y, float *yHat, int n, float threshold);

#endif
