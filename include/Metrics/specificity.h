#ifndef SPECIFICITY_H
#define SPECIFICITY_H

/**
 * @brief Computes the Specificity metric.
 *
 * Specificity measures the fraction of true negatives.
 *
 * @param y Pointer to the ground truth labels.
 * @param yHat Pointer to the predicted labels.
 * @param n The number of elements in y and yHat.
 * @param threshold The threshold for binary classification.
 * @return The computed Specificity, or an error code if inputs are invalid.
 */
float specificity(float *y, float *yHat, int n, float threshold);

#endif
