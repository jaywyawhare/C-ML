#ifndef COHENS_KAPPA_H
#define COHENS_KAPPA_H

/**
 * @brief Computes Cohen's Kappa statistic.
 *
 * @param y Pointer to the ground truth labels.
 * @param yHat Pointer to the predicted labels.
 * @param n The number of elements in y and yHat.
 * @param threshold The threshold for binary classification.
 * @return The computed Cohen's Kappa, or an error code if inputs are invalid.
 */
float cohens_kappa(float *y, float *yHat, int n, float threshold);

#endif
