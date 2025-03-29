#ifndef MCC_H
#define MCC_H

/**
 * @brief Computes the Matthews Correlation Coefficient (MCC).
 * 
 * @param y Pointer to the ground truth labels.
 * @param yHat Pointer to the predicted labels.
 * @param n The number of elements in y and yHat.
 * @param threshold The threshold for binary classification.
 * @return The computed MCC, or an error code if inputs are invalid.
 */
float mcc(float *y, float *yHat, int n, float threshold);

#endif
