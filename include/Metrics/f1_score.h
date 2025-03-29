#ifndef F1_SCORE_H
#define F1_SCORE_H

/**
 * @brief Computes the F1 Score metric.
 *
 * @param y Pointer to the ground truth labels.
 * @param yHat Pointer to the predicted labels.
 * @param n The number of elements in y and yHat.
 * @param threshold The threshold for binary classification.
 * @return The computed F1 Score, or an error code if inputs are invalid.
 */
float f1_score(float *y, float *yHat, int n, float threshold);

#endif
