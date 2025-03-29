#ifndef R2_SCORE_H
#define R2_SCORE_H

/**
 * @brief Calculate the R2 score (coefficient of determination).
 *
 * @param y_true Array of true values.
 * @param y_pred Array of predicted values.
 * @param size Number of elements in the arrays.
 * @return float The R2 score.
 */
float r2_score(const float *y_true, const float *y_pred, int size);

#endif // R2_SCORE_H
