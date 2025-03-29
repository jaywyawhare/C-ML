#include "../../include/Metrics/r2_score.h"
#include <math.h>
#include <stddef.h>

/**
 * @brief Calculate the R2 score (coefficient of determination).
 *
 * @param y_true Array of true values.
 * @param y_pred Array of predicted values.
 * @param size Number of elements in the arrays.
 * @return float The R2 score.
 */
float r2_score(const float *y_true, const float *y_pred, int size)
{
    if (y_true == NULL || y_pred == NULL || size <= 0)
    {
        return NAN; 
    }

    float mean_y_true = 0.0f;
    for (int i = 0; i < size; i++)
    {
        mean_y_true += y_true[i];
    }
    mean_y_true /= size;

    float ss_total = 0.0f;
    float ss_residual = 0.0f;
    for (int i = 0; i < size; i++)
    {
        ss_total += pow(y_true[i] - mean_y_true, 2);
        ss_residual += pow(y_true[i] - y_pred[i], 2);
    }

    return 1.0f - (ss_residual / ss_total);
}
