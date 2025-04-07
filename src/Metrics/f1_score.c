#include <stdio.h>
#include "../../include/Metrics/f1_score.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"

/**
 * @brief Computes the F1 Score metric.
 *
 * The F1 Score is the harmonic mean of precision and recall, defined as:
 * - F1 = 2 * (precision * recall) / (precision + recall)
 *
 * @note Precision is the ratio of true positives to the sum of true positives and false positives.
 * @note Recall is the ratio of true positives to the sum of true positives and false negatives.
 *
 * @param y Pointer to the ground truth labels.
 * @param yHat Pointer to the predicted labels.
 * @param n The number of elements in y and yHat.
 * @param threshold The threshold for binary classification.
 * @return The computed F1 Score, or an error code if inputs are invalid.
 */
float f1_score(float *y, float *yHat, int n, float threshold)
{
    if (!y || !yHat || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return CM_INVALID_INPUT_ERROR;
    }
    int true_positive = 0, false_positive = 0, false_negative = 0;
    for (int i = 0; i < n; i++)
    {
        int actual = (int)y[i];
        int pred = yHat[i] > threshold ? 1 : 0;
        if (actual == 1 && pred == 1)
            true_positive++;
        else if (actual == 0 && pred == 1)
            false_positive++;
        else if (actual == 1 && pred == 0)
            false_negative++;
    }
    float precision = true_positive + false_positive > 0 ? (float)true_positive / (true_positive + false_positive) : 0;
    float recall = true_positive + false_negative > 0 ? (float)true_positive / (true_positive + false_negative) : 0;
    if (precision + recall == 0)
        return 0;
    return 2 * precision * recall / (precision + recall);
}
