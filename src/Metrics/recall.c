#include <stdio.h>
#include "../../include/Metrics/recall.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"

/**
 * @brief Computes the Recall metric.
 *
 * The Recall is defined as the ratio of true positives to the sum of true positives and false negatives.
 *
 * @param y Pointer to the ground truth labels.
 * @param yHat Pointer to the predicted labels.
 * @param n The number of elements in y and yHat.
 * @param threshold The threshold for binary classification.
 * @return The computed Recall, or an error code if inputs are invalid.
 */
float recall(float *y, float *yHat, int n, float threshold)
{
    if (!y || !yHat || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return CM_INVALID_INPUT_ERROR;
    }
    int true_positive = 0, false_negative = 0;
    for (int i = 0; i < n; i++)
    {
        int actual = (int)y[i];
        int pred = yHat[i] > threshold ? 1 : 0;
        if (actual == 1 && pred == 1)
            true_positive++;
        else if (actual == 1 && pred == 0)
            false_negative++;
    }
    return (true_positive + false_negative) > 0 ? (float)true_positive / (true_positive + false_negative) : 0;
}
