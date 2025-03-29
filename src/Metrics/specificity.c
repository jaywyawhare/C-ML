#include <stdio.h>
#include "../../include/Metrics/specificity.h"
#include "../../include/Core/error_codes.h"

/**
 * @brief Computes the specificity metric.
 * 
 * Specificity is defined as the ratio of true negatives to the sum of true negatives and false positives.
 * 
 * @param y Pointer to the ground truth labels.
 * @param yHat Pointer to the predicted labels.
 * @param n The number of elements in y and yHat.
 * @param threshold The threshold for binary classification.
 * @return The computed specificity, or an error code if inputs are invalid.
 */
float specificity(float *y, float *yHat, int n, float threshold)
{
    if (!y || !yHat || n <= 0)
    {
        fprintf(stderr, "[specificity] Error: Invalid input parameters.\n");
        return CM_INVALID_INPUT_ERROR;
    }
    int true_negative = 0, false_positive = 0;
    for (int i = 0; i < n; i++)
    {
        int actual = (int)y[i];
        int pred = yHat[i] > threshold ? 1 : 0;
        if (actual == 0 && pred == 0)
            true_negative++;
        else if (actual == 0 && pred == 1)
            false_positive++;
    }
    return (true_negative + false_positive) > 0 ? (float)true_negative / (true_negative + false_positive) : 0;
}
