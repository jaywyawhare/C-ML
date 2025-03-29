#include <stdio.h>
#include "../../include/Metrics/precision.h"
#include "../../include/Core/error_codes.h"

/**
 * @brief Computes the Precision metric.
 * 
 * The Precision is defined as the ratio of true positives to the sum of true positives and false positives.
 * 
 * @param y Pointer to the ground truth labels.
 * @param yHat Pointer to the predicted labels.
 * @param n The number of elements in y and yHat.
 * @param threshold The threshold for binary classification.
 * @return The computed Precision, or an error code if inputs are invalid.
 */
float precision(float *y, float *yHat, int n, float threshold)
{
    if (!y || !yHat || n <= 0)
    {
        fprintf(stderr, "[precision] Error: Invalid input parameters.\n");
        return CM_INVALID_INPUT_ERROR;
    }
    int true_positive = 0, false_positive = 0;
    for (int i = 0; i < n; i++)
    {
        int actual = (int)y[i];
        int pred = yHat[i] > threshold ? 1 : 0;
        if (actual == 1 && pred == 1)
            true_positive++;
        else if (actual == 0 && pred == 1)
            false_positive++;
    }
    return true_positive + false_positive > 0 ? (float)true_positive / (true_positive + false_positive) : 0;
}
