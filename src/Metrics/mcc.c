#include <math.h>
#include <stdio.h>
#include "../../include/Metrics/mcc.h"
#include "../../include/Core/error_codes.h"

/**
 * @brief Computes the Matthews Correlation Coefficient (MCC).
 *
 * The MCC is a measure of the quality of binary classifications.
 * It is defined as:
 * - mcc = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
 *
 * @param y Pointer to the ground truth labels.
 * @param yHat Pointer to the predicted labels.
 * @param n The number of elements in y and yHat.
 * @param threshold The threshold for binary classification.
 * @return The computed MCC, or an error code if inputs are invalid.
 */
float mcc(float *y, float *yHat, int n, float threshold)
{
    if (!y || !yHat || n <= 0)
    {
        fprintf(stderr, "[mcc] Error: Invalid input parameters.\n");
        return CM_INVALID_INPUT_ERROR;
    }
    int tp = 0, tn = 0, fp = 0, fn = 0;
    for (int i = 0; i < n; i++)
    {
        int actual = (int)y[i];
        int pred = yHat[i] > threshold ? 1 : 0;
        if (actual == 1 && pred == 1)
            tp++;
        else if (actual == 0 && pred == 0)
            tn++;
        else if (actual == 0 && pred == 1)
            fp++;
        else if (actual == 1 && pred == 0)
            fn++;
    }
    float numerator = (float)(tp * tn - fp * fn);
    float denominator = sqrtf((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
    if (denominator == 0)
        return 0.0f;
    return numerator / denominator;
}
