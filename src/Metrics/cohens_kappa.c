#include <stdio.h>
#include "../../include/Metrics/cohens_kappa.h"
#include "../../include/Core/error_codes.h"

/**
 * @brief Computes Cohen's Kappa statistic.
 * 
 * Cohen's Kappa is a measure of inter-rater agreement for categorical items.
 * It is defined as:
 * - kappa = (observed_accuracy - expected_accuracy) / (1 - expected_accuracy)
 * 
 * @param y Pointer to the ground truth labels.
 * @param yHat Pointer to the predicted labels.
 * @param n The number of elements in y and yHat.
 * @param threshold The threshold for binary classification.
 * @return The computed Cohen's Kappa, or an error code if inputs are invalid.
 */
float cohens_kappa(float *y, float *yHat, int n, float threshold)
{
    if (!y || !yHat || n <= 0)
    {
        fprintf(stderr, "[cohens_kappa] Error: Invalid input parameters.\n");
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
    float total = tp + tn + fp + fn;
    float observed_accuracy = (tp + tn) / total;
    float expected_accuracy = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (total * total);
    if (expected_accuracy == 1.0f)
        return 0.0f;
    return (observed_accuracy - expected_accuracy) / (1.0f - expected_accuracy);
}
