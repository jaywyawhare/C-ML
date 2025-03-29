#include <stdio.h>
#include "../../include/Metrics/accuracy.h"
#include "../../include/Core/error_codes.h"

/**
 * @brief Computes the accuracy of predictions.
 *
 * The accuracy is defined as the ratio of correct predictions to the total number of predictions.
 *
 * @param y Pointer to the ground truth labels.
 * @param yHat Pointer to the predicted labels.
 * @param n The number of elements in y and yHat.
 * @param threshold The threshold for binary classification.
 * @return The computed accuracy, or an error code if inputs are invalid.
 */
float accuracy(float *y, float *yHat, int n, float threshold)
{
    if (!y || !yHat || n <= 0)
    {
        fprintf(stderr, "[accuracy] Error: Invalid input parameters.\n");
        return CM_INVALID_INPUT_ERROR;
    }
    int correct = 0;
    for (int i = 0; i < n; i++)
    {
        int pred = yHat[i] > threshold ? 1 : 0;
        if (pred == (int)y[i])
            correct++;
    }
    return (float)correct / n;
}
