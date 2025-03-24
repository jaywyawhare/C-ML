#include <math.h>
#include <stdio.h>
#include "../../include/Loss_Functions/focal_loss.h"
#include "../../include/Core/error_codes.h"

#define DEBUG_LOGGING 0

/**
 * @brief Computes the Focal Loss.
 *
 * The Focal Loss is designed to address class imbalance by down-weighting
 * well-classified examples. It is defined as:
 * - loss = -1/n * Σ [y * (1 - yHat)^gamma * log(yHat) + (1 - y) * yHat^gamma * log(1 - yHat)]
 *
 * @param y Pointer to the ground truth labels.
 * @param yHat Pointer to the predicted probabilities.
 * @param n The number of elements in y and yHat.
 * @param gamma The focusing parameter to adjust the rate at which easy examples are down-weighted.
 * @return The computed loss, or an error code if inputs are invalid.
 */
float focal_loss(float *y, float *yHat, int n, float gamma)
{
    if (!y || !yHat || n <= 0)
    {
        fprintf(stderr, "[focal_loss] Error: Invalid input parameters.\n");
        return (float)CM_INVALID_INPUT_ERROR;
    }

    float sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += -y[i] * powf(1 - yHat[i], gamma) * logf(fmaxf(yHat[i], 1e-15)) - (1 - y[i]) * powf(yHat[i], gamma) * logf(fmaxf(1 - yHat[i], 1e-15));
    }
#if DEBUG_LOGGING
    printf("[focal_loss] Computed loss: %f\n", sum / n);
#endif
    return sum / n;
}