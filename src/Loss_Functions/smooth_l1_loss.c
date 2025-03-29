#include <math.h>
#include <stdio.h>
#include "../../include/Loss_Functions/smooth_l1_loss.h"
#include "../../include/Core/error_codes.h"

#define BETA 1.0f

/**
 * @brief Computes the Smooth L1 Loss.
 *
 * The Smooth L1 Loss is defined as:
 * - loss = 1/n * Σ [0.5 * (yHat - y)^2 / beta] if |yHat - y| < beta
 * - loss = |yHat - y| - 0.5 * beta otherwise
 *
 * @param y Pointer to the ground truth values.
 * @param yHat Pointer to the predicted values.
 * @param n The number of elements in y and yHat.
 * @return The computed loss, or an error code if inputs are invalid.
 */
float smooth_l1_loss(float *y, float *yHat, int n)
{
    if (!y || !yHat || n <= 0)
    {
        fprintf(stderr, "[smooth_l1_loss] Error: Invalid input parameters.\n");
        return (float)CM_INVALID_INPUT_ERROR;
    }
    float loss = 0.0f;
    for (int i = 0; i < n; i++)
    {
        float diff = fabsf(yHat[i] - y[i]);
        if (diff < BETA)
            loss += 0.5f * diff * diff / BETA;
        else
            loss += diff - 0.5f * BETA;
    }
    return loss / n;
}

/**
 * @brief Computes the derivative of the Smooth L1 Loss.
 *
 * The derivative is defined as:
 * - d(loss)/dyHat = (yHat - y) / beta if |yHat - y| < beta
 * - d(loss)/dyHat = sign(yHat - y) otherwise
 *
 * @param y Ground truth value.
 * @param yHat Predicted value.
 * @return The derivative value.
 */
float smooth_l1_loss_derivative(float y, float yHat)
{
    float diff = yHat - y;
    if (fabsf(diff) < BETA)
        return diff / BETA;
    else
        return (diff > 0 ? 1.0f : -1.0f);
}
