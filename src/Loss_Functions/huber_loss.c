#include <math.h>
#include <stdio.h>
#include "../../include/Loss_Functions/huber_loss.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"

#define DELTA 1.0f

/**
 * @brief Computes the Huber Loss.
 *
 * The Huber Loss is defined as:
 * - loss = 1/n * Σ [0.5 * (yHat - y)^2] if |yHat - y| <= delta
 * - loss = delta * (|yHat - y| - 0.5 * delta) otherwise
 *
 * @param y Pointer to the ground truth values.
 * @param yHat Pointer to the predicted values.
 * @param n The number of elements in y and yHat.
 * @return The computed loss, or an error code if inputs are invalid.
 */
float huber_loss(float *y, float *yHat, int n)
{
    if (!y || !yHat || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return (float)CM_INVALID_INPUT_ERROR;
    }
    float loss = 0;
    for (int i = 0; i < n; i++)
    {
        float diff = yHat[i] - y[i];
        if (fabsf(diff) <= DELTA)
            loss += 0.5f * diff * diff;
        else
            loss += DELTA * (fabsf(diff) - 0.5f * DELTA);
    }
    return loss / n;
}

/**
 * @brief Computes the derivative of the Huber Loss.
 *
 * The derivative is defined as:
 * - d(loss)/dyHat = (yHat - y) if |yHat - y| <= delta
 * - d(loss)/dyHat = delta * sign(yHat - y) otherwise
 *
 * @param y Ground truth value.
 * @param yHat Predicted value.
 * @return The derivative value.
 */
float huber_loss_derivative(float y, float yHat)
{
    float diff = yHat - y;
    if (fabsf(diff) <= DELTA)
        return diff;
    else
        return (diff > 0 ? DELTA : -DELTA);
}
