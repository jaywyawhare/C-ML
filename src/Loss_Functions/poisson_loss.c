#include <math.h>
#include <stdio.h>
#include "../../include/Loss_Functions/poisson_loss.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"

#define EPSILON 1e-8

/**
 * @brief Computes the Poisson Loss.
 *
 * The Poisson Loss is defined as:
 * - loss = 1/n * Σ [yHat - y * log(yHat)]
 *
 * @param y Pointer to the ground truth values.
 * @param yHat Pointer to the predicted values.
 * @param n The number of elements in y and yHat.
 * @return The computed loss, or an error code if inputs are invalid.
 */
float poisson_loss(float *y, float *yHat, int n)
{
    if (!y || !yHat || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return (float)CM_INVALID_INPUT_ERROR;
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
    {
        sum += yHat[i] - y[i] * log(yHat[i] + EPSILON);
    }
    return sum / n;
}

/**
 * @brief Computes the derivative of the Poisson Loss.
 *
 * The derivative is defined as:
 * - d(loss)/dyHat = 1 - (y / (yHat + EPSILON))
 *
 * @param y Ground truth value.
 * @param yHat Predicted value.
 * @return The derivative value.
 */
float poisson_loss_derivative(float y, float yHat)
{
    return 1 - (y / (yHat + EPSILON));
}
