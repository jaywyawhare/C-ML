#include <math.h>
#include <stdio.h>
#include "../../include/Loss_Functions/log_cosh_loss.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"

/**
 * @brief Computes the Log-Cosh Loss.
 *
 * The Log-Cosh Loss is defined as:
 * - loss = 1/n * Σ log(cosh(yHat - y))
 *
 * @param y Pointer to the ground truth values.
 * @param yHat Pointer to the predicted values.
 * @param n The number of elements in y and yHat.
 * @return The computed loss, or an error code if inputs are invalid.
 */
float log_cosh_loss(float *y, float *yHat, int n)
{
    if (!y || !yHat || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return (float)CM_INVALID_INPUT_ERROR;
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
    {
        float diff = yHat[i] - y[i];
        sum += log(cosh(diff));
    }
    return sum / n;
}

/**
 * @brief Computes the derivative of the Log-Cosh Loss.
 *
 * The derivative is defined as:
 * - d(loss)/dyHat = tanh(yHat - y)
 *
 * @param y Ground truth value.
 * @param yHat Predicted value.
 * @return The derivative value.
 */
float log_cosh_loss_derivative(float y, float yHat)
{
    return tanh(yHat - y);
}
