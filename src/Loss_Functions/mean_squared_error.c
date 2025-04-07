#include <math.h>
#include <stdio.h>
#include "../../include/Loss_Functions/mean_squared_error.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"

/**
 * @brief Computes the Mean Squared Error (MSE).
 *
 * The MSE is defined as:
 * - error = 1/n * Σ (y - yHat)^2
 *
 * @param y Pointer to the ground truth values.
 * @param yHat Pointer to the predicted values.
 * @param n The number of elements in y and yHat.
 * @return The computed error, or an error code if inputs are invalid.
 */
float mean_squared_error(float *y, float *yHat, int n)
{
    if (!y || !yHat || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.\n");
        return (float)CM_INVALID_INPUT_ERROR;
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
    {
        float diff = yHat[i] - y[i];
        sum += diff * diff;
    }
    return sum / n;
}

/**
 * @brief Computes the derivative of the Mean Squared Error (MSE).
 *
 * The derivative is defined as:
 * - d(MSE)/dyHat = 2 * (yHat - y) / n
 *
 * @param predicted Predicted value.
 * @param actual Ground truth value.
 * @param n The total number of elements.
 * @return The derivative value.
 */
float mean_squared_error_derivative(float predicted, float actual, int n)
{
    if (n <= 0)
    {
        LOG_ERROR("Invalid size parameter.\n");
        return 0.0f;
    }
    if (predicted < 0 || predicted > 1)
    {
        LOG_ERROR("Predicted value out of bounds.\n");
        return 0.0f;
    }
    if (actual < 0 || actual > 1)
    {
        LOG_ERROR("Actual value out of bounds.\n");
        return 0.0f;
    }
    return (2.0f * (predicted - actual)) / n;
}
