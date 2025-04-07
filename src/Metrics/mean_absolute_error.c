#include <math.h>
#include <stdio.h>
#include "../../include/Metrics/mean_absolute_error.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"

/**
 * @brief Implements the Mean Absolute Error metric.
 * 
 * The Mean Absolute Error (MAE) is defined as:
 * - MAE = 1/n * Σ |y - yHat|
 * 
 * @param y Pointer to the ground truth values.
 * @param yHat Pointer to the predicted values.
 * @param n The number of elements in y and yHat.
 * @return The computed MAE, or an error code if inputs are invalid.
 */
float mean_absolute_error(float *y, float *yHat, int n)
{
    if (!y || !yHat || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return (float)CM_INVALID_INPUT_ERROR;
    }
    float sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += fabsf(y[i] - yHat[i]);
    }
    return sum / n;
}
