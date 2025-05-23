#include <math.h>
#include <stdio.h>
#include "../../include/Metrics/mean_absolute_percentage_error.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"

/**
 * @brief Computes the Mean Absolute Percentage Error (MAPE).
 *
 * The MAPE is defined as:
 * - MAPE = 1/n * Σ |(y - yHat) / y| * 100
 *
 * @param y Pointer to the ground truth values.
 * @param yHat Pointer to the predicted values.
 * @param n The number of elements in y and yHat.
 * @return The computed MAPE, or an error code if inputs are invalid.
 */
float mean_absolute_percentage_error(float *y, float *yHat, int n)
{
    if (!y || !yHat || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return (float)CM_INVALID_INPUT_ERROR;
    }
    float sum = 0;
    int valid_count = 0;
    for (int i = 0; i < n; i++)
    {
        if (fabsf(y[i]) < 1e-15)
        {
            continue;
        }
        sum += fabsf((y[i] - yHat[i]) / y[i]);
        valid_count++;
    }
    if (valid_count == 0)
    {
        return CM_SUCCESS;
    }
    return sum / valid_count * 100;
}
