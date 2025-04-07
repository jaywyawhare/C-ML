#include <math.h>
#include <stdio.h>
#include "../../include/Metrics/root_mean_squared_error.h"
#include "../../include/Loss_Functions/mean_squared_error.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"

/**
 * @brief Computes the Root Mean Squared Error (RMSE).
 *
 * The RMSE is defined as:
 * - RMSE = sqrt(1/n * Σ (y - yHat)^2)
 *
 * @param y Pointer to the ground truth values.
 * @param yHat Pointer to the predicted values.
 * @param n The number of elements in y and yHat.
 * @return The computed RMSE, or an error code if inputs are invalid.
 */
float root_mean_squared_error(float *y, float *yHat, int n)
{
    if (!y || !yHat || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return (float)CM_INVALID_INPUT_ERROR;
    }

    float sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += powf(y[i] - yHat[i], 2);
    }
    if (n == 0)
    {
        return 0;
    }
    return sqrtf(sum / n);
}