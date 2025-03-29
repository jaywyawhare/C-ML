#include <math.h>
#include <stdio.h>
#include "../../include/Metrics/root_mean_squared_error.h"
#include "../../include/Loss_Functions/mean_squared_error.h"
#include "../../include/Core/error_codes.h"

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
    float mse = mean_squared_error(y, yHat, n);
    return sqrtf(mse);
}
