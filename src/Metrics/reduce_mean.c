#include <stdio.h>
#include "../../include/Metrics/reduce_mean.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"

/**
 * @brief Computes the mean of an array of floats.
 * 
 * This function takes an array of floats and computes the mean value.
 * 
 * @param loss Pointer to the array of floats.
 * @param size The number of elements in the array.
 * @return The computed mean, or an error code if inputs are invalid.
 */
float reduce_mean(float *loss, int size)
{
    if (!loss || size <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return (float)CM_INVALID_INPUT_ERROR;
    }
    float sum = 0;
    for (int i = 0; i < size; i++)
    {
        sum += loss[i];
    }
    return sum / size;
}
