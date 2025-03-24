#include <stdio.h>
#include "../../include/Loss_Functions/reduce_mean.h"
#include "../../include/Core/error_codes.h"

#define DEBUG_LOGGING 0

/**
 * @brief Computes the mean of an array of loss values.
 *
 * The mean is defined as:
 * - mean = 1/size * Σ loss[i]
 *
 * @param loss Pointer to the array of loss values.
 * @param size The number of elements in the loss array.
 * @return The computed mean, or an error code if inputs are invalid.
 */
float reduce_mean(float *loss, int size)
{
    if (!loss || size <= 0)
    {
        fprintf(stderr, "[reduce_mean] Error: Invalid input parameters.\n");
        return (float)CM_INVALID_INPUT_ERROR;
    }

    float sum = 0;
    for (int i = 0; i < size; i++)
    {
        sum += loss[i];
    }
#if DEBUG_LOGGING
    printf("[reduce_mean] Computed mean: %f\n", sum / size);
#endif
    return sum / size;
}