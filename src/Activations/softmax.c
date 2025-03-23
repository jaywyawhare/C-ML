#include <math.h>
#include <float.h>
#include "../../include/Activations/softmax.h"
#include "../../include/Core/memory_management.h"
#include "../../include/Core/error_codes.h"

/**
 * @brief Applies the softmax activation function.
 *
 * The softmax function is defined as:
 * - softmax(z_i) = exp(z_i) / sum(exp(z_j)) for j = 1 to n
 *
 * @param z Pointer to the input array.
 * @param n The number of elements in the input array.
 * @return Pointer to the output array containing the softmax values.
 */
float *softmax(float *z, int n)
{
    if (z == NULL || n <= 0)
    {
        return (float *)CM_NULL_ERROR;
    }

    for (int i = 0; i < n; i++)
    {
        if (isnan(z[i]) || isinf(z[i]) || z[i] == -INFINITY || z[i] == INFINITY)
        {
            return (float *)CM_INVALID_INPUT_ERROR;
        }
    }

    float max_val = z[0];
    for (int i = 1; i < n; i++)
    {
        if (z[i] > max_val)
        {
            max_val = z[i];
        }
    }

    float sum = 0.0f;
    float *output = cm_safe_malloc(n * sizeof(float), __FILE__, __LINE__);
    if (output == NULL)
    {
        return (float *)CM_MEMORY_ALLOCATION_ERROR;
    }

    for (int i = 0; i < n; i++)
    {
        output[i] = expf(z[i] - max_val);
        sum += output[i];
    }

    if (sum == 0.0f)
    {
        cm_safe_free(output);
        return (float *)CM_DIVISION_BY_ZERO_ERROR;
    }

    for (int i = 0; i < n; i++)
    {
        output[i] /= sum;
    }

    return output;
}

/**
 * @brief Frees the memory allocated for the softmax output.
 *
 * @param output Pointer to the output array to be freed.
 */
void freeSoftmax(float **output)
{
    if (output != NULL && *output != NULL)
    {
        cm_safe_free(*output);
    }
}
