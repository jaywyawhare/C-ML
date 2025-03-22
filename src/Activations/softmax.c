#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../../include/Activations/softmax.h"
#include "../../include/Core/memory_management.h"

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
    float max_val = z[0];
    for (int i = 1; i < n; i++)
    {
        if (z[i] > max_val)
        {
            max_val = z[i];
        }
    }

    float sum = 0.0f; // Initialize sum to 0 for clarity
    float *output = cm_safe_malloc(n * sizeof(float), __FILE__, __LINE__);

    for (int i = 0; i < n; i++)
    {
        output[i] = expf(z[i] - max_val); // Exponentiate with max_val shift for numerical stability
        sum += output[i];
    }

    // Normalize the output array to get softmax values
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
        *output = NULL; // Set the caller's pointer to NULL
    }
}
