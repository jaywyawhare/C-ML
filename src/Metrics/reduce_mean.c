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
Node *reduce_mean(Node *x, int n)
{
    if (!x || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return NULL;
    }

    Node *sum = tensor(0.0f, 1);
    for (int i = 0; i < n; i++)
    {
        sum = tensor_add(sum, tensor(x->tensor->storage->data[i], 1));
    }

    return tensor_div(sum, tensor((float)n, 1));
}
