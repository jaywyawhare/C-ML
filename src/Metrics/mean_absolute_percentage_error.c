#include <math.h>
#include "../../include/Metrics/mean_absolute_percentage_error.h"
#include "../../include/Core/autograd.h"
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
Node *mean_absolute_percentage_error(Node *y, Node *yHat, int n)
{
    if (!y || !yHat || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return NULL;
    }

    Node *sum = tensor(0.0f, 1);
    Node *valid_count = tensor(0.0f, 1);

    for (int i = 0; i < n; i++)
    {
        Node *y_val = tensor(y->tensor->storage->data[i], 1);
        if (fabsf(y_val->value) < 1e-15)
        {
            continue;
        }

        Node *diff = tensor_sub(yHat, y);
        Node *abs_diff = tensor_abs(diff);
        Node *abs_y = tensor_abs(y_val);
        Node *abs_pct = tensor_div(abs_diff, abs_y);
        sum = tensor_add(sum, abs_pct);
        valid_count = tensor_add(valid_count, tensor(1.0f, 1));
    }

    return tensor_mul(tensor_div(sum, valid_count), tensor(100.0f, 1));
}
