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
            continue;

        Node *diff = sub(y_val, tensor(yHat->tensor->storage->data[i], 1));
        Node *abs_pct = div(abs(diff), abs(y_val));
        sum = add(sum, abs_pct);
        valid_count = add(valid_count, tensor(1.0f, 1));
    }

    return mul(div(sum, valid_count), tensor(100.0f, 1));
}
