#include <stdio.h>
#include "../../include/Metrics/mean_absolute_error.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"
#include "../../include/Core/autograd.h"

Node *mean_absolute_error(Node *y, Node *yHat, int n)
{
    if (!y || !yHat || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return NULL;
    }

    Node *sum = tensor(0.0f, 1);
    for (int i = 0; i < n; i++)
    {
        Node *y_i = tensor(y->tensor->storage->data[i], 1);
        Node *yhat_i = tensor(yHat->tensor->storage->data[i], 1);
        Node *diff = tensor_sub(y_i, yhat_i);
        Node *abs_diff = tensor_mul(diff, tensor(diff->value >= 0 ? 1.0f : -1.0f, 1));
        sum = tensor_add(sum, abs_diff);
    }

    return tensor_div(sum, tensor((float)n, 1));
}
