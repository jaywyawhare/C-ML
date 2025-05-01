#include <math.h>
#include <stdio.h>
#include "../../include/Loss_Functions/mean_squared_error.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/memory_management.h"
#include "../../include/Core/logging.h"
#include "../../include/Core/autograd.h"

Node *mean_squared_error(Node *y, Node *yHat, int n)
{
    if (!y || !yHat || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return NULL;
    }

    Node *sum = tensor(0.0f, 1);
    Node *n_tensor = tensor((float)n, 0);

    for (int i = 0; i < n; i++)
    {
        Node *y_i = tensor(y->tensor->storage->data[i], 1);
        Node *yhat_i = tensor(yHat->tensor->storage->data[i], 1);
        Node *diff = tensor_sub(y_i, yhat_i);
        sum = tensor_add(sum, tensor_mul(diff, diff));

        cm_safe_free((void **)&y_i);
        cm_safe_free((void **)&yhat_i);
        cm_safe_free((void **)&diff);
    }

    return tensor_div(sum, n_tensor);
}
