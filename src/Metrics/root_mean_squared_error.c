#include <math.h>
#include <stdio.h>
#include "../../include/Metrics/root_mean_squared_error.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"
#include "../../include/Core/autograd.h"

Node *root_mean_squared_error(Node *y, Node *yHat, int n)
{
    if (!y || !yHat || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return NULL;
    }

    Node *sum = tensor(0.0f, 1);
    Node *n_tensor = tensor((float)n, 1);

    for (int i = 0; i < n; i++)
    {
        Node *diff = tensor_sub(y, yHat);
        sum = tensor_add(sum, tensor_mul(diff, diff));
    }

    return tensor_pow(tensor_div(sum, n_tensor), tensor(0.5f, 1));
}