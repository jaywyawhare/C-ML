#include "../../include/Core/autograd.h"
// ...existing includes...

Node *mean_squared_error(Node *y, Node *yHat, int n)
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
        Node *y_i = tensor(y->tensor->storage->data[i], 1);
        Node *yhat_i = tensor(yHat->tensor->storage->data[i], 1);
        Node *diff = sub(y_i, yhat_i);
        sum = add(sum, mul(diff, diff));
    }

    return div(sum, n_tensor);
}
