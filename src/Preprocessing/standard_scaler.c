#include <math.h>
#include "../../include/Preprocessing/standard_scaler.h"
#include "../../include/Core/logging.h"

Node *standard_scaler_tensor(Node *x)
{
    if (!x)
    {
        LOG_ERROR("Null input tensor");
        return NULL;
    }

    // Calculate mean
    Node *sum = tensor(0.0f, 1);
    for (int i = 0; i < x->tensor->storage->size; i++)
    {
        sum = add(sum, tensor(x->tensor->storage->data[i], 0));
    }
    Node *mean = div(sum, tensor((float)x->tensor->storage->size, 0));

    // Calculate variance
    Node *var_sum = tensor(0.0f, 1);
    for (int i = 0; i < x->tensor->storage->size; i++)
    {
        Node *diff = sub(tensor(x->tensor->storage->data[i], 0), mean);
        var_sum = add(var_sum, mul(diff, diff));
    }
    Node *std = tensor(sqrt(var_sum->value / x->tensor->storage->size), 1);

    // (x - mean) / std
    Node *centered = sub(x, mean);
    Node *scaled = div(centered, std);
    scaled->requires_grad = x->requires_grad;

    return scaled;
}