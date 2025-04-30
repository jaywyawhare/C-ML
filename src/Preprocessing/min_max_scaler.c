#include <math.h>
#include "../../include/Preprocessing/min_max_scaler.h"
#include "../../include/Core/logging.h"

Node *min_max_scaler_tensor(Node *x)
{
    if (!x)
    {
        LOG_ERROR("Null input tensor");
        return NULL;
    }

    Node *min_val = tensor(x->tensor->storage->data[0], 0);
    Node *max_val = tensor(x->tensor->storage->data[0], 0);

    for (int i = 1; i < x->tensor->storage->size; i++)
    {
        float val = x->tensor->storage->data[i];
        if (val < min_val->value)
            min_val->value = val;
        if (val > max_val->value)
            max_val->value = val;
    }

    if (max_val->value == min_val->value)
    {
        LOG_ERROR("Max and min are equal");
        return NULL;
    }

    // (x - min) / (max - min)
    Node *range = sub(max_val, min_val);
    Node *normalized = div(sub(x, min_val), range);
    normalized->requires_grad = x->requires_grad;

    return normalized;
}