#include "../../include/Core/autograd.h"
#include "../../include/Metrics/accuracy.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"

Node *accuracy(Node *y, Node *yHat, int n, float threshold)
{
    if (!y || !yHat || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return NULL;
    }

    Node *matches = tensor(0.0f, 1);
    for (int i = 0; i < n; i++)
    {
        Node *y_i = tensor(y->tensor->storage->data[i], 1);
        Node *yhat_i = tensor(yHat->tensor->storage->data[i], 1);
        Node *pred = tensor(yhat_i->value > threshold ? 1.0f : 0.0f, 1);
        Node *match = tensor(pred->value == y_i->value ? 1.0f : 0.0f, 1);
        matches = add(matches, match);
    }

    return div(matches, tensor((float)n, 1));
}
