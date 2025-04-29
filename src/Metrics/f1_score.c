#include "../../include/Core/autograd.h"
#include "../../include/Metrics/f1_score.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"

Node *f1_score(Node *y, Node *yHat, int n, float threshold)
{
    if (!y || !yHat || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return NULL;
    }

    Node *tp = tensor(0.0f, 1);
    Node *fp = tensor(0.0f, 1);
    Node *fn = tensor(0.0f, 1);

    for (int i = 0; i < n; i++)
    {
        Node *y_i = tensor(y->tensor->storage->data[i], 1);
        Node *yhat_i = tensor(yHat->tensor->storage->data[i], 1);
        Node *pred = tensor(yhat_i->value > threshold ? 1.0f : 0.0f, 1);

        if (y_i->value == 1.0f && pred->value == 1.0f)
            tp = add(tp, tensor(1.0f, 1));
        else if (y_i->value == 0.0f && pred->value == 1.0f)
            fp = add(fp, tensor(1.0f, 1));
        else if (y_i->value == 1.0f && pred->value == 0.0f)
            fn = add(fn, tensor(1.0f, 1));
    }

    Node *precision = div(tp, add(tp, fp));
    Node *recall = div(tp, add(tp, fn));

    return div(mul(mul(tensor(2.0f, 1), precision), recall),
               add(precision, recall));
}
