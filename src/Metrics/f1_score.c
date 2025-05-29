#include "../../include/Core/autograd.h"
#include "../../include/Metrics/f1_score.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/memory_management.h"
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
        float actual = y->tensor->storage->data[i];
        float pred_val = yHat->tensor->storage->data[i] > threshold ? 1.0f : 0.0f;
        Node *pred = tensor(pred_val, 1);

        if (actual == 1.0f && pred_val == 1.0f)
            tp = add(tp, tensor(1.0f, 1));
        else if (actual == 0.0f && pred_val == 1.0f)
            fp = add(fp, tensor(1.0f, 1));
        else if (actual == 1.0f && pred_val == 0.0f)
            fn = add(fn, tensor(1.0f, 1));

        cm_safe_free((void **)&pred);
    }

    Node *precision = div_tensor(tp, add(tp, fp));
    Node *recall = div_tensor(tp, add(tp, fn));

    return div_tensor(mul(mul(tensor(2.0f, 1), precision), recall),
                      add(precision, recall));
}
