#include <stdio.h>
#include "../../include/Metrics/balanced_accuracy.h"
#include "../../include/Metrics/specificity.h"
#include "../../include/Metrics/recall.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"
#include "../../include/Core/autograd.h"

Node *balanced_accuracy(Node *y, Node *yHat, int n, float threshold)
{
    if (!y || !yHat || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return NULL;
    }

    Node *sens = recall(y, yHat, n, threshold);
    Node *spec = specificity(y, yHat, n, threshold);

    Node *sum = tensor_add(sens, spec);
    return tensor_div(sum, tensor(2.0f, 0));
}
