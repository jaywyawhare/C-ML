#include <math.h>
#include <stdio.h>
#include "../../include/Loss_Functions/smooth_l1_loss.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"
#include "../../include/Core/autograd.h"

Node *smooth_l1_loss(Node *y, Node *yHat, int n, float beta_val)
{
    if (!y || !yHat || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return NULL;
    }

    Node *loss = tensor(0.0f, 1);
    Node *beta = tensor(beta_val, 0);
    Node *half = tensor(0.5f, 0);

    for (int i = 0; i < n; i++)
    {
        Node *diff = tensor_sub(yHat, y);
        Node *abs_diff = tensor_abs(diff);
        Node *squared = tensor_mul(diff, diff);
        Node *condition = tensor_sub(abs_diff, beta);

        Node *smooth_term = tensor_mul(half, tensor_div(squared, beta));
        Node *l1_term = tensor_sub(abs_diff, tensor_mul(half, beta));

        loss = tensor_add(loss, condition->tensor->storage->data[0] < 0 ? smooth_term : l1_term);
    }

    return tensor_div(loss, tensor((float)n, 0));
}
