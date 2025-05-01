#include <math.h>
#include <stdio.h>
#include "../../include/Loss_Functions/log_cosh_loss.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"
#include "../../include/Core/autograd.h"

Node *log_cosh_loss(Node *y, Node *yHat, int n)
{
    if (!y || !yHat || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return NULL;
    }

    Node *sum = tensor(0.0f, 1);

    for (int i = 0; i < n; i++)
    {
        Node *diff = tensor_sub(yHat, y);
        Node *exp_plus = tensor_exp(diff);
        Node *exp_minus = tensor_exp(tensor_neg(diff));
        Node *cosh_val = tensor_mul(tensor(0.5f, 0), tensor_add(exp_plus, exp_minus));
        Node *log_val = tensor_log(cosh_val);
        sum = tensor_add(sum, log_val);
    }

    return tensor_div(sum, tensor((float)n, 0));
}
