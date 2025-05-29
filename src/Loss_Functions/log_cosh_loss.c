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
        Node *diff = sub(yHat, y);
        Node *exp_plus = exp_tensor(diff);
        Node *exp_minus = exp_tensor(neg(diff));
        Node *cosh_val = mul(tensor(0.5f, 0), add(exp_plus, exp_minus));
        Node *log_val = log_tensor(cosh_val);
        sum = add(sum, log_val);
    }

    return div_tensor(sum, tensor((float)n, 0));
}
