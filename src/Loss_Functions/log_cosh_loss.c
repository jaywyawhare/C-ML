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
    Node *one = tensor(1.0f, 0);

    for (int i = 0; i < n; i++)
    {
        Node *diff = sub(yHat, y);
        Node *cosh_val = exp(diff);
        Node *log_val = log(cosh_val);
        sum = add(sum, log_val);
    }

    return div(sum, tensor((float)n, 0));
}
