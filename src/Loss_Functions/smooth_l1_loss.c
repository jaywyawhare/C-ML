#include <math.h>
#include <stdio.h>
#include "../../include/Loss_Functions/smooth_l1_loss.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"
#include "../../include/Core/autograd.h"

Node *smooth_l1_loss(Node *y, Node *yHat, int n)
{
    if (!y || !yHat || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return NULL;
    }

    Node *loss = tensor(0.0f, 1);
    Node *beta = tensor(1.0f, 0);
    Node *half = tensor(0.5f, 0);

    for (int i = 0; i < n; i++)
    {
        Node *diff = sub(yHat, y);
        Node *abs_diff = abs(diff);
        Node *squared = mul(diff, diff);

        Node *condition = sub(abs_diff, beta);
        Node *smooth_term = mul(half, div(squared, beta));
        Node *l1_term = sub(abs_diff, mul(half, beta));

        loss = add(loss, condition->value < 0 ? smooth_term : l1_term);
    }

    return div(loss, tensor((float)n, 0));
}
