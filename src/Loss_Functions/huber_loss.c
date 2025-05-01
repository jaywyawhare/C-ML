#include <math.h>
#include <stdio.h>
#include "../../include/Loss_Functions/huber_loss.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"
#include "../../include/Core/autograd.h"

Node *huber_loss(Node *y, Node *yHat, int n, float delta_val)
{
    if (!y || !yHat || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return NULL;
    }

    Node *loss = tensor(0.0f, 1);
    Node *delta = tensor(delta_val, 0);
    Node *half = tensor(0.5f, 0);

    for (int i = 0; i < n; i++)
    {
        Node *diff = tensor_sub(yHat, y);
        Node *abs_diff = tensor_abs(diff);

        Node *condition = tensor_sub(abs_diff, delta);

        if (condition->tensor->storage->data[0] < 0)
        {
            Node *diff_squared = tensor_mul(diff, diff);
            loss = tensor_add(loss, tensor_mul(half, diff_squared));
        }
        else
        {
            loss = tensor_add(loss, tensor_sub(tensor_mul(delta, abs_diff), tensor_mul(half, tensor_mul(delta, delta))));
        }
    }

    return tensor_div(loss, tensor((float)n, 0));
}
