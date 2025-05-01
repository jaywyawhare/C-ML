#include <math.h>
#include <stdio.h>
#include "../../include/Loss_Functions/tversky_loss.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"
#include "../../include/Core/autograd.h"

Node *tversky_loss(Node *y, Node *yHat, int n)
{
    if (!y || !yHat || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return NULL;
    }

    Node *tp = tensor(0.0f, 1);
    Node *fp = tensor(0.0f, 1);
    Node *fn = tensor(0.0f, 1);
    Node *one = tensor(1.0f, 0);
    Node *alpha = tensor(0.5f, 0);
    Node *beta = tensor(0.5f, 0);

    for (int i = 0; i < n; i++)
    {
        tp = tensor_add(tp, tensor_mul(y, yHat));
        fp = tensor_add(fp, tensor_mul(tensor_sub(one, y), yHat));
        fn = tensor_add(fn, tensor_mul(y, tensor_sub(one, yHat)));
    }

    Node *denominator = tensor_add(tp, tensor_add(tensor_mul(alpha, fp), tensor_mul(beta, fn)));
    return tensor_sub(one, tensor_div(tp, denominator));
}