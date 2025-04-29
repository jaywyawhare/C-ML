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
        tp = add(tp, mul(y, yHat));
        fp = add(fp, mul(sub(one, y), yHat));
        fn = add(fn, mul(y, sub(one, yHat)));
    }

    Node *denominator = add(tp, add(mul(alpha, fp), mul(beta, fn)));
    return sub(one, div(tp, denominator));
}