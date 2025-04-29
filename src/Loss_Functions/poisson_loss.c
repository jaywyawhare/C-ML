#include <math.h>
#include <stdio.h>
#include "../../include/Loss_Functions/poisson_loss.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"
#include "../../include/Core/autograd.h"

Node *poisson_loss(Node *y, Node *yHat, int n)
{
    if (!y || !yHat || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return NULL;
    }

    Node *sum = tensor(0.0f, 1);
    Node *epsilon = tensor(1e-8, 0);

    for (int i = 0; i < n; i++)
    {
        Node *yhat_safe = add(yHat, epsilon);
        Node *term1 = yhat_safe;
        Node *term2 = mul(y, log(yhat_safe));
        sum = add(sum, sub(term1, term2));
    }

    return div(sum, tensor((float)n, 0));
}
