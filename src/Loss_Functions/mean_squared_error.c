#include <math.h>
#include <stdio.h>
#include "../../include/Loss_Functions/mean_squared_error.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"
#include "../../include/Core/autograd.h"

Node *mean_squared_error(Node *y, Node *yHat, int n)
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
        Node *squared = mul(diff, diff);
        sum = add(sum, squared);
    }

    return div(sum, tensor((float)n, 0));
}
