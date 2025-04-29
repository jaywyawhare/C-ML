#include <math.h>
#include <stdio.h>
#include "../../include/Loss_Functions/focal_loss.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"
#include "../../include/Core/autograd.h"

Node *focal_loss(Node *y, Node *yHat, int n, Node *gamma)
{
    if (!y || !yHat || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return NULL;
    }

    Node *epsilon = tensor(1e-8, 0);
    Node *one = tensor(1.0f, 0);
    Node *loss = tensor(0.0f, 1);

    for (int i = 0; i < n; i++)
    {
        Node *yHat_clamped = max(min(yHat, sub(one, epsilon)), epsilon);
        Node *term1 = mul(pow(sub(one, yHat_clamped), gamma), log(yHat_clamped));
        Node *term2 = mul(pow(yHat_clamped, gamma), log(sub(one, yHat_clamped)));
        loss = add(loss, add(mul(y, term1), mul(sub(one, y), term2)));
    }

    return div(loss, tensor((float)n, 0));
}