#include <math.h>
#include <stdio.h>
#include "../../include/Loss_Functions/binary_cross_entropy_loss.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"
#include "../../include/Core/autograd.h"

Node *binary_cross_entropy_loss(Node *y, Node *yHat, int size)
{
    if (!y || !yHat || size <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return NULL;
    }

    Node *epsilon = tensor(1e-15, 0);
    Node *one = tensor(1.0f, 0);
    Node *loss = tensor(0.0f, 1);

    for (int i = 0; i < size; ++i)
    {
        Node *predicted = max(min(yHat, sub(one, epsilon)), epsilon);
        Node *term1 = mul(y, log(predicted));
        Node *term2 = mul(sub(one, y), log(sub(one, predicted)));
        loss = add(loss, neg(add(term1, term2)));
    }

    return div(loss, tensor((float)size, 0));
}