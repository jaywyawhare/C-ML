#include <math.h>
#include <stdio.h>
#include "../../include/Loss_Functions/cosine_similarity_loss.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"
#include "../../include/Core/autograd.h"

Node *cosine_similarity_loss(Node *y, Node *yHat, int n)
{
    if (!y || !yHat || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return NULL;
    }

    Node *dot_product = tensor(0.0f, 1);
    Node *norm_y = tensor(0.0f, 1);
    Node *norm_yHat = tensor(0.0f, 1);

    for (int i = 0; i < n; i++)
    {
        dot_product = add(dot_product, mul(y, yHat));
        norm_y = add(norm_y, mul(y, y));
        norm_yHat = add(norm_yHat, mul(yHat, yHat));
    }

    norm_y = pow(norm_y, tensor(0.5f, 0));
    norm_yHat = pow(norm_yHat, tensor(0.5f, 0));

    return sub(tensor(1.0f, 0), div(dot_product, mul(norm_y, norm_yHat)));
}
