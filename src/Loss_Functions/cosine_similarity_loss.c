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
        dot_product = tensor_add(dot_product, tensor_mul(y, yHat));
        norm_y = tensor_add(norm_y, tensor_mul(y, y));
        norm_yHat = tensor_add(norm_yHat, tensor_mul(yHat, yHat));
    }

    norm_y = tensor_pow(norm_y, tensor(0.5f, 0));
    norm_yHat = tensor_pow(norm_yHat, tensor(0.5f, 0));

    return tensor_sub(tensor(1.0f, 0), tensor_div(dot_product, tensor_mul(norm_y, norm_yHat)));
}
