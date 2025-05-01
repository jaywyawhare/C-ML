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

    Node *epsilon = tensor(1e-7, 0);
    Node *one = tensor(1.0f, 0);
    Node *loss = tensor(0.0f, 1);

    for (int i = 0; i < n; i++)
    {
        Node *yHat_clamped = tensor_max(tensor_min(yHat, tensor_sub(one, epsilon)), epsilon);
        Node *term1 = tensor_mul(tensor_pow(tensor_sub(one, yHat_clamped), gamma), tensor_log(yHat_clamped));
        Node *term2 = tensor_mul(tensor_pow(yHat_clamped, gamma), tensor_log(tensor_sub(one, yHat_clamped)));
        loss = tensor_add(loss, tensor_add(tensor_mul(y, term1), tensor_mul(tensor_sub(one, y), term2)));
    }

    return tensor_div(loss, tensor((float)n, 0));
}