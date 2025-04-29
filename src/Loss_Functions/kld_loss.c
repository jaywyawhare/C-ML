#include <math.h>
#include <stdio.h>
#include "../../include/Loss_Functions/kld_loss.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"
#include "../../include/Core/autograd.h"

Node *kld_loss(Node *p, Node *q, int n)
{
    if (!p || !q || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return NULL;
    }

    Node *epsilon = tensor(1e-8, 0);
    Node *loss = tensor(0.0f, 1);

    for (int i = 0; i < n; i++)
    {
        Node *p_val = max(p, epsilon);
        Node *q_val = max(q, epsilon);
        loss = add(loss, mul(p_val, log(div(p_val, q_val))));
    }

    return div(loss, tensor((float)n, 0));
}
