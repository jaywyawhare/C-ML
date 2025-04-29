#include "../../include/Metrics/r2_score.h"
#include "../../include/Core/autograd.h"

#include <math.h>
#include <stddef.h>

/**
 * @brief Calculate the R2 score (coefficient of determination).
 *
 * @param y_true Array of true values.
 * @param y_pred Array of predicted values.
 * @param size Number of elements in the arrays.
 * @return float The R2 score.
 */
Node *r2_score(Node *y, Node *yHat, int n)
{
    if (!y || !yHat || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return NULL;
    }

    // Calculate mean of y
    Node *sum_y = tensor(0.0f, 1);
    for (int i = 0; i < n; i++)
    {
        sum_y = add(sum_y, tensor(y->tensor->storage->data[i], 1));
    }
    Node *mean_y = div(sum_y, tensor((float)n, 1));

    // Calculate total and residual sum of squares
    Node *ss_tot = tensor(0.0f, 1);
    Node *ss_res = tensor(0.0f, 1);

    for (int i = 0; i < n; i++)
    {
        Node *y_i = tensor(y->tensor->storage->data[i], 1);
        Node *yhat_i = tensor(yHat->tensor->storage->data[i], 1);

        Node *diff_tot = sub(y_i, mean_y);
        Node *diff_res = sub(y_i, yhat_i);

        ss_tot = add(ss_tot, mul(diff_tot, diff_tot));
        ss_res = add(ss_res, mul(diff_res, diff_res));
    }

    return sub(tensor(1.0f, 1), div(ss_res, ss_tot));
}
