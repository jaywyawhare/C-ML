#include "../../include/Metrics/cohens_kappa.h"
#include "../../include/Core/autograd.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"

/**
 * @brief Computes Cohen's Kappa statistic.
 *
 * Cohen's Kappa is a measure of inter-rater agreement for categorical items.
 * It is defined as:
 * - kappa = (observed_accuracy - expected_accuracy) / (1 - expected_accuracy)
 *
 * @param y Pointer to the ground truth labels.
 * @param yHat Pointer to the predicted labels.
 * @param n The number of elements in y and yHat.
 * @param threshold The threshold for binary classification.
 * @return The computed Cohen's Kappa, or an error code if inputs are invalid.
 */
Node *cohens_kappa(Node *y, Node *yHat, int n, float threshold)
{
    if (!y || !yHat || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return NULL;
    }

    Node *tp = tensor(0.0f, 1);
    Node *tn = tensor(0.0f, 1);
    Node *fp = tensor(0.0f, 1);
    Node *fn = tensor(0.0f, 1);

    for (int i = 0; i < n; i++)
    {
        float actual = y->tensor->storage->data[i];
        float pred = yHat->tensor->storage->data[i] > threshold ? 1.0f : 0.0f;

        if (actual == 1.0f && pred == 1.0f)
            tp = add(tp, tensor(1.0f, 1));
        else if (actual == 0.0f && pred == 0.0f)
            tn = add(tn, tensor(1.0f, 1));
        else if (actual == 0.0f && pred == 1.0f)
            fp = add(fp, tensor(1.0f, 1));
        else if (actual == 1.0f && pred == 0.0f)
            fn = add(fn, tensor(1.0f, 1));
    }

    Node *total = tensor((float)n, 1);
    Node *observed_acc = div(add(tp, tn), total);
    Node *expected_acc = div(
        mul(add(tp, fn), add(tp, fp)) + mul(add(tn, fp), add(tn, fn)),
        mul(total, total));

    return div(
        sub(observed_acc, expected_acc),
        sub(tensor(1.0f, 1), expected_acc));
}
