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
            tp = tensor_add(tp, tensor(1.0f, 1));
        else if (actual == 0.0f && pred == 0.0f)
            tn = tensor_add(tn, tensor(1.0f, 1));
        else if (actual == 0.0f && pred == 1.0f)
            fp = tensor_add(fp, tensor(1.0f, 1));
        else if (actual == 1.0f && pred == 0.0f)
            fn = tensor_add(fn, tensor(1.0f, 1));
    }

    Node *total = tensor((float)n, 0);

    Node *numerator = tensor_sub(
        tensor_mul(tensor_add(tp, tn), total),
        tensor_add(
            tensor_mul(tensor_add(tp, fn), tensor_add(tp, fp)),
            tensor_mul(tensor_add(tn, fp), tensor_add(tn, fn))));

    Node *denominator = tensor_sub(
        tensor_mul(total, total),
        tensor_add(
            tensor_mul(tensor_add(tp, fn), tensor_add(tp, fp)),
            tensor_mul(tensor_add(tn, fp), tensor_add(tn, fn))));

    return tensor_div(numerator, denominator);
}
