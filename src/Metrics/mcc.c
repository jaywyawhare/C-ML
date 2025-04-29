#include "../../include/Metrics/mcc.h"
#include "../../include/Core/autograd.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"

/**
 * @brief Computes the Matthews Correlation Coefficient (MCC).
 *
 * The MCC is a measure of the quality of binary classifications.
 * It is defined as:
 * - mcc = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
 *
 * @param y Pointer to the ground truth labels.
 * @param yHat Pointer to the predicted labels.
 * @param n The number of elements in y and yHat.
 * @param threshold The threshold for binary classification.
 * @return The computed MCC, or an error code if inputs are invalid.
 */
Node *mcc(Node *y, Node *yHat, int n, float threshold)
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

    Node *numerator = sub(mul(tp, tn), mul(fp, fn));
    Node *denominator = pow(
        mul(
            mul(add(tp, fp), add(tp, fn)),
            mul(add(tn, fp), add(tn, fn))),
        tensor(0.5f, 1));

    return div(numerator, denominator);
}
