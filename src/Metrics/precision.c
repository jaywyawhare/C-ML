#include "../../include/Metrics/precision.h"
#include "../../include/Core/autograd.h"

/**
 * @brief Computes the Precision metric.
 *
 * The Precision is defined as the ratio of true positives to the sum of true positives and false positives.
 *
 * @param y Pointer to the ground truth labels.
 * @param yHat Pointer to the predicted labels.
 * @param n The number of elements in y and yHat.
 * @param threshold The threshold for binary classification.
 * @return The computed Precision, or an error code if inputs are invalid.
 */
Node *precision(Node *y, Node *yHat, int n, float threshold)
{
    if (!y || !yHat || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return NULL;
    }

    Node *true_positive = tensor(0.0f, 1);
    Node *false_positive = tensor(0.0f, 1);

    for (int i = 0; i < n; i++)
    {
        float actual = y->tensor->storage->data[i];
        float pred = yHat->tensor->storage->data[i] > threshold ? 1.0f : 0.0f;

        if (actual == 1.0f && pred == 1.0f)
            true_positive = add(true_positive, tensor(1.0f, 1));
        else if (actual == 0.0f && pred == 1.0f)
            false_positive = add(false_positive, tensor(1.0f, 1));
    }

    Node *denominator = add(true_positive, false_positive);
    return div(true_positive, denominator);
}
