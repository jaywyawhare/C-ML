#include "../../include/Metrics/specificity.h"
#include "../../include/Core/autograd.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"

/**
 * @brief Computes the specificity metric.
 *
 * Specificity is defined as the ratio of true negatives to the sum of true negatives and false positives.
 *
 * @param y Pointer to the ground truth labels.
 * @param yHat Pointer to the predicted labels.
 * @param n The number of elements in y and yHat.
 * @param threshold The threshold for binary classification.
 * @return The computed specificity, or an error code if inputs are invalid.
 */
Node *specificity(Node *y, Node *yHat, int n, float threshold)
{
    if (!y || !yHat || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return NULL;
    }

    Node *true_negative = tensor(0.0f, 1);
    Node *false_positive = tensor(0.0f, 1);

    for (int i = 0; i < n; i++)
    {
        float actual = y->tensor->storage->data[i];
        float pred = yHat->tensor->storage->data[i] > threshold ? 1.0f : 0.0f;

        if (actual == 0.0f && pred == 0.0f)
            true_negative = add(true_negative, tensor(1.0f, 1));
        else if (actual == 0.0f && pred == 1.0f)
            false_positive = add(false_positive, tensor(1.0f, 1));
    }

    Node *denominator = add(true_negative, false_positive);
    return div(true_negative, denominator);
}
