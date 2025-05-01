#include "../../include/Metrics/recall.h"
#include "../../include/Core/autograd.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"

/**
 * @brief Computes the Recall metric.
 *
 * The Recall is defined as the ratio of true positives to the sum of true positives and false negatives.
 *
 * @param y Pointer to the ground truth labels.
 * @param yHat Pointer to the predicted labels.
 * @param n The number of elements in y and yHat.
 * @param threshold The threshold for binary classification.
 * @return The computed Recall, or an error code if inputs are invalid.
 */
Node *recall(Node *y, Node *yHat, int n, float threshold)
{
    if (!y || !yHat || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return NULL;
    }

    Node *true_positive = tensor(0.0f, 1);
    Node *false_negative = tensor(0.0f, 1);

    for (int i = 0; i < n; i++)
    {
        float actual = y->tensor->storage->data[i];
        float pred = yHat->tensor->storage->data[i] > threshold ? 1.0f : 0.0f;

        if (actual == 1.0f && pred == 1.0f)
            true_positive = tensor_add(true_positive, tensor(1.0f, 1));
        else if (actual == 1.0f && pred == 0.0f)
            false_negative = tensor_add(false_negative, tensor(1.0f, 1));
    }

    Node *denominator = tensor_add(true_positive, false_negative);
    return tensor_div(true_positive, denominator);
}
