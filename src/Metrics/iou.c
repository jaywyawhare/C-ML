#include "../../include/Metrics/iou.h"
#include "../../include/Core/autograd.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"

/**
 * @brief Computes the Intersection over Union (IoU) metric.
 *
 * The IoU is defined as the ratio of the intersection area to the union area.
 *
 * @param y Pointer to the ground truth labels.
 * @param yHat Pointer to the predicted labels.
 * @param n The number of elements in y and yHat.
 * @param threshold The threshold for binary classification.
 * @return The computed IoU, or an error code if inputs are invalid.
 */
Node *iou(Node *y, Node *yHat, int n, float threshold)
{
    if (!y || !yHat || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return NULL;
    }

    Node *intersection = tensor(0.0f, 1);
    Node *union_area = tensor(0.0f, 1);

    for (int i = 0; i < n; i++)
    {
        float actual = y->tensor->storage->data[i];
        float pred = yHat->tensor->storage->data[i] > threshold ? 1.0f : 0.0f;

        intersection = tensor_add(intersection, tensor_mul(tensor(actual, 1), tensor(pred, 1)));
        union_area = tensor_add(union_area,
                                tensor_sub(tensor_add(tensor(actual, 1), tensor(pred, 1)),
                                           tensor_mul(tensor(actual, 1), tensor(pred, 1))));
    }

    return tensor_div(intersection, union_area);
}
