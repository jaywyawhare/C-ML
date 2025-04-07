#include <stdio.h>
#include "../../include/Metrics/balanced_accuracy.h"
#include "../../include/Metrics/specificity.h"
#include "../../include/Metrics/recall.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"

/**
 * @brief Computes the Balanced Accuracy metric.
 *
 * The Balanced Accuracy is defined as the average of sensitivity (recall) and specificity.
 *
 * @param y Pointer to the ground truth labels.
 * @param yHat Pointer to the predicted labels.
 * @param n The number of elements in y and yHat.
 * @param threshold The threshold for binary classification.
 * @return The computed balanced accuracy, or an error code if inputs are invalid.
 */
float balanced_accuracy(float *y, float *yHat, int n, float threshold)
{
    if (!y || !yHat || n <= 0)
    {
        LOG_ERROR("Invalid input parameters.");
        return CM_INVALID_INPUT_ERROR;
    }

    float sensitivity = recall(y, yHat, n, threshold);
    float spec = specificity(y, yHat, n, threshold);

    return (sensitivity + spec) / 2.0f;
}
