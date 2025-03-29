#include <stdio.h>
#include "../../include/Metrics/iou.h"
#include "../../include/Core/error_codes.h"

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
float iou(float *y, float *yHat, int n, float threshold)
{
    if (!y || !yHat || n <= 0)
    {
        fprintf(stderr, "[iou] Error: Invalid input parameters.\n");
        return CM_INVALID_INPUT_ERROR;
    }
    float intersection = 0.0f, union_area = 0.0f;
    for (int i = 0; i < n; i++)
    {
        int actual = (int)y[i];
        int pred = yHat[i] > threshold ? 1 : 0;
        intersection += actual * pred;
        union_area += actual + pred - (actual * pred);
    }
    if (union_area == 0)
    {
        fprintf(stderr, "[iou] Error: Division by zero.\n");
        return 0.0f;
    }
    return intersection / union_area;
}
