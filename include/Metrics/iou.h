#ifndef IOU_H
#define IOU_H

/**
 * @brief Computes the Intersection over Union (IoU) metric.
 *
 * @param y Pointer to the ground truth labels.
 * @param yHat Pointer to the predicted labels.
 * @param n The number of elements in y and yHat.
 * @param threshold The threshold for binary classification.
 * @return The computed IoU, or an error code if inputs are invalid.
 */
float iou(float *y, float *yHat, int n, float threshold);

#endif
