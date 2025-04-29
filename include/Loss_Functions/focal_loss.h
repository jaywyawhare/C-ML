#ifndef FOCAL_LOSS_H
#define FOCAL_LOSS_H

#include "../Core/autograd.h"

/**
 * @brief Computes the Focal Loss using autograd.
 *
 * @param y Pointer to the ground truth Node.
 * @param yHat Pointer to the predicted Node.
 * @param n The number of elements in y and yHat.
 * @param gamma The focusing parameter Node.
 * @return The computed loss Node, or NULL if inputs are invalid.
 */
Node *focal_loss(Node *y, Node *yHat, int n, Node *gamma);

#endif
