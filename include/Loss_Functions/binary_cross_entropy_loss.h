#ifndef BINARY_CROSS_ENTROPY_LOSS_H
#define BINARY_CROSS_ENTROPY_LOSS_H

#include "../Core/autograd.h"

/**
 * @brief Computes the Binary Cross-Entropy Loss using autograd.
 *
 * @param y Pointer to the ground truth Node.
 * @param yHat Pointer to the predicted Node.
 * @param n The number of elements in y and yHat.
 * @return The computed loss Node, or NULL if inputs are invalid.
 */
Node *binary_cross_entropy_loss(Node *y, Node *yHat, int n);

#endif
