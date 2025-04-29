#ifndef SMOOTH_L1_LOSS_H
#define SMOOTH_L1_LOSS_H

#include "../Core/autograd.h"

/**
 * @brief Computes the Smooth L1 Loss.
 *
 * @param y Pointer to the ground truth Node.
 * @param yHat Pointer to the predicted Node.
 * @param n The number of elements in y and yHat.
 * @return The computed loss Node, or NULL if inputs are invalid.
 */
Node *smooth_l1_loss(Node *y, Node *yHat, int n);

#endif
