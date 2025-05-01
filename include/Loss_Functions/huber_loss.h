#ifndef HUBER_LOSS_H
#define HUBER_LOSS_H

#include "../Core/autograd.h"

/**
 * @brief Computes the Huber Loss using autograd.
 *
 * @param y Pointer to the ground truth Node.
 * @param yHat Pointer to the predicted Node.
 * @param n The number of elements in y and yHat.
 * @param delta_val delta value
 * @return The computed loss Node, or NULL if inputs are invalid.
 */

Node *huber_loss(Node *y, Node *yHat, int n, float delta_val);

#endif // HUBER_LOSS_H
