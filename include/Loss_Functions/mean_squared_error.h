#ifndef MEAN_SQUARED_ERROR_H
#define MEAN_SQUARED_ERROR_H

#include "../Core/autograd.h"

/**
 * @brief Computes the Mean Squared Error using autograd.
 *
 * @param y Pointer to the ground truth Node.
 * @param yHat Pointer to the predicted Node.
 * @param n The number of elements in y and yHat.
 * @return The computed loss Node, or NULL if inputs are invalid.
 */
Node *mean_squared_error(Node *y, Node *yHat, int n);

#endif
