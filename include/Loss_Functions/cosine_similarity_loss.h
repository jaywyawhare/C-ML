#ifndef COSINE_SIMILARITY_LOSS_H
#define COSINE_SIMILARITY_LOSS_H

#include "../Core/autograd.h"

/**
 * @brief Computes the Cosine Similarity Loss using autograd.
 *
 * @param y Pointer to the ground truth Node.
 * @param yHat Pointer to the predicted Node.
 * @param n The number of elements in y and yHat.
 * @return The computed loss Node, or NULL if inputs are invalid.
 */
Node* cosine_similarity_loss(Node* y, Node* yHat, int n);

#endif
