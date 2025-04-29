#ifndef KLD_LOSS_H
#define KLD_LOSS_H

#include "../Core/autograd.h"

/**
 * @brief Computes the KL Divergence Loss using autograd.
 *
 * @param p Pointer to the true distribution Node.
 * @param q Pointer to the predicted distribution Node.
 * @param n The number of elements in p and q.
 * @return The computed loss Node, or NULL if inputs are invalid.
 */
Node *kld_loss(Node *p, Node *q, int n);

#endif
