#ifndef KLD_LOSS_H
#define KLD_LOSS_H

/**
 * @brief Computes the Kullback-Leibler Divergence Loss.
 *
 * @param p Pointer to the true distribution.
 * @param q Pointer to the predicted distribution.
 * @param n The number of elements in p and q.
 * @return The computed loss, or an error code if inputs are invalid.
 */
float kld_loss(float *p, float *q, int n);

/**
 * @brief Computes the derivative of the Kullback-Leibler Divergence Loss.
 *
 * @param p True distribution value.
 * @param q Predicted distribution value.
 * @return The derivative value.
 */
float kld_loss_derivative(float p, float q);

#endif
