#ifndef L1_L2_H
#define L1_L2_H

/**
 * @brief Applies combined L1 and L2 regularization to update gradients.
 *
 * @param w Pointer to the weights array.
 * @param dw Pointer to the gradients array.
 * @param l1 Regularization strength for L1.
 * @param l2 Regularization strength for L2.
 * @param n Number of weights.
 * @return The computed loss or an error code.
 */
float l1_l2(float *w, float *dw, float l1, float l2, int n);

#endif
