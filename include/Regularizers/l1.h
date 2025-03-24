#ifndef L1_H
#define L1_H

/**
 * @brief Applies L1 regularization to update weights and biases.
 *
 * @param x Input feature value.
 * @param y Target value.
 * @param lr Learning rate.
 * @param w Pointer to the weight.
 * @param b Pointer to the bias.
 * @param v_w Pointer to the velocity of the weight.
 * @param v_b Pointer to the velocity of the bias.
 * @param s_w Pointer to the squared gradient of the weight.
 * @param s_b Pointer to the squared gradient of the bias.
 * @param beta1 Exponential decay rate for the first moment estimates.
 * @param beta2 Exponential decay rate for the second moment estimates.
 * @param epsilon Small constant to prevent division by zero.
 * @return The computed loss or an error code.
 */
float l1(float x, float y, float lr, float *w, float *b, float *v_w, float *v_b, float *s_w, float *s_b, float beta1, float beta2, float epsilon);

#endif
