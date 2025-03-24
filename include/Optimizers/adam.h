#ifndef ADAM_H
#define ADAM_H

/**
 * @brief Performs the Adam optimization algorithm.
 *
 * @param x The input feature value.
 * @param y The target value.
 * @param lr The learning rate.
 * @param w Pointer to the weight parameter.
 * @param b Pointer to the bias parameter.
 * @param v_w Pointer to the first moment vector for the weight.
 * @param v_b Pointer to the first moment vector for the bias.
 * @param s_w Pointer to the second moment vector for the weight.
 * @param s_b Pointer to the second moment vector for the bias.
 * @param beta1 The exponential decay rate for the first moment estimates.
 * @param beta2 The exponential decay rate for the second moment estimates.
 * @param epsilon A small constant to prevent division by zero.
 * @return The computed loss value, or an error code.
 */
float adam(float x, float y, float lr, float *w, float *b, float *v_w, float *v_b, float *s_w, float *s_b, float beta1, float beta2, float epsilon);

#endif
