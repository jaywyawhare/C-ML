#ifndef SGD_H
#define SGD_H

/**
 * @brief Performs the Stochastic Gradient Descent (SGD) optimization algorithm.
 *
 * @param x The input feature value.
 * @param y The target value.
 * @param lr The learning rate.
 * @param w Pointer to the weight parameter.
 * @param b Pointer to the bias parameter.
 * @return The computed loss value, or an error code.
 */
float sgd(float x, float y, float lr, float *w, float *b);

#endif
