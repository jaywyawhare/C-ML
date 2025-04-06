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

/**
 * @brief Update weights and biases using SGD optimizer.
 *
 * @param w Pointer to the weight.
 * @param b Pointer to the bias.
 * @param gradient Gradient value.
 * @param input Input value.
 * @param learning_rate Learning rate.
 */
void update_sgd(float *w, float *b, float gradient, float input, float learning_rate);

#endif
