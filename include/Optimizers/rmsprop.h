#ifndef RMSPROP_H
#define RMSPROP_H

/**
 * @brief Performs the RMSProp optimization algorithm.
 *
 * @param x The input feature value.
 * @param y The target value.
 * @param lr The learning rate.
 * @param w Pointer to the weight parameter.
 * @param b Pointer to the bias parameter.
 * @param cache_w Pointer to the cache for the weight parameter.
 * @param cache_b Pointer to the cache for the bias parameter.
 * @param epsilon A small constant to prevent division by zero.
 * @param beta The decay rate for the moving average of squared gradients.
 * @return The computed loss value, or an error code.
 */
float rms_prop(float x, float y, float lr, float *w, float *b, float *cache_w, float *cache_b, float epsilon, float beta);

/**
 * @brief Update weights and biases using RMSProp optimizer.
 *
 * @param w Pointer to the weight.
 * @param b Pointer to the bias.
 * @param cache_w Pointer to the weight cache.
 * @param cache_b Pointer to the bias cache.
 * @param gradient Gradient value.
 * @param input Input value.
 * @param learning_rate Learning rate.
 * @param beta Decay rate for RMSProp.
 * @param epsilon Small value to prevent division by zero.
 */
void update_rmsprop(float *w, float *b, float *cache_w, float *cache_b, float gradient, float input, float learning_rate, float beta, float epsilon);

#endif
