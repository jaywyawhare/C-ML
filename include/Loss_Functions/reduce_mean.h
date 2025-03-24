#ifndef REDUCE_MEAN_H
#define REDUCE_MEAN_H

/**
 * @brief Computes the mean of an array of loss values.
 *
 * @param loss Pointer to the array of loss values.
 * @param size The number of elements in the loss array.
 * @return The computed mean, or an error code if inputs are invalid.
 */
float reduce_mean(float *loss, int size);

#endif
