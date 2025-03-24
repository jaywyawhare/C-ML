#ifndef DENSE_H
#define DENSE_H

/**
 * @brief Structure representing a Dense Layer.
 *
 * @param weights Pointer to the weights matrix.
 * @param biases Pointer to the biases vector.
 * @param input_size Number of input neurons.
 * @param output_size Number of output neurons.
 */
typedef struct
{
    float *weights;
    float *biases;
    int input_size;
    int output_size;
} DenseLayer;

/**
 * @brief Initializes a Dense Layer with random weights and biases.
 *
 * @param layer Pointer to the DenseLayer structure.
 * @param input_size Number of input neurons.
 * @param output_size Number of output neurons.
 * @return int Error code.
 */
int initialize_dense(DenseLayer *layer, int input_size, int output_size);

/**
 * @brief Performs the forward pass for the Dense Layer.
 *
 * @param layer Pointer to the DenseLayer structure.
 * @param input Input data array.
 * @param output Output data array.
 * @return int Error code.
 */
int forward_dense(DenseLayer *layer, float *input, float *output);

/**
 * @brief Performs the backward pass for the Dense Layer.
 *
 * @param layer Pointer to the DenseLayer structure.
 * @param input Input data array.
 * @param output Output data array.
 * @param d_output Gradient of the output.
 * @param d_input Gradient of the input.
 * @param d_weights Gradient of the weights.
 * @param d_biases Gradient of the biases.
 * @return int Error code.
 */
int backward_dense(DenseLayer *layer, float *input, float *output, float *d_output, float *d_input, float *d_weights, float *d_biases);

/**
 * @brief Updates the weights and biases of the Dense Layer.
 *
 * @param layer Pointer to the DenseLayer structure.
 * @param d_weights Gradient of the weights.
 * @param d_biases Gradient of the biases.
 * @param learning_rate Learning rate for the update.
 * @return int Error code.
 */
int update_dense(DenseLayer *layer, float *d_weights, float *d_biases, float learning_rate);

/**
 * @brief Frees the memory allocated for the Dense Layer.
 *
 * @param layer Pointer to the DenseLayer structure.
 * @return int Error code.
 */
int free_dense(DenseLayer *layer);

#endif
