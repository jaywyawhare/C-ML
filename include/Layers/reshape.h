#ifndef RESHAPE_H
#define RESHAPE_H

#include "../../include/Core/memory_management.h"

/**
 * @brief Structure representing a Reshape Layer.
 *
 * @param input_size Size of the input data.
 * @param output_size Size of the output data.
 */
typedef struct
{
    int input_size;
    int output_size;
} ReshapeLayer;

/**
 * @brief Initializes a Reshape Layer.
 *
 * @param layer Pointer to the ReshapeLayer structure.
 * @param input_size Size of the input data.
 * @param output_size Size of the output data.
 * @return int Error code.
 */
int initialize_reshape(ReshapeLayer *layer, int input_size, int output_size);

/**
 * @brief Performs the forward pass for the Reshape Layer.
 *
 * @param layer Pointer to the ReshapeLayer structure.
 * @param input Input data array.
 * @param output Output data array.
 * @return int Error code.
 */
int forward_reshape(ReshapeLayer *layer, float *input, float *output);

/**
 * @brief Performs the backward pass for the Reshape Layer.
 *
 * @param layer Pointer to the ReshapeLayer structure.
 * @param d_output Gradient of the output.
 * @param d_input Gradient of the input.
 * @return int Error code.
 */
int backward_reshape(ReshapeLayer *layer, float *d_output, float *d_input);

/**
 * @brief Frees the memory allocated for the Reshape Layer.
 *
 * @param layer Pointer to the ReshapeLayer structure.
 * @return int Error code.
 */
int free_reshape(ReshapeLayer *layer);

#endif
