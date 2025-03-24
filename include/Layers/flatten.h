#ifndef FLATTEN_H
#define FLATTEN_H

#include <stdio.h>

/**
 * @brief Structure representing a Flatten Layer.
 *
 * @param input_size Size of the input data.
 * @param output_size Size of the output data (same as input size).
 */
typedef struct
{
    int input_size;
    int output_size;
} FlattenLayer;

/**
 * @brief Initializes a Flatten Layer.
 *
 * @param layer Pointer to the FlattenLayer structure.
 * @param input_size Size of the input data.
 * @return int Error code.
 */
int initialize_flatten(FlattenLayer *layer, int input_size);

/**
 * @brief Performs the forward pass for the Flatten Layer.
 *
 * @param layer Pointer to the FlattenLayer structure.
 * @param input Input data array.
 * @param output Output data array.
 * @return int Error code.
 */
int forward_flatten(FlattenLayer *layer, float *input, float *output);

/**
 * @brief Performs the backward pass for the Flatten Layer.
 *
 * @param layer Pointer to the FlattenLayer structure.
 * @param input Input data array.
 * @param output Output data array.
 * @param d_output Gradient of the output.
 * @param d_input Gradient of the input.
 * @return int Error code.
 */
int backward_flatten(FlattenLayer *layer, float *input, float *output, float *d_output, float *d_input);

/**
 * @brief Frees the memory allocated for the Flatten Layer.
 *
 * @param layer Pointer to the FlattenLayer structure.
 * @return int Error code.
 */
int free_flatten(FlattenLayer *layer);

#endif