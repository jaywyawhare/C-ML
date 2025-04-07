#ifndef INPUT_H
#define INPUT_H

#include "../../include/Core/memory_management.h"

/**
 * @brief Structure representing an Input Layer.
 *
 * @param input_size Size of the input data.
 */
typedef struct
{
    int input_size;
} InputLayer;

/**
 * @brief Initializes an Input Layer.
 *
 * @param layer Pointer to the InputLayer structure.
 * @param input_size Size of the input.
 * @return int Error code.
 */
int initialize_input(InputLayer *layer, int input_size);

/**
 * @brief Performs the forward pass for the Input Layer.
 *
 * @param layer Pointer to the InputLayer structure.
 * @param input Input data array.
 * @param output Output data array.
 * @return int Error code.
 */
int forward_input(InputLayer *layer, float *input, float *output);

/**
 * @brief Performs the backward pass for the Input Layer.
 *
 * @param layer Pointer to the InputLayer structure.
 * @param input Input data array.
 * @param output Output data array.
 * @param d_output Gradient of the output.
 * @param d_input Gradient of the input.
 * @return int Error code.
 */
int backward_input(InputLayer *layer, float *input, float *output, float *d_output, float *d_input);

/**
 * @brief Frees the memory allocated for the Input Layer.
 *
 * @param layer Pointer to the InputLayer structure.
 * @return int Error code.
 */
int free_input(InputLayer *layer);

#endif
