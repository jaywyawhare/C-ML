#ifndef DROPOUT_H
#define DROPOUT_H

/**
 * @brief Structure representing a Dropout Layer.
 *
 * @param dropout_rate Dropout rate (0.0 to 1.0).
 */
typedef struct
{
    float dropout_rate;
} DropoutLayer;

/**
 * @brief Initializes a Dropout Layer with a given dropout rate.
 *
 * @param layer Pointer to the DropoutLayer structure.
 * @param dropout_rate Dropout rate (0.0 to 1.0).
 * @return int Error code.
 */
int initialize_dropout(DropoutLayer *layer, float dropout_rate);

/**
 * @brief Performs the forward pass for the Dropout Layer.
 *
 * @param layer Pointer to the DropoutLayer structure.
 * @param input Input data array.
 * @param output Output data array.
 * @param size Size of the input/output arrays.
 * @return int Error code.
 */
int forward_dropout(DropoutLayer *layer, float *input, float *output, int size);

/**
 * @brief Performs the backward pass for the Dropout Layer.
 *
 * @param layer Pointer to the DropoutLayer structure.
 * @param input Input data array.
 * @param output Output data array.
 * @param d_output Gradient of the output.
 * @param d_input Gradient of the input.
 * @param size Size of the input/output arrays.
 * @return int Error code.
 */
int backward_dropout(DropoutLayer *layer, float *input, float *output, float *d_output, float *d_input, int size);

#endif
