#ifndef CML_TENSOR_MANIPULATION_H
#define CML_TENSOR_MANIPULATION_H

#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Tensor Manipulation Operations

/**
 * @brief Concatenate tensors along specified dimension
 *
 * @param tensors Array of tensors to concatenate
 * @param num_tensors Number of tensors
 * @param dim Dimension along which to concatenate
 * @return New concatenated tensor, or NULL on failure
 */
Tensor* tensor_concat(Tensor** tensors, int num_tensors, int dim);

/**
 * @brief Stack tensors along new dimension
 *
 * @param tensors Array of tensors to stack
 * @param num_tensors Number of tensors
 * @param dim Dimension along which to stack (inserts new dimension)
 * @return New stacked tensor, or NULL on failure
 */
Tensor* tensor_stack(Tensor** tensors, int num_tensors, int dim);

/**
 * @brief Split tensor into multiple tensors along dimension
 *
 * @param tensor Input tensor
 * @param num_splits Number of splits (or sizes array if split_sizes provided)
 * @param dim Dimension along which to split
 * @param split_sizes Optional array of split sizes (NULL for equal splits)
 * @return Array of split tensors (caller must free), or NULL on failure
 */
Tensor** tensor_split(Tensor* tensor, int num_splits, int dim, int* split_sizes);

/**
 * @brief Gather values from tensor using indices
 *
 * @param input Input tensor
 * @param indices Index tensor
 * @param dim Dimension along which to gather
 * @return New tensor with gathered values, or NULL on failure
 */
Tensor* tensor_gather(Tensor* input, Tensor* indices, int dim);

/**
 * @brief Scatter values into tensor at specified indices
 *
 * @param input Input tensor
 * @param dim Dimension along which to scatter
 * @param index Index tensor
 * @param src Source tensor with values to scatter
 * @return New tensor with scattered values, or NULL on failure
 */
Tensor* tensor_scatter(Tensor* input, int dim, Tensor* index, Tensor* src);

#ifdef __cplusplus
}
#endif

#endif // CML_TENSOR_MANIPULATION_H
