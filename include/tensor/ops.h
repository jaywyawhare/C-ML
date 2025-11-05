/**
 * @file ops.h
 * @brief Tensor operations and mathematical functions
 *
 * This header provides all the basic tensor operations including:
 * - Element-wise operations (add, sub, mul, div)
 * - Mathematical functions (exp, log, sin, cos)
 * - Activation functions (relu, sigmoid, tanh)
 * - Reduction operations (sum, mean, max, min)
 */

#ifndef CML_TENSOR_OPS_H
#define CML_TENSOR_OPS_H

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Element-wise Operations

/**
 * @brief Element-wise addition: result = a + b
 *
 * @param a First tensor
 * @param b Second tensor
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_add(Tensor* a, Tensor* b);

/**
 * @brief Element-wise subtraction: result = a - b
 *
 * @param a First tensor
 * @param b Second tensor
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_sub(Tensor* a, Tensor* b);

/**
 * @brief Element-wise multiplication: result = a * b
 *
 * @param a First tensor
 * @param b Second tensor
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_mul(Tensor* a, Tensor* b);

/**
 * @brief Element-wise division: result = a / b
 *
 * @param a First tensor
 * @param b Second tensor
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_div(Tensor* a, Tensor* b);

/**
 * @brief Element-wise power: result = a ^ b
 *
 * @param a Base tensor
 * @param b Exponent tensor
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_pow(Tensor* a, Tensor* b);

// Mathematical Functions

/**
 * @brief Element-wise exponential: result = exp(a)
 *
 * @param a Input tensor
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_exp(Tensor* a);

/**
 * @brief Element-wise natural logarithm: result = ln(a)
 *
 * @param a Input tensor
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_log(Tensor* a);

/**
 * @brief Element-wise square root: result = sqrt(a)
 *
 * @param a Input tensor
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_sqrt(Tensor* a);

/**
 * @brief Element-wise sine: result = sin(a)
 *
 * @param a Input tensor
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_sin(Tensor* a);

/**
 * @brief Element-wise cosine: result = cos(a)
 *
 * @param a Input tensor
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_cos(Tensor* a);

/**
 * @brief Element-wise tangent: result = tan(a)
 *
 * @param a Input tensor
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_tan(Tensor* a);

// Activation Functions

/**
 * @brief Rectified Linear Unit: result = max(0, a)
 *
 * @param a Input tensor
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_relu(Tensor* a);

/**
 * @brief Sigmoid activation: result = 1 / (1 + exp(-a))
 *
 * @param a Input tensor
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_sigmoid(Tensor* a);

/**
 * @brief Hyperbolic tangent: result = tanh(a)
 *
 * @param a Input tensor
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_tanh(Tensor* a);

/**
 * @brief Softmax activation along specified dimension
 *
 * @param a Input tensor
 * @param dim Dimension to apply softmax
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_softmax(Tensor* a, int dim);

/**
 * @brief ELU (Exponential Linear Unit): result = x if x > 0, else alpha * (exp(x) - 1)
 *
 * @param a Input tensor
 * @param alpha Alpha parameter (default: 1.0)
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_elu(Tensor* a, float alpha);

/**
 * @brief SELU (Scaled Exponential Linear Unit): scaled ELU with fixed parameters
 *
 * @param a Input tensor
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_selu(Tensor* a);

/**
 * @brief Swish activation: result = x * sigmoid(x)
 *
 * @param a Input tensor
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_swish(Tensor* a);

/**
 * @brief Mish activation: result = x * tanh(softplus(x))
 *
 * @param a Input tensor
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_mish(Tensor* a);

/**
 * @brief Hard Swish activation: result = x * ReLU6(x + 3) / 6
 *
 * @param a Input tensor
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_hard_swish(Tensor* a);

// Reduction Operations

/**
 * @brief Sum of tensor elements along specified dimension
 *
 * @param a Input tensor
 * @param dim Dimension to reduce (use -1 for all dimensions)
 * @param keepdim Whether to keep the reduced dimension
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_sum(Tensor* a, int dim, bool keepdim);

/**
 * @brief Mean of tensor elements along specified dimension
 *
 * @param a Input tensor
 * @param dim Dimension to reduce (use -1 for all dimensions)
 * @param keepdim Whether to keep the reduced dimension
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_mean(Tensor* a, int dim, bool keepdim);

/**
 * @brief Maximum of tensor elements along specified dimension
 *
 * @param a Input tensor
 * @param dim Dimension to reduce (use -1 for all dimensions)
 * @param keepdim Whether to keep the reduced dimension
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_max(Tensor* a, int dim, bool keepdim);

/**
 * @brief Minimum of tensor elements along specified dimension
 *
 * @param a Input tensor
 * @param dim Dimension to reduce (use -1 for all dimensions)
 * @param keepdim Whether to keep the reduced dimension
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_min(Tensor* a, int dim, bool keepdim);

// Matrix Operations

/**
 * @brief Matrix multiplication: result = a @ b
 *
 * @param a First matrix
 * @param b Second matrix
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_matmul(Tensor* a, Tensor* b);

/**
 * @brief Transpose tensor along specified dimensions
 *
 * @param a Input tensor
 * @param dim1 First dimension to transpose
 * @param dim2 Second dimension to transpose
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_transpose(Tensor* a, int dim1, int dim2);

/**
 * @brief Reshape tensor to new shape
 *
 * @param a Input tensor
 * @param new_shape New shape array
 * @param new_ndim Number of dimensions in new shape
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_reshape(Tensor* a, int* new_shape, int new_ndim);

// Utility Operations

/**
 * @brief Clone tensor (deep copy)
 *
 * @param a Input tensor
 * @return New tensor with copied data, or NULL on failure
 */
Tensor* tensor_clone(Tensor* a);

/**
 * @brief Detach tensor from computation graph
 *
 * @param a Input tensor
 * @return New tensor detached from graph, or NULL on failure
 */
Tensor* tensor_detach(Tensor* a);

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

/**
 * @brief Check if tensor requires gradients
 *
 * @param a Input tensor
 * @return true if tensor requires gradients, false otherwise
 */
bool tensor_requires_grad(Tensor* a);

/**
 * @brief Check if tensor has gradients
 *
 * @param a Input tensor
 * @return true if tensor has gradients, false otherwise
 */
bool tensor_has_grad(Tensor* a);

#ifdef __cplusplus
}
#endif

#endif // CML_TENSOR_OPS_H
