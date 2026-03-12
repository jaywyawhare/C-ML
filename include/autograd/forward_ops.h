#ifndef CML_AUTOGRAD_FORWARD_OPS_H
#define CML_AUTOGRAD_FORWARD_OPS_H

#include "tensor/tensor.h"

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
 * @brief Negation: result = -a
 *
 * @param a Input tensor
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_neg(Tensor* a);

/**
 * @brief Leaky ReLU activation: result = max(negative_slope * a, a)
 *
 * @param a Input tensor
 * @param negative_slope Slope for negative values
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_leaky_relu(Tensor* a, float negative_slope);

/**
 * @brief Power: result = a ^ b
 *
 * @param a Base tensor
 * @param b Exponent tensor
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_pow(Tensor* a, Tensor* b);

/**
 * @brief Sine: result = sin(a)
 *
 * @param a Input tensor
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_sin(Tensor* a);

/**
 * @brief Cosine: result = cos(a)
 *
 * @param a Input tensor
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_cos(Tensor* a);

/**
 * @brief Tangent: result = tan(a)
 *
 * @param a Input tensor
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_tan(Tensor* a);

/**
 * @brief Softmax activation along specified dimension
 *
 * @param a Input tensor
 * @param dim Dimension to apply softmax
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_softmax(Tensor* a, int dim);

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

// Advanced Reduction Operations

/**
 * @brief Standard deviation of tensor elements along specified dimension
 *
 * @param a Input tensor
 * @param dim Dimension to reduce (use -1 for all dimensions)
 * @param unbiased If true, use Bessel's correction (divide by N-1)
 * @param keepdim Whether to keep the reduced dimension
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_std(Tensor* a, int dim, bool unbiased, bool keepdim);

/**
 * @brief Variance of tensor elements along specified dimension
 *
 * @param a Input tensor
 * @param dim Dimension to reduce (use -1 for all dimensions)
 * @param unbiased If true, use Bessel's correction (divide by N-1)
 * @param keepdim Whether to keep the reduced dimension
 * @return New tensor with result, or NULL on failure
 */
Tensor* tensor_var(Tensor* a, int dim, bool unbiased, bool keepdim);

/**
 * @brief Index of maximum element along specified dimension
 *
 * @param a Input tensor
 * @param dim Dimension to reduce (use -1 for all dimensions)
 * @return New INT32 tensor with argmax indices, or NULL on failure
 */
Tensor* tensor_argmax(Tensor* a, int dim);

/**
 * @brief Index of minimum element along specified dimension
 *
 * @param a Input tensor
 * @param dim Dimension to reduce (use -1 for all dimensions)
 * @return New INT32 tensor with argmin indices, or NULL on failure
 */
Tensor* tensor_argmin(Tensor* a, int dim);

// Note: tensor_reshape and tensor_clone are declared in tensor/tensor.h
// Note: tensor_detach is declared in autograd/autograd.h
// and autograd/autograd.h respectively

// Note: tensor_requires_grad is declared in autograd/autograd.h

/**
 * @brief Check if tensor has gradients
 *
 * @param a Input tensor
 * @return true if tensor has gradients, false otherwise
 */
bool tensor_has_grad(Tensor* a);

// Additional Activations

/** @brief ELU activation: x > 0 ? x : alpha*(exp(x)-1) */
Tensor* tensor_elu(Tensor* a, float alpha);

/** @brief SELU activation: scale * elu(x, alpha) */
Tensor* tensor_selu(Tensor* a);

/** @brief Mish activation: x * tanh(softplus(x)) */
Tensor* tensor_mish(Tensor* a);

/** @brief SiLU/Swish activation: x * sigmoid(x) */
Tensor* tensor_silu(Tensor* a);

/** @brief HardSwish activation */
Tensor* tensor_hardswish(Tensor* a);

// Additional Tensor Operations

/** @brief Sort along dimension */
Tensor* tensor_sort(Tensor* a, int dim, bool descending);

/** @brief Top-k values along dimension */
Tensor* tensor_topk(Tensor* a, int k, int dim, bool largest, bool sorted);

/** @brief Select elements where mask is true */
Tensor* tensor_masked_select(Tensor* a, Tensor* mask);

// Note: tensor_split and tensor_chunk are declared in tensor/tensor.h

/** @brief Create coordinate matrices from 1D vectors */
Tensor** tensor_meshgrid(Tensor** tensors, int num_tensors, int* num_outputs);

/** @brief Extract diagonal from tensor */
Tensor* tensor_diagonal(Tensor* a, int offset, int dim1, int dim2);

/** @brief Linear interpolation: a + weight*(b-a) */
Tensor* tensor_lerp(Tensor* a, Tensor* b, float weight);

/** @brief Integer division */
Tensor* tensor_idiv(Tensor* a, Tensor* b);

/** @brief Modulo */
Tensor* tensor_mod(Tensor* a, Tensor* b);

#ifdef __cplusplus
}
#endif

#endif // CML_AUTOGRAD_FORWARD_OPS_H
