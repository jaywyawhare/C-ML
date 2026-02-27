/**
 * @file uops.h
 * @brief Micro-Operations (UOps) - Minimal set of fundamental operations
 *
 * Inspired by tinygrad and ggml, C-ML provides a minimal set of uops
 * that can be composed to build complex operations.
 *
 * Philosophy:
 * - Minimal set of fundamental operations
 * - High-level layers built from uops
 * - IR generation for different accelerators
 * - Direct uops access for researchers
 */

#ifndef CML_CORE_UOPS_H
#define CML_CORE_UOPS_H

#include "tensor/tensor.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief UOp type enumeration
 */
typedef enum {
    // Elementwise Binary Ops
    UOP_ADD = 0, // a + b
    UOP_SUB,     // a - b
    UOP_MUL,     // a * b
    UOP_DIV,     // a / b
    UOP_MAX,     // max(a, b)
    UOP_CMPLT,   // a < b (returns 0 or 1)

    // Elementwise Unary Ops
    UOP_NEG,     // -a
    UOP_EXP,     // exp(a)
    UOP_LOG,     // log(a)
    UOP_SQRT,    // sqrt(a)
    UOP_RECIP,   // 1 / a
    UOP_ABS,     // |a|
    UOP_SIN,     // sin(a)
    UOP_COS,     // cos(a)
    UOP_TAN,     // tan(a)
    UOP_TANH,    // tanh(a)
    UOP_SIGMOID, // sigmoid(a)
    UOP_POW,     // a ^ b

    // Reduction Ops
    UOP_SUM,        // sum along dimension(s)
    UOP_MAX_REDUCE, // max along dimension(s)
    UOP_MEAN,       // mean along dimension(s)

    // Movement Ops (views, no data copy)
    UOP_RESHAPE, // reshape (view)
    UOP_PERMUTE, // permute dimensions (view)
    UOP_EXPAND,  // broadcast to shape (view)
    UOP_STRIDE,  // change stride (view)
    UOP_SLICE,   // slice tensor (view)

    // Special Ops
    UOP_MATMUL, // matrix multiplication
    UOP_CONV2D, // 2D convolution
    UOP_WHERE,  // conditional: where(cond, a, b)
    UOP_FILL,   // fill tensor with constant value
    UOP_GATHER, // gather elements by index: out[i] = input[i, indices[i]]

    UOP_COUNT // Total count
} UOpType;

/**
 * @brief UOp execution context
 */
typedef struct {
    UOpType type;
    Tensor** inputs;
    int num_inputs;
    Tensor* output;
    void* params; // Operation-specific parameters
} UOp;

/**
 * @brief Execute a single uop
 *
 * @param uop UOp to execute
 * @return 0 on success, negative on failure
 */
int uop_execute(UOp* uop);

/**
 * @brief Create and execute uop (convenience function)
 */
Tensor* uop_create_and_execute(UOpType type, Tensor** inputs, int num_inputs, void* params);

/**
 * @brief Element-wise addition
 * @param a First tensor
 * @param b Second tensor
 * @return Result tensor
 */
Tensor* uop_add(Tensor* a, Tensor* b);

/**
 * @brief Element-wise subtraction
 * @param a First tensor
 * @param b Second tensor
 * @return Result tensor
 */
Tensor* uop_sub(Tensor* a, Tensor* b);

/**
 * @brief Element-wise multiplication
 * @param a First tensor
 * @param b Second tensor
 * @return Result tensor
 */
Tensor* uop_mul(Tensor* a, Tensor* b);

/**
 * @brief Element-wise division
 * @param a First tensor
 * @param b Second tensor
 * @return Result tensor
 */
Tensor* uop_div(Tensor* a, Tensor* b);

/**
 * @brief Element-wise maximum
 * @param a First tensor
 * @param b Second tensor
 * @return Result tensor
 */
Tensor* uop_max(Tensor* a, Tensor* b);

/**
 * @brief Element-wise less than comparison
 * @param a First tensor
 * @param b Second tensor
 * @return Result tensor (boolean)
 */
Tensor* uop_cmplt(Tensor* a, Tensor* b);

/**
 * @brief Element-wise negation
 * @param a Input tensor
 * @return Result tensor
 */
Tensor* uop_neg(Tensor* a);

/**
 * @brief Element-wise exponential
 * @param a Input tensor
 * @return Result tensor
 */
Tensor* uop_exp(Tensor* a);

/**
 * @brief Element-wise natural logarithm
 * @param a Input tensor
 * @return Result tensor
 */
Tensor* uop_log(Tensor* a);

/**
 * @brief Element-wise square root
 * @param a Input tensor
 * @return Result tensor
 */
Tensor* uop_sqrt(Tensor* a);

/**
 * @brief Element-wise reciprocal
 * @param a Input tensor
 * @return Result tensor
 */
Tensor* uop_recip(Tensor* a);

/**
 * @brief Element-wise absolute value
 * @param a Input tensor
 * @return Result tensor
 */
Tensor* uop_abs(Tensor* a);

/**
 * @brief Element-wise sine
 * @param a Input tensor
 * @return Result tensor
 */
Tensor* uop_sin(Tensor* a);

/**
 * @brief Element-wise cosine
 * @param a Input tensor
 * @return Result tensor
 */
Tensor* uop_cos(Tensor* a);

/**
 * @brief Element-wise tangent
 * @param a Input tensor
 * @return Result tensor
 */
Tensor* uop_tan(Tensor* a);

/**
 * @brief Element-wise power function
 * @param a Base tensor
 * @param b Exponent tensor
 * @return Result tensor
 */
Tensor* uop_pow(Tensor* a, Tensor* b);

typedef struct {
    int* dims;    // Dimensions to reduce (NULL = all)
    int num_dims; // Number of dimensions
    bool keepdim; // Keep reduced dimensions
} ReduceParams;

/**
 * @brief Sum reduction
 * @param a Input tensor
 * @param params Reduction parameters
 * @return Result tensor
 */
Tensor* uop_sum(Tensor* a, ReduceParams* params);

/**
 * @brief Max reduction
 * @param a Input tensor
 * @param params Reduction parameters
 * @return Result tensor
 */
Tensor* uop_max_reduce(Tensor* a, ReduceParams* params);

/**
 * @brief Mean reduction
 * @param a Input tensor
 * @param params Reduction parameters
 * @return Result tensor
 */
Tensor* uop_mean(Tensor* a, ReduceParams* params);

typedef struct {
    int* new_shape;
    int new_ndim;
} ReshapeParams;

typedef struct {
    int* perm;
    int num_dims;
} PermuteParams;

typedef struct {
    int* new_shape;
    int new_ndim;
} ExpandParams;

typedef struct {
    size_t* new_strides;
    int num_dims;
} StrideParams;

typedef struct {
    int* start;
    int* end;
    int* step;
    int num_dims;
} SliceParams;

/**
 * @brief Reshape tensor (view)
 * @param a Input tensor
 * @param params Reshape parameters
 * @return Result tensor
 */
Tensor* uop_reshape(Tensor* a, ReshapeParams* params);

/**
 * @brief Permute tensor dimensions (view)
 * @param a Input tensor
 * @param params Permute parameters
 * @return Result tensor
 */
Tensor* uop_permute(Tensor* a, PermuteParams* params);

/**
 * @brief Expand tensor dimensions (view)
 * @param a Input tensor
 * @param params Expand parameters
 * @return Result tensor
 */
Tensor* uop_expand(Tensor* a, ExpandParams* params);

/**
 * @brief Change tensor strides (view)
 * @param a Input tensor
 * @param params Stride parameters
 * @return Result tensor
 */
Tensor* uop_stride(Tensor* a, StrideParams* params);

/**
 * @brief Slice tensor (view)
 * @param a Input tensor
 * @param params Slice parameters
 * @return Result tensor
 */
Tensor* uop_slice(Tensor* a, SliceParams* params);

/**
 * @brief Matrix multiplication
 * @param a First matrix
 * @param b Second matrix
 * @return Result tensor
 */
Tensor* uop_matmul(Tensor* a, Tensor* b);

typedef struct {
    int* kernel_size;
    int* stride;
    int* padding;
    int* dilation;
    int groups;
    bool bias;
} Conv2DParams;

typedef struct {
    Tensor* cond;
    Tensor* a;
    Tensor* b;
} WhereParams;

typedef struct {
    float value; // Value to fill
    int* shape;  // Output shape
    int ndim;    // Number of dimensions
} FillParams;

typedef struct {
    int dim; // Dimension to gather along (-1 for last dim)
} GatherParams;

/**
 * @brief Fill tensor with constant value (lazy)
 * @param shape Output shape
 * @param ndim Number of dimensions
 * @param value Value to fill
 * @return Result tensor (lazy - data filled on execution)
 */
Tensor* uop_fill(int* shape, int ndim, float value);

/**
 * @brief 2D Convolution
 * @param input Input tensor
 * @param weight Weight tensor
 * @param bias Bias tensor (optional)
 * @param params Convolution parameters
 * @return Result tensor
 */
Tensor* uop_conv2d(Tensor* input, Tensor* weight, Tensor* bias, Conv2DParams* params);

/**
 * @brief Conditional selection
 * @param params Where parameters
 * @return Result tensor
 */
Tensor* uop_where(WhereParams* params);

/**
 * @brief Gather elements by index along a dimension (lazy)
 *
 * For 2D input [N, C] and 1D indices [N]:
 *   out[i] = input[i, indices[i]]
 *
 * This is used for cross-entropy loss to select log probabilities.
 *
 * @param input Input tensor [N, C] (e.g., log_softmax output)
 * @param indices Index tensor [N] with values in [0, C-1]
 * @param dim Dimension to gather along (-1 for last dim)
 * @return Result tensor [N] - gathered elements (lazy)
 */
Tensor* uop_gather(Tensor* input, Tensor* indices, int dim);

/**
 * @brief ReLU activation
 * @param x Input tensor
 * @return Result tensor
 */
Tensor* uop_relu(Tensor* x);

/**
 * @brief Sigmoid activation
 * @param x Input tensor
 * @return Result tensor
 */
Tensor* uop_sigmoid(Tensor* x);

/**
 * @brief Tanh activation
 * @param x Input tensor
 * @return Result tensor
 */
Tensor* uop_tanh(Tensor* x);

/**
 * @brief GELU activation
 * @param x Input tensor
 * @return Result tensor
 */
Tensor* uop_gelu(Tensor* x);

/**
 * @brief Softmax activation
 * @param x Input tensor
 * @param dim Dimension to apply softmax
 * @return Result tensor
 */
Tensor* uop_softmax(Tensor* x, int dim);

/**
 * @brief Leaky ReLU built from uops: max(x, alpha * x) = x if x > 0, else alpha * x
 */
Tensor* uop_leaky_relu(Tensor* x, float negative_slope);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_UOPS_H
