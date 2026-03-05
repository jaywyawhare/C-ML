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

    // Additional Unary Ops
    UOP_SIGN,    // sign(a): -1, 0, or 1
    UOP_FLOOR,   // floor(a)
    UOP_CEIL,    // ceil(a)
    UOP_ROUND,   // round(a)
    UOP_LOG2,    // log2(a)
    UOP_EXP2,    // 2^a
    UOP_ASIN,    // asin(a)
    UOP_ACOS,    // acos(a)
    UOP_ATAN,    // atan(a)
    UOP_SQUARE,  // a * a
    UOP_RSQRT,   // 1 / sqrt(a)
    UOP_ERF,     // erf(a)
    UOP_CLAMP,   // clamp(a, min, max)

    // Additional Reduction Ops
    UOP_PROD,       // product along dimension(s)
    UOP_ARGMAX,     // argmax along dimension
    UOP_ARGMIN,     // argmin along dimension
    UOP_CUMSUM,     // cumulative sum along dimension

    // Special Ops
    UOP_MATMUL, // matrix multiplication
    UOP_CONV2D, // 2D convolution
    UOP_WHERE,  // conditional: where(cond, a, b)
    UOP_FILL,   // fill tensor with constant value
    UOP_GATHER, // gather elements by index: out[i] = input[i, indices[i]]
    UOP_TRIU,   // upper triangular
    UOP_TRIL,   // lower triangular
    UOP_PAD,    // pad tensor

    // Sorting Ops
    UOP_SORT,       // sort along dimension
    UOP_ARGSORT,    // indices that would sort
    UOP_TOPK,       // top-k elements along dimension

    // Cumulative Ops
    UOP_CUMPROD,    // cumulative product along dimension

    // Bitwise Ops
    UOP_BITWISE_AND,
    UOP_BITWISE_OR,
    UOP_BITWISE_XOR,
    UOP_BITWISE_NOT,

    // Masking Ops
    UOP_NONZERO,     // indices of non-zero elements
    UOP_MASKED_FILL, // fill where mask is true

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

typedef struct {
    float min_val;
    float max_val;
} ClampParams;

typedef struct {
    int dim; // Dimension for cumulative op
} CumsumParams;

typedef struct {
    int diagonal; // Offset from main diagonal (0 = main, positive = above, negative = below)
} TriParams;

typedef struct {
    int* pad_widths; // [before_0, after_0, before_1, after_1, ...]
    int num_dims;
    float value; // Pad value (usually 0)
} PadParams;

typedef struct {
    int dim;
    bool descending;
} SortParams;

typedef struct {
    int k;
    int dim;
    bool largest;
} TopkParams;

typedef struct {
    float value;
} MaskedFillParams;

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

// ===== New Unary Operations =====

/** @brief Element-wise sign: -1, 0, or 1 */
Tensor* uop_sign(Tensor* a);

/** @brief Element-wise floor */
Tensor* uop_floor(Tensor* a);

/** @brief Element-wise ceil */
Tensor* uop_ceil(Tensor* a);

/** @brief Element-wise round */
Tensor* uop_round(Tensor* a);

/** @brief Element-wise log base 2 */
Tensor* uop_log2(Tensor* a);

/** @brief Element-wise 2^a */
Tensor* uop_exp2(Tensor* a);

/** @brief Element-wise arcsine */
Tensor* uop_asin(Tensor* a);

/** @brief Element-wise arccosine */
Tensor* uop_acos(Tensor* a);

/** @brief Element-wise arctangent */
Tensor* uop_atan(Tensor* a);

/** @brief Element-wise square: a*a */
Tensor* uop_square(Tensor* a);

/** @brief Element-wise inverse square root: 1/sqrt(a) */
Tensor* uop_rsqrt(Tensor* a);

/** @brief Element-wise error function */
Tensor* uop_erf(Tensor* a);

/** @brief Element-wise clamp to [min, max] */
Tensor* uop_clamp(Tensor* a, float min_val, float max_val);

// ===== New Reduction Operations =====

/** @brief Product reduction along dimension(s) */
Tensor* uop_prod(Tensor* a, ReduceParams* params);

/** @brief Argmax along a dimension */
Tensor* uop_argmax(Tensor* a, ReduceParams* params);

/** @brief Argmin along a dimension */
Tensor* uop_argmin(Tensor* a, ReduceParams* params);

/** @brief Cumulative sum along a dimension */
Tensor* uop_cumsum(Tensor* a, int dim);

// ===== New Special Operations =====

/** @brief Upper triangular matrix */
Tensor* uop_triu(Tensor* a, int diagonal);

/** @brief Lower triangular matrix */
Tensor* uop_tril(Tensor* a, int diagonal);

/** @brief Pad tensor with constant value */
Tensor* uop_pad(Tensor* a, int* pad_widths, int num_dims, float value);

/** @brief Sort along dimension. descending=true for largest first */
Tensor* uop_sort(Tensor* a, int dim, bool descending);

/** @brief Return indices that would sort along dimension */
Tensor* uop_argsort(Tensor* a, int dim, bool descending);

/** @brief Return top-k elements along dimension */
Tensor* uop_topk(Tensor* a, int k, int dim, bool largest, Tensor** indices_out);

/** @brief Cumulative product along dimension */
Tensor* uop_cumprod(Tensor* a, int dim);

/** @brief Bitwise AND (operates on int cast of float) */
Tensor* uop_bitwise_and(Tensor* a, Tensor* b);

/** @brief Bitwise OR */
Tensor* uop_bitwise_or(Tensor* a, Tensor* b);

/** @brief Bitwise XOR */
Tensor* uop_bitwise_xor(Tensor* a, Tensor* b);

/** @brief Bitwise NOT */
Tensor* uop_bitwise_not(Tensor* a);

/** @brief Return indices of non-zero elements as [N, ndim] tensor */
Tensor* uop_nonzero(Tensor* a);

/** @brief Fill tensor with value where mask is true */
Tensor* uop_masked_fill(Tensor* a, Tensor* mask, float value);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_UOPS_H
