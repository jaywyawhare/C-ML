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

    // Additional Unary Ops (tinygrad parity)
    UOP_LOG10,       // log10(a)
    UOP_SINH,        // sinh(a)
    UOP_COSH,        // cosh(a)
    UOP_ASINH,       // asinh(a)
    UOP_ACOSH,       // acosh(a)
    UOP_ATANH,       // atanh(a)
    UOP_TRUNC,       // truncate to integer
    UOP_ISINF,       // test for infinity (returns 0 or 1)
    UOP_ISNAN,       // test for NaN (returns 0 or 1)
    UOP_ISFINITE,    // test for finite (returns 0 or 1)
    UOP_LOGICAL_NOT, // logical NOT (returns 0 or 1)

    // Additional Binary Ops (tinygrad parity)
    UOP_IDIV,        // integer division: floor(a / b)
    UOP_MOD,         // modulo: a % b (fmodf)
    UOP_MINIMUM,     // min(a, b)
    UOP_COPYSIGN,    // copysign(a, b)
    UOP_LOGADDEXP,   // log(exp(a) + exp(b))
    UOP_LSHIFT,      // a << b (left bit shift)
    UOP_RSHIFT,      // a >> b (right bit shift)
    UOP_LOGICAL_AND, // a && b (returns 0 or 1)
    UOP_LOGICAL_OR,  // a || b (returns 0 or 1)

    // Comparison Ops (tinygrad parity)
    UOP_CMPEQ,       // a == b (returns 0 or 1)
    UOP_CMPNE,       // a != b (returns 0 or 1)
    UOP_CMPLE,       // a <= b (returns 0 or 1)
    UOP_CMPGT,       // a > b (returns 0 or 1)
    UOP_CMPGE,       // a >= b (returns 0 or 1)

    // Additional Reduction Ops (tinygrad parity)
    UOP_MIN_REDUCE,  // min along dimension(s)
    UOP_VAR,         // variance along dimension(s)
    UOP_STD,         // standard deviation along dimension(s)
    UOP_ANY,         // logical OR reduce along dimension(s)
    UOP_ALL,         // logical AND reduce along dimension(s)
    UOP_LOGSUMEXP,   // log(sum(exp(a))) along dimension(s)
    UOP_CUMMAX,      // cumulative max along dimension
    UOP_CUMMIN,      // cumulative min along dimension

    // Movement/Shape Ops (tinygrad parity)
    UOP_CAT,         // concatenate tensors along dimension
    UOP_STACK,       // stack tensors along new dimension
    UOP_SCATTER,     // scatter elements by index
    UOP_ROLL,        // circular shift along dimension
    UOP_FLATTEN,     // collapse dimensions into one
    UOP_UNFLATTEN,   // expand a dimension into multiple
    UOP_DIAG,        // create/extract diagonal
    UOP_ONE_HOT,     // one-hot encoding

    // Additional Unary Ops (tinygrad parity round 2)
    UOP_ERFC,            // complementary error function erfc(a)
    UOP_LOGCUMSUMEXP,    // log(cumsum(exp(a))) along dimension

    // Additional Binary Ops (tinygrad parity round 2)
    UOP_LERP,            // linear interpolation: a + t*(b-a)

    // Additional Movement/Shape Ops (tinygrad parity round 2)
    UOP_TILE,            // repeat tensor along dimensions
    UOP_REPEAT_INTERLEAVE, // repeat elements along dimension
    UOP_TRACE,           // sum of diagonal elements
    UOP_SHRINK,          // shrink tensor (slice with start/end per dim)

    // Activation Ops (tinygrad parity)
    UOP_RELU6,           // min(max(x, 0), 6)
    UOP_HARD_SIGMOID,    // clamp((x + 3) / 6, 0, 1)
    UOP_HARD_TANH,       // clamp(x, -1, 1)
    UOP_CELU,            // max(0,x) + min(0, alpha*(exp(x/alpha)-1))
    UOP_QUICK_GELU,      // x * sigmoid(1.702 * x)
    UOP_SOFTPLUS,        // log(1 + exp(x))
    UOP_SOFTSIGN,        // x / (1 + |x|)
    UOP_LOGSIGMOID,      // log(sigmoid(x))

    // Additional Ops (tinygrad parity round 3)
    UOP_UNFOLD,          // sliding window extraction (im2col-like)

    // Additional Activation Ops (tinygrad parity round 4)
    UOP_ELU,             // x > 0 ? x : alpha*(exp(x)-1)
    UOP_SELU,            // scale*(x > 0 ? x : alpha*(exp(x)-1))
    UOP_MISH,            // x * tanh(softplus(x))
    UOP_SILU,            // x * sigmoid(x) (swish)
    UOP_HARDSWISH,       // x > 3 ? x : x < -3 ? 0 : x*(x+3)/6

    // Masking/Selection Ops (tinygrad parity round 4)
    UOP_MASKED_SELECT,   // select elements where mask is true

    // Movement/Shape Ops (tinygrad parity round 4)
    UOP_SPLIT,           // split tensor into chunks along dim
    UOP_CHUNK,           // split tensor into N chunks along dim
    UOP_MESHGRID,        // create coordinate matrices from vectors
    UOP_DIAGONAL,        // extract diagonal with offset, dim1, dim2

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
    bool use_winograd;  /* set by uop_conv2d when Winograd is applicable */
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

/**
 * @brief Padding mode
 */
typedef enum {
    PAD_CONSTANT = 0,  // Fill with constant value
    PAD_REFLECT,       // Reflect at boundary (e.g. [a,b,c,d] -> [c,b,a,b,c,d,c,b])
    PAD_REPLICATE      // Replicate edge values (e.g. [a,b,c,d] -> [a,a,a,b,c,d,d,d])
} PadMode;

typedef struct {
    int* pad_widths; // [before_0, after_0, before_1, after_1, ...]
    int num_dims;
    float value; // Pad value (usually 0)
    PadMode mode; // Padding mode (default: PAD_CONSTANT)
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

/** @brief Product reduction along dimension(s) */
Tensor* uop_prod(Tensor* a, ReduceParams* params);

/** @brief Argmax along a dimension */
Tensor* uop_argmax(Tensor* a, ReduceParams* params);

/** @brief Argmin along a dimension */
Tensor* uop_argmin(Tensor* a, ReduceParams* params);

/** @brief Cumulative sum along a dimension */
Tensor* uop_cumsum(Tensor* a, int dim);

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

/** @brief Element-wise log base 10 */
Tensor* uop_log10(Tensor* a);

/** @brief Element-wise hyperbolic sine */
Tensor* uop_sinh(Tensor* a);

/** @brief Element-wise hyperbolic cosine */
Tensor* uop_cosh(Tensor* a);

/** @brief Element-wise inverse hyperbolic sine */
Tensor* uop_asinh(Tensor* a);

/** @brief Element-wise inverse hyperbolic cosine */
Tensor* uop_acosh(Tensor* a);

/** @brief Element-wise inverse hyperbolic tangent */
Tensor* uop_atanh(Tensor* a);

/** @brief Element-wise truncate to integer */
Tensor* uop_trunc(Tensor* a);

/** @brief Element-wise test for infinity (returns 0 or 1) */
Tensor* uop_isinf(Tensor* a);

/** @brief Element-wise test for NaN (returns 0 or 1) */
Tensor* uop_isnan(Tensor* a);

/** @brief Element-wise test for finite (returns 0 or 1) */
Tensor* uop_isfinite(Tensor* a);

/** @brief Element-wise logical NOT (returns 0 or 1) */
Tensor* uop_logical_not(Tensor* a);

/** @brief Integer division: floor(a / b) */
Tensor* uop_idiv(Tensor* a, Tensor* b);

/** @brief Modulo: fmod(a, b) */
Tensor* uop_mod(Tensor* a, Tensor* b);

/** @brief Element-wise minimum */
Tensor* uop_minimum(Tensor* a, Tensor* b);

/** @brief Copy sign of b to magnitude of a */
Tensor* uop_copysign(Tensor* a, Tensor* b);

/** @brief log(exp(a) + exp(b)) numerically stable */
Tensor* uop_logaddexp(Tensor* a, Tensor* b);

/** @brief Left bit shift */
Tensor* uop_lshift(Tensor* a, Tensor* b);

/** @brief Right bit shift */
Tensor* uop_rshift(Tensor* a, Tensor* b);

/** @brief Logical AND */
Tensor* uop_logical_and(Tensor* a, Tensor* b);

/** @brief Logical OR */
Tensor* uop_logical_or(Tensor* a, Tensor* b);

/** @brief Element-wise equal */
Tensor* uop_cmpeq(Tensor* a, Tensor* b);

/** @brief Element-wise not equal */
Tensor* uop_cmpne(Tensor* a, Tensor* b);

/** @brief Element-wise less than or equal */
Tensor* uop_cmple(Tensor* a, Tensor* b);

/** @brief Element-wise greater than */
Tensor* uop_cmpgt(Tensor* a, Tensor* b);

/** @brief Element-wise greater than or equal */
Tensor* uop_cmpge(Tensor* a, Tensor* b);

/** @brief Min reduction along dimension(s) */
Tensor* uop_min_reduce(Tensor* a, ReduceParams* params);

/** @brief Variance along dimension(s) */
Tensor* uop_var(Tensor* a, ReduceParams* params);

/** @brief Standard deviation along dimension(s) */
Tensor* uop_std(Tensor* a, ReduceParams* params);

/** @brief Logical OR reduce (any non-zero) along dimension(s) */
Tensor* uop_any(Tensor* a, ReduceParams* params);

/** @brief Logical AND reduce (all non-zero) along dimension(s) */
Tensor* uop_all(Tensor* a, ReduceParams* params);

/** @brief log(sum(exp(a))) along dimension(s) */
Tensor* uop_logsumexp(Tensor* a, ReduceParams* params);

/** @brief Cumulative max along dimension */
Tensor* uop_cummax(Tensor* a, int dim);

/** @brief Cumulative min along dimension */
Tensor* uop_cummin(Tensor* a, int dim);

typedef struct {
    int dim;         // Dimension to concatenate along
    int num_tensors; // Number of input tensors
} CatParams;

typedef struct {
    int dim;         // Dimension to stack along
    int num_tensors; // Number of input tensors
} StackParams;

typedef struct {
    int dim;   // Dimension to scatter along
} ScatterParams;

typedef struct {
    int shift; // Number of positions to roll
    int dim;   // Dimension to roll along
} RollParams;

typedef struct {
    int start_dim; // First dimension to flatten
    int end_dim;   // Last dimension to flatten
} FlattenParams;

typedef struct {
    int dim;       // Dimension to unflatten
    int* sizes;    // New sizes for the unflattened dimension
    int num_sizes; // Number of new sizes
} UnflattenParams;

typedef struct {
    int offset;    // Diagonal offset (0=main, positive=above, negative=below)
} DiagParams;

typedef struct {
    int num_classes; // Number of classes (-1 = auto from max value)
} OneHotParams;

/** @brief Concatenate tensors along dimension */
Tensor* uop_cat(Tensor** tensors, int num_tensors, int dim);

/** @brief Stack tensors along new dimension */
Tensor* uop_stack(Tensor** tensors, int num_tensors, int dim);

/** @brief Scatter elements by index along dimension */
Tensor* uop_scatter(Tensor* a, int dim, Tensor* index, Tensor* src);

/** @brief Circular shift along dimension */
Tensor* uop_roll(Tensor* a, int shift, int dim);

/** @brief Flatten dimensions [start_dim, end_dim] into one */
Tensor* uop_flatten(Tensor* a, int start_dim, int end_dim);

/** @brief Unflatten a dimension into multiple dimensions */
Tensor* uop_unflatten(Tensor* a, int dim, int* sizes, int num_sizes);

/** @brief Create/extract diagonal. For 1D: creates diagonal matrix. For 2D: extracts diagonal. */
Tensor* uop_diag(Tensor* a, int offset);

/** @brief One-hot encoding. Input is integer tensor, output is float. */
Tensor* uop_one_hot(Tensor* a, int num_classes);

/** @brief Scaled dot product attention: softmax(Q*K^T / sqrt(d)) * V */
Tensor* uop_scaled_dot_product_attention(Tensor* q, Tensor* k, Tensor* v, Tensor* mask);

typedef struct {
    int* repeats;   // Number of repeats per dimension
    int num_dims;   // Number of dimensions
} TileParams;

typedef struct {
    int repeats;    // Number of times to repeat each element
    int dim;        // Dimension along which to repeat
} RepeatInterleaveParams;

typedef struct {
    int* starts;    // Start index per dimension
    int* ends;      // End index per dimension
    int num_dims;
} ShrinkParams;

/** @brief Complementary error function: erfc(a) = 1 - erf(a) */
Tensor* uop_erfc(Tensor* a);

/** @brief Linear interpolation: a + t*(b-a) */
Tensor* uop_lerp(Tensor* a, Tensor* b, Tensor* t);

/** @brief Tile (repeat) tensor along dimensions */
Tensor* uop_tile(Tensor* a, int* repeats, int num_dims);

/** @brief Repeat elements along dimension */
Tensor* uop_repeat_interleave(Tensor* a, int repeats, int dim);

/** @brief Sum of diagonal elements (trace) */
Tensor* uop_trace(Tensor* a);

/** @brief Shrink tensor dimensions (slice with start/end per dim) */
Tensor* uop_shrink(Tensor* a, int* starts, int* ends, int num_dims);

/** @brief log(cumsum(exp(a))) along dimension */
Tensor* uop_logcumsumexp(Tensor* a, int dim);

/** @brief ReLU6: min(max(x, 0), 6) */
Tensor* uop_relu6(Tensor* x);

/** @brief Hard sigmoid: clamp((x + 3) / 6, 0, 1) */
Tensor* uop_hard_sigmoid(Tensor* x);

/** @brief Hard tanh: clamp(x, -1, 1) */
Tensor* uop_hard_tanh(Tensor* x);

/** @brief CELU: max(0,x) + min(0, alpha*(exp(x/alpha)-1)) */
Tensor* uop_celu(Tensor* x, float alpha);

/** @brief Quick GELU: x * sigmoid(1.702 * x) */
Tensor* uop_quick_gelu(Tensor* x);

/** @brief Softplus: log(1 + exp(x)) */
Tensor* uop_softplus(Tensor* x);

/** @brief Softsign: x / (1 + |x|) */
Tensor* uop_softsign(Tensor* x);

/** @brief Log sigmoid: log(sigmoid(x)) = -softplus(-x) */
Tensor* uop_logsigmoid(Tensor* x);

typedef struct {
    int kernel_size;  // Size of the sliding window
    int stride;       // Stride of the sliding window (default = 1)
} UnfoldParams;

/** @brief Unfold (sliding window) along last dimension */
Tensor* uop_unfold(Tensor* a, int kernel_size, int stride);

/** @brief Variance and mean together: returns (var, mean) */
void uop_var_mean(Tensor* a, ReduceParams* params, Tensor** out_var, Tensor** out_mean);

/** @brief Standard deviation and mean together: returns (std, mean) */
void uop_std_mean(Tensor* a, ReduceParams* params, Tensor** out_std, Tensor** out_mean);

/** @brief ELU: x > 0 ? x : alpha*(exp(x)-1) */
Tensor* uop_elu(Tensor* x, float alpha);

/** @brief SELU: scale * elu(x, alpha), scale=1.0507, alpha=1.6733 */
Tensor* uop_selu(Tensor* x);

/** @brief Mish: x * tanh(softplus(x)) = x * tanh(ln(1+exp(x))) */
Tensor* uop_mish(Tensor* x);

/** @brief SiLU (Swish): x * sigmoid(x) */
Tensor* uop_silu(Tensor* x);

/** @brief HardSwish: x > 3 ? x : x < -3 ? 0 : x*(x+3)/6 */
Tensor* uop_hardswish(Tensor* x);

/** @brief Select elements where mask is true (returns 1D tensor) */
Tensor* uop_masked_select(Tensor* a, Tensor* mask);

/** @brief Split tensor into chunks of split_size along dim */
Tensor** uop_split(Tensor* a, int split_size, int dim, int* num_splits);

/** @brief Split tensor into N roughly equal chunks along dim */
Tensor** uop_chunk(Tensor* a, int chunks, int dim, int* num_chunks);

/** @brief Create coordinate matrices from 1D vectors */
Tensor** uop_meshgrid(Tensor** tensors, int num_tensors, int* num_outputs);

/** @brief Extract diagonal with offset from dimensions dim1, dim2 */
Tensor* uop_diagonal(Tensor* a, int offset, int dim1, int dim2);

/** @brief Pad tensor with reflect mode */
Tensor* uop_pad_reflect(Tensor* a, int* pad_widths, int num_dims);

/** @brief Pad tensor with replicate mode */
Tensor* uop_pad_replicate(Tensor* a, int* pad_widths, int num_dims);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_UOPS_H
