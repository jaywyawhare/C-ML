/*
 * Micro-Operations (UOps) - Minimal set of fundamental operations
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

typedef struct {
    UOpType type;
    Tensor** inputs;
    int num_inputs;
    Tensor* output;
    void* params; // Operation-specific parameters
} UOp;

int uop_execute(UOp* uop);
Tensor* uop_create_and_execute(UOpType type, Tensor** inputs, int num_inputs, void* params);

Tensor* uop_add(Tensor* a, Tensor* b);
Tensor* uop_sub(Tensor* a, Tensor* b);
Tensor* uop_mul(Tensor* a, Tensor* b);
Tensor* uop_div(Tensor* a, Tensor* b);
Tensor* uop_max(Tensor* a, Tensor* b);
Tensor* uop_cmplt(Tensor* a, Tensor* b);
Tensor* uop_neg(Tensor* a);
Tensor* uop_exp(Tensor* a);
Tensor* uop_log(Tensor* a);
Tensor* uop_sqrt(Tensor* a);
Tensor* uop_recip(Tensor* a);
Tensor* uop_abs(Tensor* a);
Tensor* uop_sin(Tensor* a);
Tensor* uop_cos(Tensor* a);
Tensor* uop_tan(Tensor* a);
Tensor* uop_pow(Tensor* a, Tensor* b);

typedef struct {
    int* dims;    // Dimensions to reduce (NULL = all)
    int num_dims; // Number of dimensions
    bool keepdim; // Keep reduced dimensions
} ReduceParams;

Tensor* uop_sum(Tensor* a, ReduceParams* params);
Tensor* uop_max_reduce(Tensor* a, ReduceParams* params);
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

Tensor* uop_reshape(Tensor* a, ReshapeParams* params);
Tensor* uop_permute(Tensor* a, PermuteParams* params);
Tensor* uop_expand(Tensor* a, ExpandParams* params);
Tensor* uop_stride(Tensor* a, StrideParams* params);
Tensor* uop_slice(Tensor* a, SliceParams* params);
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

/* Lazy - data filled on execution */
Tensor* uop_fill(int* shape, int ndim, float value);
Tensor* uop_conv2d(Tensor* input, Tensor* weight, Tensor* bias, Conv2DParams* params);
Tensor* uop_where(WhereParams* params);

/*
 * For 2D input [N, C] and 1D indices [N]:
 *   out[i] = input[i, indices[i]]
 * Used for cross-entropy loss to select log probabilities.
 */
Tensor* uop_gather(Tensor* input, Tensor* indices, int dim);

Tensor* uop_relu(Tensor* x);
Tensor* uop_sigmoid(Tensor* x);
Tensor* uop_tanh(Tensor* x);
Tensor* uop_gelu(Tensor* x);
Tensor* uop_softmax(Tensor* x, int dim);

/* max(x, alpha * x) = x if x > 0, else alpha * x */
Tensor* uop_leaky_relu(Tensor* x, float negative_slope);

Tensor* uop_sign(Tensor* a);
Tensor* uop_floor(Tensor* a);
Tensor* uop_ceil(Tensor* a);
Tensor* uop_round(Tensor* a);
Tensor* uop_log2(Tensor* a);
Tensor* uop_exp2(Tensor* a);
Tensor* uop_asin(Tensor* a);
Tensor* uop_acos(Tensor* a);
Tensor* uop_atan(Tensor* a);
Tensor* uop_square(Tensor* a);
Tensor* uop_rsqrt(Tensor* a);
Tensor* uop_erf(Tensor* a);
Tensor* uop_clamp(Tensor* a, float min_val, float max_val);
Tensor* uop_prod(Tensor* a, ReduceParams* params);
Tensor* uop_argmax(Tensor* a, ReduceParams* params);
Tensor* uop_argmin(Tensor* a, ReduceParams* params);
Tensor* uop_cumsum(Tensor* a, int dim);
Tensor* uop_triu(Tensor* a, int diagonal);
Tensor* uop_tril(Tensor* a, int diagonal);
Tensor* uop_pad(Tensor* a, int* pad_widths, int num_dims, float value);
Tensor* uop_sort(Tensor* a, int dim, bool descending);
Tensor* uop_argsort(Tensor* a, int dim, bool descending);
Tensor* uop_topk(Tensor* a, int k, int dim, bool largest, Tensor** indices_out);
Tensor* uop_cumprod(Tensor* a, int dim);
Tensor* uop_bitwise_and(Tensor* a, Tensor* b);
Tensor* uop_bitwise_or(Tensor* a, Tensor* b);
Tensor* uop_bitwise_xor(Tensor* a, Tensor* b);
Tensor* uop_bitwise_not(Tensor* a);

/* Returns indices of non-zero elements as [N, ndim] tensor */
Tensor* uop_nonzero(Tensor* a);
Tensor* uop_masked_fill(Tensor* a, Tensor* mask, float value);
Tensor* uop_log10(Tensor* a);
Tensor* uop_sinh(Tensor* a);
Tensor* uop_cosh(Tensor* a);
Tensor* uop_asinh(Tensor* a);
Tensor* uop_acosh(Tensor* a);
Tensor* uop_atanh(Tensor* a);
Tensor* uop_trunc(Tensor* a);
Tensor* uop_isinf(Tensor* a);
Tensor* uop_isnan(Tensor* a);
Tensor* uop_isfinite(Tensor* a);
Tensor* uop_logical_not(Tensor* a);
Tensor* uop_idiv(Tensor* a, Tensor* b);
Tensor* uop_mod(Tensor* a, Tensor* b);
Tensor* uop_minimum(Tensor* a, Tensor* b);
Tensor* uop_copysign(Tensor* a, Tensor* b);

/* log(exp(a) + exp(b)) numerically stable */
Tensor* uop_logaddexp(Tensor* a, Tensor* b);
Tensor* uop_lshift(Tensor* a, Tensor* b);
Tensor* uop_rshift(Tensor* a, Tensor* b);
Tensor* uop_logical_and(Tensor* a, Tensor* b);
Tensor* uop_logical_or(Tensor* a, Tensor* b);
Tensor* uop_cmpeq(Tensor* a, Tensor* b);
Tensor* uop_cmpne(Tensor* a, Tensor* b);
Tensor* uop_cmple(Tensor* a, Tensor* b);
Tensor* uop_cmpgt(Tensor* a, Tensor* b);
Tensor* uop_cmpge(Tensor* a, Tensor* b);
Tensor* uop_min_reduce(Tensor* a, ReduceParams* params);
Tensor* uop_var(Tensor* a, ReduceParams* params);
Tensor* uop_std(Tensor* a, ReduceParams* params);
Tensor* uop_any(Tensor* a, ReduceParams* params);
Tensor* uop_all(Tensor* a, ReduceParams* params);

/* log(sum(exp(a))) along dimension(s) */
Tensor* uop_logsumexp(Tensor* a, ReduceParams* params);
Tensor* uop_cummax(Tensor* a, int dim);
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

Tensor* uop_cat(Tensor** tensors, int num_tensors, int dim);
Tensor* uop_stack(Tensor** tensors, int num_tensors, int dim);
Tensor* uop_scatter(Tensor* a, int dim, Tensor* index, Tensor* src);
Tensor* uop_roll(Tensor* a, int shift, int dim);
Tensor* uop_flatten(Tensor* a, int start_dim, int end_dim);
Tensor* uop_unflatten(Tensor* a, int dim, int* sizes, int num_sizes);

/* For 1D: creates diagonal matrix. For 2D: extracts diagonal. */
Tensor* uop_diag(Tensor* a, int offset);
Tensor* uop_one_hot(Tensor* a, int num_classes);

/* softmax(Q*K^T / sqrt(d)) * V */
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

/* erfc(a) = 1 - erf(a) */
Tensor* uop_erfc(Tensor* a);

/* a + t*(b-a) */
Tensor* uop_lerp(Tensor* a, Tensor* b, Tensor* t);
Tensor* uop_tile(Tensor* a, int* repeats, int num_dims);
Tensor* uop_repeat_interleave(Tensor* a, int repeats, int dim);
Tensor* uop_trace(Tensor* a);
Tensor* uop_shrink(Tensor* a, int* starts, int* ends, int num_dims);

/* log(cumsum(exp(a))) along dimension */
Tensor* uop_logcumsumexp(Tensor* a, int dim);

/* min(max(x, 0), 6) */
Tensor* uop_relu6(Tensor* x);

/* clamp((x + 3) / 6, 0, 1) */
Tensor* uop_hard_sigmoid(Tensor* x);

/* clamp(x, -1, 1) */
Tensor* uop_hard_tanh(Tensor* x);

/* max(0,x) + min(0, alpha*(exp(x/alpha)-1)) */
Tensor* uop_celu(Tensor* x, float alpha);

/* x * sigmoid(1.702 * x) */
Tensor* uop_quick_gelu(Tensor* x);

/* log(1 + exp(x)) */
Tensor* uop_softplus(Tensor* x);

/* x / (1 + |x|) */
Tensor* uop_softsign(Tensor* x);

/* log(sigmoid(x)) = -softplus(-x) */
Tensor* uop_logsigmoid(Tensor* x);

typedef struct {
    int kernel_size;  // Size of the sliding window
    int stride;       // Stride of the sliding window (default = 1)
} UnfoldParams;

Tensor* uop_unfold(Tensor* a, int kernel_size, int stride);
void uop_var_mean(Tensor* a, ReduceParams* params, Tensor** out_var, Tensor** out_mean);
void uop_std_mean(Tensor* a, ReduceParams* params, Tensor** out_std, Tensor** out_mean);

/* x > 0 ? x : alpha*(exp(x)-1) */
Tensor* uop_elu(Tensor* x, float alpha);

/* scale * elu(x, alpha), scale=1.0507, alpha=1.6733 */
Tensor* uop_selu(Tensor* x);

/* x * tanh(softplus(x)) = x * tanh(ln(1+exp(x))) */
Tensor* uop_mish(Tensor* x);

/* x * sigmoid(x) */
Tensor* uop_silu(Tensor* x);

/* x > 3 ? x : x < -3 ? 0 : x*(x+3)/6 */
Tensor* uop_hardswish(Tensor* x);

/* Returns 1D tensor of elements where mask is true */
Tensor* uop_masked_select(Tensor* a, Tensor* mask);
Tensor** uop_split(Tensor* a, int split_size, int dim, int* num_splits);
Tensor** uop_chunk(Tensor* a, int chunks, int dim, int* num_chunks);
Tensor** uop_meshgrid(Tensor** tensors, int num_tensors, int* num_outputs);
Tensor* uop_diagonal(Tensor* a, int offset, int dim1, int dim2);
Tensor* uop_pad_reflect(Tensor* a, int* pad_widths, int num_dims);
Tensor* uop_pad_replicate(Tensor* a, int* pad_widths, int num_dims);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_UOPS_H
