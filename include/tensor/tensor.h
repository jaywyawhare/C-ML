#ifndef CML_TENSOR_H
#define CML_TENSOR_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include "backend/device.h"

#ifdef __cplusplus
extern "C" {
#endif

struct CMLBackendBuffer;
typedef struct CMLBackendBuffer* CMLBackendBuffer_t;

typedef enum {
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
    DTYPE_INT32,
    DTYPE_INT64,
    DTYPE_BOOL,
    DTYPE_FLOAT16,
    DTYPE_BFLOAT16,
    DTYPE_INT8,
    DTYPE_UINT8,
    DTYPE_INT16,
    DTYPE_UINT16,
    DTYPE_UINT32,
    DTYPE_UINT64,
    DTYPE_FLOAT8_E4M3,       // 1 sign, 4 exponent, 3 mantissa (range [-448, 448])
    DTYPE_FLOAT8_E5M2,       // 1 sign, 5 exponent, 2 mantissa (range [-57344, 57344])
    DTYPE_FLOAT8_E4M3_FNUZ,  // FNUZ: no negative zero, no inf, bias=8 (AMD MI300)
    DTYPE_FLOAT8_E5M2_FNUZ,  // FNUZ: no negative zero, no inf, bias=16 (AMD MI300)
} DType;

struct IRNode;
struct CMLGraph;
typedef struct CMLGraph* CMLGraph_t;

typedef struct Tensor {
    // Shape info (computed from IR, not execution)
    int* shape;        // Shape array
    int ndim;          // Number of dimensions
    size_t numel;      // Total number of elements
    DType dtype;       // Data type
    DeviceType device; // Device

    // IR node reference (THE KEY!)
    struct IRNode* ir_node; // Points to IR node in graph
    CMLGraph_t ir_context;  // Which IR graph this belongs to

    // Execution state (lazy)
    bool is_executed; // Has this been executed?
    void* data;       // NULL until executed (lazy!)
    bool owns_data;        // Does this tensor own its data?
    bool from_buffer_cache; // Data was allocated via cml_buffer_cache_alloc

    bool requires_grad;
    struct Tensor* grad; // Gradient tensor (also lazy!)

    int ref_count;       // Reference counting
    struct Tensor* base; // Base tensor (if this is a view)

    size_t* strides;       // Stride array (for efficient views)
    size_t storage_offset; // Offset into data (for views/slices)
    bool is_contiguous;    // Is memory layout contiguous?
    CMLBackendBuffer_t buffer_handle;

    void* user_data;
} Tensor;

size_t cml_dtype_size(DType dtype);
size_t tensor_numel(int* shape, int ndim);
DType cml_promote_dtype(DType dtype1, DType dtype2);

/* Returns allocated strides array (caller must free) */
size_t* compute_contiguous_strides(int* shape, int ndim);

bool tensor_check_is_contiguous(int* shape, size_t* strides, int ndim);
size_t tensor_compute_storage_size(int* shape, size_t* strides, int ndim);

/* Defaults: dtype=DTYPE_FLOAT32, device=DEVICE_AUTO (auto-detected) */
typedef struct TensorConfig {
    DType dtype;       // Data type (use -1 or DTYPE_FLOAT32 for default)
    DeviceType device; // Device (use DEVICE_AUTO for auto-detection)
    bool has_dtype;    // Set to true if dtype is explicitly set
    bool has_device;   // Set to true if device is explicitly set
} TensorConfig;

Tensor* tensor_empty(int* shape, int ndim, const TensorConfig* config);
Tensor* tensor_zeros(int* shape, int ndim, const TensorConfig* config);
Tensor* tensor_ones(int* shape, int ndim, const TensorConfig* config);
Tensor* tensor_full(int* shape, int ndim, const TensorConfig* config, float value);
Tensor* tensor_from_data(const void* data, int* shape, int ndim, const TensorConfig* config);

#include "tensor/tensor_views.h"

void tensor_free(Tensor* t);
Tensor* tensor_clone(Tensor* t);
float tensor_get_float(Tensor* t, size_t idx);
void tensor_set_float(Tensor* t, size_t idx, float value);
void* tensor_data_ptr(Tensor* t); /* Triggers lazy execution if needed */
size_t tensor_compute_offset(Tensor* t, int* indices);
int tensor_ensure_executed(Tensor* t);
CMLGraph_t tensor_get_ir_context(Tensor* t);
bool tensor_is_scalar(Tensor* t);
bool tensor_is_contiguous(Tensor* t);
int* tensor_shape_copy(int* shape, int ndim);

Tensor* tensor_from_flat(const float* data, int rows, int cols);

/* TENSOR2D(2, 3, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f) */
#define TENSOR2D(rows, cols, ...) tensor_from_array_2d((float[]){__VA_ARGS__}, rows, cols)

Tensor* tensor_from_array_2d(const float* data, int rows, int cols);
Tensor* tensor_zeros_2d(int rows, int cols);
Tensor* tensor_ones_2d(int rows, int cols);
Tensor* tensor_empty_2d(int rows, int cols);
int* tensor_shape(int ndim, ...);
int tensor_to_device(Tensor* tensor, DeviceType device);
Tensor* tensor_from_ir_node(struct IRNode* node, CMLGraph_t ir_context);

Tensor* tensor_arange(float start, float end, float step, const TensorConfig* config);
Tensor* tensor_linspace(float start, float end, int steps, const TensorConfig* config);
Tensor* tensor_eye(int n, const TensorConfig* config);
Tensor* tensor_rand(int* shape, int ndim, const TensorConfig* config);
Tensor* tensor_randn(int* shape, int ndim, const TensorConfig* config);
Tensor* tensor_randint(int low, int high, int* shape, int ndim, const TensorConfig* config);
void tensor_manual_seed(uint64_t seed);
Tensor* tensor_zeros_like(Tensor* a);
Tensor* tensor_ones_like(Tensor* a);
Tensor* tensor_rand_like(Tensor* a);
Tensor* tensor_randn_like(Tensor* a);
Tensor* tensor_full_like(Tensor* a, float value);

/* dim=-1 to squeeze all */
Tensor* tensor_squeeze(Tensor* a, int dim);
Tensor* tensor_unsqueeze(Tensor* a, int dim);
Tensor* tensor_flip(Tensor* a, int dim);
Tensor* tensor_repeat(Tensor* a, int* repeats, int num_repeats);
Tensor** tensor_split(Tensor* a, int num_splits, int dim, int* out_count);
Tensor** tensor_chunk(Tensor* a, int chunks, int dim, int* out_count);

Tensor* tensor_kaiming_uniform(int* shape, int ndim, int fan_in, const TensorConfig* config);
Tensor* tensor_kaiming_normal(int* shape, int ndim, int fan_in, const TensorConfig* config);
Tensor* tensor_glorot_uniform(int* shape, int ndim, int fan_in, int fan_out, const TensorConfig* config);
Tensor* tensor_xavier_normal(int* shape, int ndim, int fan_in, int fan_out, const TensorConfig* config);

Tensor* tensor_cast(Tensor* a, DType dtype);
Tensor* tensor_contiguous(Tensor* a);

/* Caller retains ownership of data */
Tensor* tensor_from_blob(void* data, int* shape, int ndim, const TensorConfig* config);

Tensor* tensor_randperm(int n, const TensorConfig* config);
Tensor* tensor_half(Tensor* a);
Tensor* tensor_float(Tensor* a);
Tensor* tensor_double(Tensor* a);
Tensor* tensor_int(Tensor* a);
Tensor* tensor_long(Tensor* a);
Tensor* tensor_short(Tensor* a);
Tensor* tensor_bool(Tensor* a);
Tensor* tensor_bfloat16(Tensor* a);
Tensor* tensor_fp8e4m3fnuz(Tensor* a);
Tensor* tensor_fp8e5m2fnuz(Tensor* a);

typedef enum {
    INTERP_NEAREST,
    INTERP_BILINEAR,
} InterpMode;

/* 4D input: [N,C,H,W] */
Tensor* tensor_interpolate(Tensor* a, int* output_size, int num_dims, InterpMode mode);

Tensor* tensor_dot(Tensor* a, Tensor* b);

typedef enum {
    SCATTER_REDUCE_SUM,
    SCATTER_REDUCE_PROD,
    SCATTER_REDUCE_MEAN,
    SCATTER_REDUCE_AMAX,
    SCATTER_REDUCE_AMIN,
} ScatterReduceMode;

/* self[index[i]] = reduce(self[index[i]], src[i]) */
Tensor* tensor_scatter_reduce(Tensor* self, int dim, Tensor* index, Tensor* src, ScatterReduceMode mode);

/* Reinterpret bits, no conversion */
Tensor* tensor_bitcast(Tensor* a, DType target_dtype);

typedef struct {
    Tensor* Q;  // Orthogonal matrix [m, m] or [m, k] (reduced)
    Tensor* R;  // Upper triangular [m, n] or [k, n] (reduced)
} QRResult;

/* Householder reflections, reduced form: Q=[m,k], R=[k,n], k=min(m,n) */
QRResult tensor_qr(Tensor* a);

typedef struct {
    Tensor* U;  // Left singular vectors [m, k]
    Tensor* S;  // Singular values [k]
    Tensor* Vt; // Right singular vectors transposed [k, n]
} SVDResult;

/* One-sided Jacobi, reduced form, k=min(m,n) */
SVDResult tensor_svd(Tensor* a);

Tensor* tensor_from_url(const char* url);

Tensor* tensor_where(Tensor* condition, Tensor* x, Tensor* y);
Tensor* tensor_einsum(const char* equation, Tensor** tensors, int num_tensors);
Tensor* tensor_one_hot(Tensor* indices, int num_classes);
Tensor* tensor_multinomial(Tensor* probs, int num_samples, bool replacement);
Tensor* tensor_roll(Tensor* t, int shift, int axis);
Tensor* tensor_nonzero(Tensor* t);
Tensor* tensor_copysign(Tensor* a, Tensor* b);
Tensor* tensor_logaddexp(Tensor* a, Tensor* b);

int tensor_assign(Tensor* t, Tensor* src);
int tensor_assign_data(Tensor* t, const void* data, size_t nbytes);

uint64_t tensor_hash(Tensor* t);
int tensor_keccak(Tensor* t, uint8_t* out, size_t out_len);

#ifdef __cplusplus
}
#endif

#endif // CML_TENSOR_H
