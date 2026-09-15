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

typedef enum {
    CML_QUANT_NONE = 0,
    CML_QUANT_GGUF_Q8_0,
    CML_QUANT_GGUF_Q4_0,
    CML_QUANT_AFFINE_INT8, /* per-tensor affine int8: q = round(x/scale)+zero_point */
    CML_QUANT_AFFINE_INT4, /* per-tensor symmetric int4, packed 2/byte in quant_data */
    CML_QUANT_NF4,         /* block-wise NF4, [scales | packed nibbles] in quant_data */
} CMLQuantType;

/* Shared storage: keeps a data block alive while any view references it.
 * Attached lazily to the owning root tensor the first time a view is taken;
 * the block is freed when the last referencing tensor (owner or view) dies. */
typedef struct CMLTensorStorage {
    void* data;             /* The shared block (== owner's data pointer) */
    size_t nbytes;          /* Payload bytes of the original allocation */
    int refs;               /* Owner + live views sharing this block */
    bool from_buffer_cache; /* Block came from cml_buffer_cache_alloc */
    DeviceType device;      /* Device the block lives on */
} CMLTensorStorage;

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

    /* Saved IR linkage for tensor_unrealize (gradient checkpointing).
     * Populated by tensor_realize before it detaches the live ir_node. */
    struct IRNode* saved_ir_node;
    CMLGraph_t saved_ir_context;

    // Execution state (lazy)
    bool is_executed; // Has this been executed?
    void* data;       // NULL until executed (lazy!)
    bool owns_data;        // Does this tensor own its data?
    bool from_buffer_cache; // Data was allocated via cml_buffer_cache_alloc

    bool requires_grad;
    bool retains_grad; /* PyTorch retain_grad(): keep .grad after backward */
    struct Tensor* grad; // Gradient tensor (also lazy!)

    int ref_count;       // Reference counting (internal graph/API references)
    /* External references held outside the C core (e.g. a Python/other-language
     * wrapper that will free the tensor itself). While > 0, a graph teardown
     * (cml_ir_free) must NOT free this tensor out from under its owner: it is
     * detached from the graph and kept alive instead. See tensor_pin/
     * tensor_release and tensor_free. */
    int external_refs;
    struct Tensor* base; // Base tensor (if this is a view)
    CMLTensorStorage* storage; // Shared storage block (owner + its views)

    size_t* strides;       // Stride array (for efficient views)
    size_t storage_offset; // Offset into data (for views/slices)
    bool is_contiguous;    // Is memory layout contiguous?
    CMLBackendBuffer_t buffer_handle;

    void* user_data;
    void* backward_hooks; /* TensorHookList*, owned by autograd */

    CMLQuantType quant_type;
    void* quant_data;
    size_t quant_data_bytes;
    /* Affine-int8 dequant params (used when quant_type == CML_QUANT_AFFINE_INT8;
     * the int8 weights live in ->data). dequant = (q - zero_point) * scale. */
    float quant_scale;
    int32_t quant_zero_point;
    /* Block size for block-wise quant types (CML_QUANT_NF4); 0 otherwise. */
    int32_t quant_block_size;
} Tensor;

size_t cml_dtype_size(DType dtype);
DType cml_promote_dtype(DType dtype1, DType dtype2);

/* Overflow-checked shape product: returns false if any dim is negative or
 * the product exceeds SIZE_MAX (*out untouched then). */
bool tensor_numel_checked(const int* shape, int ndim, size_t* out);

/* Unchecked legacy wrapper: partial product if a dim is negative. Prefer
 * tensor_numel_checked on unvalidated shapes. */
size_t tensor_numel(int* shape, int ndim);

/* numel * cml_dtype_size(dtype), checked against SIZE_MAX overflow. */
bool tensor_nbytes_checked(size_t numel, DType dtype, size_t* out);

/* Precision-preserving element-wise dtype conversion of a raw buffer of n
 * elements (int->int lossless via int64, else via double). Returns 0 on
 * success, -1 if a dtype isn't directly supported (f16/bf16/fp8). */
int cml_cast_buffer(const void* src, DType from, void* dst, DType to, size_t n);

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

/**
 * @brief Allocate a contiguous eager tensor with uninitialized storage (no IR graph).
 *
 * Used when callers need an immediate backing buffer (e.g. arena or external init).
 */
Tensor* tensor_create(DType dtype, DeviceType device, int ndim, const int* shape,
                      bool requires_grad);

Tensor* tensor_zeros(int* shape, int ndim, const TensorConfig* config);
Tensor* tensor_ones(int* shape, int ndim, const TensorConfig* config);
Tensor* tensor_full(int* shape, int ndim, const TensorConfig* config, float value);
Tensor* tensor_from_data(const void* data, int* shape, int ndim, const TensorConfig* config);

#include "tensor/tensor_views.h"

void tensor_free(Tensor* t);

/* External-ownership refcount (for language bindings). tensor_pin() records that
 * an external owner holds this tensor; tensor_release() drops that hold and frees
 * the tensor if nothing else references it. While pinned, a graph teardown detaches
 * the tensor instead of freeing it, so the owner's later free is always safe. */
void tensor_pin(Tensor* t);
void tensor_release(Tensor* t);

/* Detach a tensor from the IR graph without freeing it: copy any borrowed
 * execution-plan data into an owned allocation and clear all links into the
 * graph. Graph teardown uses this for tensors an external owner still holds. */
void tensor_detach_keep(Tensor* t);

/* Shared-storage lifetime for views. tensor_storage_share() links `view` to
 * the storage block behind `src` (attaching one to src's root if needed), so
 * the block outlives the base temporary; call it once at view creation.
 * tensor_storage_release() drops a tensor's hold and frees the block when the
 * last reference goes — use it wherever owned data is freed or replaced. */
void tensor_storage_share(Tensor* view, Tensor* src);
void tensor_storage_release(Tensor* t);
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
