#ifndef CML_TENSOR_H
#define CML_TENSOR_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include "backend/device.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration to avoid circular dependency with backend_buffer.h
struct CMLBackendBuffer;
typedef struct CMLBackendBuffer* CMLBackendBuffer_t;

// DType enum
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
    DTYPE_FLOAT8_E4M3,  // 1 sign, 4 exponent, 3 mantissa (range [-448, 448])
    DTYPE_FLOAT8_E5M2,  // 1 sign, 5 exponent, 2 mantissa (range [-57344, 57344])
} DType;

// DeviceType is now defined in Core/device.h

// Forward declaration for IR (to avoid circular dependency)
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
    bool owns_data;   // Does this tensor own its data?

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

/**
 * @brief Get size of data type in bytes
 * @param dtype Data type
 * @return Size in bytes
 */
size_t cml_dtype_size(DType dtype);

/**
 * @brief Calculate total number of elements from shape
 * @param shape Shape array
 * @param ndim Number of dimensions
 * @return Total number of elements
 */
size_t tensor_numel(int* shape, int ndim);

/**
 * @brief Promote two dtypes to a common dtype
 * @param dtype1 First data type
 * @param dtype2 Second data type
 * @return Promoted data type
 */
DType cml_promote_dtype(DType dtype1, DType dtype2);

/**
 * @brief Compute contiguous strides from shape
 * @param shape Shape array
 * @param ndim Number of dimensions
 * @return Allocated strides array (caller must free)
 */
size_t* compute_contiguous_strides(int* shape, int ndim);

/**
 * @brief Check if tensor is contiguous based on shape and strides
 * @param shape Shape array
 * @param strides Strides array
 * @param ndim Number of dimensions
 * @return true if contiguous, false otherwise
 */
bool tensor_check_is_contiguous(int* shape, size_t* strides, int ndim);

/**
 * @brief Compute storage size required for strided tensor
 * @param shape Shape array
 * @param strides Strides array
 * @param ndim Number of dimensions
 * @return Storage size in elements
 */
size_t tensor_compute_storage_size(int* shape, size_t* strides, int ndim);

/**
 * @brief Tensor creation configuration options
 *
 * All fields are optional. Use NULL to use defaults, or set specific fields.
 * Defaults: dtype=DTYPE_FLOAT32, device=DEVICE_AUTO (auto-detected)
 */
typedef struct TensorConfig {
    DType dtype;       // Data type (use -1 or DTYPE_FLOAT32 for default)
    DeviceType device; // Device (use DEVICE_AUTO for auto-detection)
    bool has_dtype;    // Set to true if dtype is explicitly set
    bool has_device;   // Set to true if device is explicitly set
} TensorConfig;

/**
 * @brief Create empty tensor
 *
 * @param shape Shape array
 * @param ndim Number of dimensions
 * @param config Configuration options (NULL for defaults: dtype=DTYPE_FLOAT32,
 * device=auto-detected) Can also pass NULL to use all defaults
 * @return New tensor, or NULL on failure
 */
Tensor* tensor_empty(int* shape, int ndim, const TensorConfig* config);

/**
 * @brief Create zeros tensor
 *
 * @param shape Shape array
 * @param ndim Number of dimensions
 * @param config Configuration options (NULL for defaults: dtype=DTYPE_FLOAT32,
 * device=auto-detected)
 * @return New tensor, or NULL on failure
 */
Tensor* tensor_zeros(int* shape, int ndim, const TensorConfig* config);

/**
 * @brief Create ones tensor
 *
 * @param shape Shape array
 * @param ndim Number of dimensions
 * @param config Configuration options (NULL for defaults: dtype=DTYPE_FLOAT32,
 * device=auto-detected)
 * @return New tensor, or NULL on failure
 */
Tensor* tensor_ones(int* shape, int ndim, const TensorConfig* config);

/**
 * @brief Create tensor filled with a constant value
 *
 * @param shape Shape array
 * @param ndim Number of dimensions
 * @param config Configuration options (NULL for defaults: dtype=DTYPE_FLOAT32,
 * device=auto-detected)
 * @param value Value to fill the tensor with
 * @return New tensor, or NULL on failure
 */
Tensor* tensor_full(int* shape, int ndim, const TensorConfig* config, float value);

/**
 * @brief Create tensor from data
 *
 * @param data Source data pointer (assumed to be on CPU)
 * @param shape Shape array
 * @param ndim Number of dimensions
 * @param config Configuration options (NULL for defaults: dtype=DTYPE_FLOAT32,
 * device=auto-detected)
 * @return New tensor, or NULL on failure
 */
Tensor* tensor_from_data(const void* data, int* shape, int ndim, const TensorConfig* config);

#include "tensor/tensor_views.h"

/**
 * @brief Free tensor and its resources
 * @param t Tensor to free
 */
void tensor_free(Tensor* t);

/**
 * @brief Clone tensor
 * @param t Input tensor
 * @return Cloned tensor
 */
Tensor* tensor_clone(Tensor* t);

/**
 * @brief Get float value at flat index
 * @param t Input tensor
 * @param idx Flat index
 * @return Float value
 */
float tensor_get_float(Tensor* t, size_t idx);

/**
 * @brief Set float value at flat index
 * @param t Input tensor
 * @param idx Flat index
 * @param value Float value
 */
void tensor_set_float(Tensor* t, size_t idx, float value);

/**
 * @brief Get raw data pointer
 * @param t Input tensor
 * @return Pointer to raw data (triggers execution if lazy)
 */
void* tensor_data_ptr(Tensor* t); // Triggers lazy execution if needed

/**
 * @brief Compute storage offset from multi-dimensional indices
 * @param t Input tensor
 * @param indices Array of indices
 * @return Storage offset
 */
size_t tensor_compute_offset(Tensor* t, int* indices);

/**
 * @brief Ensure tensor is executed
 * @param t Input tensor
 * @return 0 on success, negative on failure
 */
int tensor_ensure_executed(Tensor* t); // Explicit execution

/**
 * @brief Get IR context associated with tensor
 * @param t Input tensor
 * @return IR context
 */
CMLGraph_t tensor_get_ir_context(Tensor* t); // Get IR context

/**
 * @brief Check if tensor is a scalar (0-dim)
 * @param t Input tensor
 * @return true if scalar, false otherwise
 */
bool tensor_is_scalar(Tensor* t);

/**
 * @brief Check if tensor is contiguous in memory
 * @param t Input tensor
 * @return true if contiguous, false otherwise
 */
bool tensor_is_contiguous(Tensor* t);

/**
 * @brief Create a copy of shape array
 * @param shape Source shape array
 * @param ndim Number of dimensions
 * @return Allocated copy of shape array (caller must free)
 */
int* tensor_shape_copy(int* shape, int ndim);

/**
 * @brief Create tensor from flat array
 *
 * Uses default dtype and device from global config.
 *
 * @param data Flat array of data
 * @param rows Number of rows
 * @param cols Number of columns
 * @return New tensor, or NULL on failure
 */
Tensor* tensor_from_flat(const float* data, int rows, int cols);

/**
 * @brief Create 2D tensor from array literal
 *
 * Macro for easy tensor creation:
 *   Tensor* t = TENSOR2D(2, 3, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @param ... Data values (rows * cols values)
 */
#define TENSOR2D(rows, cols, ...) tensor_from_array_2d((float[]){__VA_ARGS__}, rows, cols)

/**
 * @brief Create tensor from 2D array
 *
 * @param data 2D array data (flattened)
 * @param rows Number of rows
 * @param cols Number of columns
 * @return New tensor, or NULL on failure
 */
Tensor* tensor_from_array_2d(const float* data, int rows, int cols);

/**
 * @brief Create zeros tensor with implicit shape
 *
 * Uses default dtype and device from global config.
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @return New tensor, or NULL on failure
 */
Tensor* tensor_zeros_2d(int rows, int cols);

/**
 * @brief Create ones tensor with implicit shape
 *
 * Uses default dtype and device from global config.
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @return New tensor, or NULL on failure
 */
Tensor* tensor_ones_2d(int rows, int cols);

/**
 * @brief Create empty tensor with implicit shape
 *
 * Uses default dtype and device from global config.
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @return New tensor, or NULL on failure
 */
Tensor* tensor_empty_2d(int rows, int cols);

/**
 * @brief Helper to create shape array
 *
 * @param ndim Number of dimensions
 * @param ... Shape values (ndim arguments)
 * @return Allocated shape array (caller must free)
 */
int* tensor_shape(int ndim, ...);

/**
 * @brief Move tensor to device
 *
 * @param tensor Tensor to move
 * @param device Target device
 * @return 0 on success, negative value on failure
 */
int tensor_to_device(Tensor* tensor, DeviceType device);

/**
 * @brief Create tensor facade from IR node
 *
 * @param node IR node
 * @param ir_context IR context
 * @return New tensor facade
 */
Tensor* tensor_from_ir_node(struct IRNode* node, CMLGraph_t ir_context);

/** @brief Create 1D tensor with evenly spaced values in [start, end) */
Tensor* tensor_arange(float start, float end, float step, const TensorConfig* config);

/** @brief Create 1D tensor with linearly spaced values between start and end */
Tensor* tensor_linspace(float start, float end, int steps, const TensorConfig* config);

/** @brief Create NxN identity matrix */
Tensor* tensor_eye(int n, const TensorConfig* config);

/** @brief Create tensor with uniform random values in [0, 1) */
Tensor* tensor_rand(int* shape, int ndim, const TensorConfig* config);

/** @brief Create tensor with standard normal random values */
Tensor* tensor_randn(int* shape, int ndim, const TensorConfig* config);

/** @brief Create tensor with random integers in [low, high) */
Tensor* tensor_randint(int low, int high, int* shape, int ndim, const TensorConfig* config);

/** @brief Set the random seed for reproducibility */
void tensor_manual_seed(uint64_t seed);

/** @brief Create zeros tensor with same shape/dtype/device as input */
Tensor* tensor_zeros_like(Tensor* a);

/** @brief Create ones tensor with same shape/dtype/device as input */
Tensor* tensor_ones_like(Tensor* a);

/** @brief Create random uniform tensor with same shape/dtype/device as input */
Tensor* tensor_rand_like(Tensor* a);

/** @brief Create random normal tensor with same shape/dtype/device as input */
Tensor* tensor_randn_like(Tensor* a);

/** @brief Create constant tensor with same shape/dtype/device as input */
Tensor* tensor_full_like(Tensor* a, float value);

/** @brief Remove dimensions of size 1. dim=-1 to squeeze all. */
Tensor* tensor_squeeze(Tensor* a, int dim);

/** @brief Insert a dimension of size 1 at the specified position */
Tensor* tensor_unsqueeze(Tensor* a, int dim);

/** @brief Reverse elements along a dimension */
Tensor* tensor_flip(Tensor* a, int dim);

/** @brief Repeat tensor along each dimension */
Tensor* tensor_repeat(Tensor* a, int* repeats, int num_repeats);

/** @brief Split tensor into chunks along a dimension */
Tensor** tensor_split(Tensor* a, int num_splits, int dim, int* out_count);

/** @brief Split tensor into chunks (alias for tensor_split) */
Tensor** tensor_chunk(Tensor* a, int chunks, int dim, int* out_count);

/** @brief Kaiming uniform initialization (for ReLU networks) */
Tensor* tensor_kaiming_uniform(int* shape, int ndim, int fan_in, const TensorConfig* config);

/** @brief Kaiming normal initialization (for ReLU networks) */
Tensor* tensor_kaiming_normal(int* shape, int ndim, int fan_in, const TensorConfig* config);

/** @brief Glorot/Xavier uniform initialization */
Tensor* tensor_glorot_uniform(int* shape, int ndim, int fan_in, int fan_out, const TensorConfig* config);

/** @brief Xavier normal initialization */
Tensor* tensor_xavier_normal(int* shape, int ndim, int fan_in, int fan_out, const TensorConfig* config);

/** @brief Cast tensor to a different dtype */
Tensor* tensor_cast(Tensor* a, DType dtype);

/** @brief Make a contiguous copy of tensor */
Tensor* tensor_contiguous(Tensor* a);

/** @brief Create tensor from existing memory without copy (caller retains ownership) */
Tensor* tensor_from_blob(void* data, int* shape, int ndim, const TensorConfig* config);

/** @brief Create random permutation of integers [0, n) */
Tensor* tensor_randperm(int n, const TensorConfig* config);

/** @brief Convenience cast to float16 */
Tensor* tensor_half(Tensor* a);

/** @brief Convenience cast to float32 */
Tensor* tensor_float(Tensor* a);

/** @brief Convenience cast to float64 */
Tensor* tensor_double(Tensor* a);

/** @brief Convenience cast to int32 */
Tensor* tensor_int(Tensor* a);

/** @brief Convenience cast to int64 */
Tensor* tensor_long(Tensor* a);

/** @brief Convenience cast to int16 */
Tensor* tensor_short(Tensor* a);

/** @brief Convenience cast to bool */
Tensor* tensor_bool(Tensor* a);

/** @brief Convenience cast to bfloat16 */
Tensor* tensor_bfloat16(Tensor* a);

/** @brief Interpolation modes */
typedef enum {
    INTERP_NEAREST,
    INTERP_BILINEAR,
} InterpMode;

/** @brief Interpolate/upsample tensor (4D: [N,C,H,W]) */
Tensor* tensor_interpolate(Tensor* a, int* output_size, int num_dims, InterpMode mode);

/** @brief Dot product of two 1D tensors */
Tensor* tensor_dot(Tensor* a, Tensor* b);

/** @brief Scatter reduce modes */
typedef enum {
    SCATTER_REDUCE_SUM,
    SCATTER_REDUCE_PROD,
    SCATTER_REDUCE_MEAN,
    SCATTER_REDUCE_AMAX,
    SCATTER_REDUCE_AMIN,
} ScatterReduceMode;

/** @brief Scatter with reduction: self[index[i]] = reduce(self[index[i]], src[i]) */
Tensor* tensor_scatter_reduce(Tensor* self, int dim, Tensor* index, Tensor* src, ScatterReduceMode mode);

/** @brief Reinterpret tensor bits as a different dtype (no conversion) */
Tensor* tensor_bitcast(Tensor* a, DType target_dtype);

/** @brief QR decomposition result */
typedef struct {
    Tensor* Q;  // Orthogonal matrix [m, m] or [m, k] (reduced)
    Tensor* R;  // Upper triangular [m, n] or [k, n] (reduced)
} QRResult;

/** @brief QR decomposition via Householder reflections (reduced form: Q=[m,k], R=[k,n], k=min(m,n)) */
QRResult tensor_qr(Tensor* a);

/** @brief SVD decomposition result */
typedef struct {
    Tensor* U;  // Left singular vectors [m, k]
    Tensor* S;  // Singular values [k]
    Tensor* Vt; // Right singular vectors transposed [k, n]
} SVDResult;

/** @brief SVD decomposition via one-sided Jacobi (reduced form, k=min(m,n)) */
SVDResult tensor_svd(Tensor* a);

/** @brief Download tensor from URL and load it */
Tensor* tensor_from_url(const char* url);

#ifdef __cplusplus
}
#endif

#endif // CML_TENSOR_H
