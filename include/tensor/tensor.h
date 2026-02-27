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

// Minimal DType enum
typedef enum { DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_INT32, DTYPE_INT64, DTYPE_BOOL } DType;

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

#ifdef __cplusplus
}
#endif

#endif // CML_TENSOR_H
