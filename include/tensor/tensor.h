#ifndef CML_TENSOR_H
#define CML_TENSOR_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

// Minimal DType enum
typedef enum { DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_INT32, DTYPE_INT64, DTYPE_BOOL } DType;

// Minimal Device enum
typedef enum { DEVICE_CPU, DEVICE_CUDA } DeviceType;

// Forward declaration for autograd
struct Function;

// Production Tensor struct with stride tracking
typedef struct Tensor {
    // Core data
    void* data;            // Raw data pointer
    int* shape;            // Shape array
    size_t* strides;       // Stride array (for efficient views)
    int ndim;              // Number of dimensions
    size_t numel;          // Total number of elements
    size_t storage_offset; // Offset into data (for views/slices)

    // Type and device
    DType dtype;       // Data type
    DeviceType device; // Device

    // Layout flags
    bool is_contiguous; // Is memory layout contiguous?
    bool owns_data;     // Does this tensor own its data?

    // Autograd fields
    struct Tensor* grad;      // Gradient tensor
    struct Function* grad_fn; // Creator function
    bool requires_grad;       // Track gradients?

    // Memory management
    int ref_count;       // Reference counting
    struct Tensor* base; // Base tensor (if this is a view)
} Tensor;

// Essential utility functions
size_t dtype_size(DType dtype);
size_t tensor_numel(int* shape, int ndim);

// Stride utilities
size_t* compute_contiguous_strides(int* shape, int ndim);
bool check_is_contiguous(int* shape, size_t* strides, int ndim);
size_t compute_storage_size(int* shape, size_t* strides, int ndim);

// Tensor creation (minimal API)
Tensor* tensor_empty(int* shape, int ndim, DType dtype, DeviceType device);
Tensor* tensor_zeros(int* shape, int ndim, DType dtype, DeviceType device);
Tensor* tensor_ones(int* shape, int ndim, DType dtype, DeviceType device);
Tensor* tensor_from_data(void* data, int* shape, int ndim, DType dtype, DeviceType device);

// View operations
Tensor* tensor_view(Tensor* t, int* new_shape, int new_ndim);
Tensor* tensor_as_strided(Tensor* t, int* shape, int ndim, size_t* strides, size_t storage_offset);
Tensor* tensor_reshape(Tensor* t, int* new_shape, int new_ndim);
Tensor* tensor_contiguous(Tensor* t);

// Memory management
void tensor_free(Tensor* t);
Tensor* tensor_clone(Tensor* t);

// Data access helpers (stride-aware)
float tensor_get_float(Tensor* t, size_t idx);
void tensor_set_float(Tensor* t, size_t idx, float value);
void* tensor_data_ptr(Tensor* t);
size_t tensor_compute_offset(Tensor* t, int* indices);

// Shape utilities
bool tensor_is_scalar(Tensor* t);
bool tensor_is_contiguous(Tensor* t);
int* tensor_shape_copy(int* shape, int ndim);

#endif // CML_TENSOR_H
