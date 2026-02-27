#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>
#include "tensor/tensor.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "backend/backend_buffer.h"
#include "core/logging.h"
#include "backend/device.h"
#include "core/error_stack.h"
#include "core/config.h"

static void resolve_config(const TensorConfig* config, DType* dtype, DeviceType* device) {
    if (!config) {
        *dtype  = DTYPE_FLOAT32;
        *device = device_get_default();
        return;
    }

    *dtype = config->dtype;
    if (config->device == DEVICE_AUTO) {
        *device = device_get_best_available();
    } else {
        *device = config->device;
    }
}

Tensor* tensor_create(DType dtype, DeviceType device, int ndim, const int* shape,
                      bool requires_grad) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t)
        return NULL;

    t->dtype  = dtype;
    t->device = device;
    t->ndim   = ndim;
    t->shape  = (int*)malloc(ndim * sizeof(int));
    if (!t->shape) {
        free(t);
        return NULL;
    }
    memcpy(t->shape, shape, ndim * sizeof(int));

    size_t total_size = 1;
    for (int i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }

    switch (dtype) {
    case DTYPE_FLOAT32:
        total_size *= sizeof(float);
        break;
    case DTYPE_FLOAT64:
        total_size *= sizeof(double);
        break;
    case DTYPE_INT32:
        total_size *= sizeof(int32_t);
        break;
    case DTYPE_INT64:
        total_size *= sizeof(int64_t);
        break;
    case DTYPE_BOOL:
        total_size *= sizeof(uint8_t);
        break;
    }

    t->data = malloc(total_size);
    if (!t->data) {
        free(t->shape);
        free(t);
        return NULL;
    }

    t->requires_grad  = requires_grad;
    t->is_executed    = true; // Leaf tensors are already "executed"
    t->ir_context     = NULL;
    t->ir_node        = NULL;
    t->grad           = NULL; // Initialize gradient to NULL
    t->ref_count      = 1;
    t->base           = NULL;
    t->strides        = NULL;
    t->storage_offset = 0;
    t->is_contiguous  = true;
    t->buffer_handle  = NULL;
    t->user_data      = NULL;
    t->owns_data      = true;

    // Calculate numel
    t->numel = 1;
    for (int i = 0; i < ndim; i++) {
        t->numel *= shape[i];
    }

    return t;
}

Tensor* tensor_zeros(int* shape, int ndim, const TensorConfig* config) {
    DType dtype;
    DeviceType device;
    resolve_config(config, &dtype, &device);

    Tensor* t = tensor_create(dtype, device, ndim, shape, false);
    if (!t)
        return NULL;

    size_t total_size = 1;
    for (int i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }

    memset(t->data, 0, total_size * cml_dtype_size(dtype));
    return t;
}

Tensor* tensor_ones(int* shape, int ndim, const TensorConfig* config) {
    DType dtype;
    DeviceType device;
    resolve_config(config, &dtype, &device);

    Tensor* t = tensor_create(dtype, device, ndim, shape, false);
    if (!t)
        return NULL;

    size_t total_size = 1;
    for (int i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }

    switch (dtype) {
    case DTYPE_BOOL:
        memset(t->data, 1, total_size * sizeof(uint8_t));
        break;
    case DTYPE_INT32:
        for (size_t i = 0; i < total_size; i++) {
            ((int32_t*)t->data)[i] = 1;
        }
        break;
    case DTYPE_INT64:
        for (size_t i = 0; i < total_size; i++) {
            ((int64_t*)t->data)[i] = 1;
        }
        break;
    case DTYPE_FLOAT32:
        for (size_t i = 0; i < total_size; i++) {
            ((float*)t->data)[i] = 1.0f;
        }
        break;
    case DTYPE_FLOAT64:
        for (size_t i = 0; i < total_size; i++) {
            ((double*)t->data)[i] = 1.0;
        }
        break;
    }

    return t;
}

Tensor* tensor_full(int* shape, int ndim, const TensorConfig* config, float value) {
    DType dtype;
    DeviceType device;
    resolve_config(config, &dtype, &device);

    Tensor* t = tensor_create(dtype, device, ndim, shape, false);
    if (!t)
        return NULL;

    size_t total_size = 1;
    for (int i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }

    switch (dtype) {
    case DTYPE_FLOAT32:
        for (size_t i = 0; i < total_size; i++) {
            ((float*)t->data)[i] = (float)value;
        }
        break;
    case DTYPE_FLOAT64:
        for (size_t i = 0; i < total_size; i++) {
            ((double*)t->data)[i] = (double)value;
        }
        break;
    case DTYPE_INT32:
        for (size_t i = 0; i < total_size; i++) {
            ((int32_t*)t->data)[i] = (int32_t)value;
        }
        break;
    case DTYPE_INT64:
        for (size_t i = 0; i < total_size; i++) {
            ((int64_t*)t->data)[i] = (int64_t)value;
        }
        break;
    case DTYPE_BOOL:
        for (size_t i = 0; i < total_size; i++) {
            ((uint8_t*)t->data)[i] = (uint8_t)(fabsf(value) > 1e-9f);
        }
        break;
    }

    return t;
}

void* tensor_data_ptr(Tensor* t) {
    if (!t)
        return NULL;

    // Leaf tensors (no IR node) already have data, just return it
    if (!t->ir_node || !t->ir_context) {
        return t->data;
    }

    // If not executed, execute now! (lazy evaluation)
    if (!t->is_executed) {
        // Execute entire graph up to this node
        int ret = cml_ir_execute_up_to(t->ir_context, t->ir_node);
        if (ret == 0) {
            t->is_executed = true;
            // JIT execution uses destination-passing style, so data is written
            // directly to t->data by the JIT function. If data is still NULL,
            // allocate it (shouldn't happen, but be safe).
            if (!t->data && t->numel > 0) {
                size_t size = t->numel * cml_dtype_size(t->dtype);
                t->data     = calloc(1, size);
                if (!t->data) {
                    LOG_ERROR("Failed to allocate data after execution");
                    return NULL;
                }
            }
        } else {
            // Execution failed - try to allocate zero-initialized data to prevent crashes
            LOG_WARNING("IR execution failed, returning zero-initialized data");
            if (!t->data && t->numel > 0) {
                size_t size = t->numel * cml_dtype_size(t->dtype);
                t->data     = calloc(1, size);
            }
            t->is_executed = true;
            return t->data;
        }
    }

    return t->data;
}

size_t cml_dtype_size(DType dtype) {
    switch (dtype) {
    case DTYPE_FLOAT32:
        return sizeof(float);
    case DTYPE_FLOAT64:
        return sizeof(double);
    case DTYPE_INT32:
        return sizeof(int32_t);
    case DTYPE_INT64:
        return sizeof(int64_t);
    case DTYPE_BOOL:
        return sizeof(uint8_t);
    default:
        return sizeof(float);
    }
}

/**
 * @brief Promote two dtypes to a common dtype following NumPy rules
 *
 * Promotion order (lowest to highest):
 * - BOOL < INT32 < INT64 < FLOAT32 < FLOAT64
 *
 * @param dtype1 First dtype
 * @param dtype2 Second dtype
 * @return Promoted dtype (the higher precision one)
 */
DType cml_promote_dtype(DType dtype1, DType dtype2) {
    // Same dtype - no promotion needed
    if (dtype1 == dtype2) {
        return dtype1;
    }

    // Promotion hierarchy: BOOL < INT32 < INT64 < FLOAT32 < FLOAT64
    int rank1 = 0, rank2 = 0;

    switch (dtype1) {
    case DTYPE_BOOL:
        rank1 = 0;
        break;
    case DTYPE_INT32:
        rank1 = 1;
        break;
    case DTYPE_INT64:
        rank1 = 2;
        break;
    case DTYPE_FLOAT32:
        rank1 = 3;
        break;
    case DTYPE_FLOAT64:
        rank1 = 4;
        break;
    default:
        rank1 = 0;
        break;
    }

    switch (dtype2) {
    case DTYPE_BOOL:
        rank2 = 0;
        break;
    case DTYPE_INT32:
        rank2 = 1;
        break;
    case DTYPE_INT64:
        rank2 = 2;
        break;
    case DTYPE_FLOAT32:
        rank2 = 3;
        break;
    case DTYPE_FLOAT64:
        rank2 = 4;
        break;
    default:
        rank2 = 0;
        break;
    }

    return (rank1 > rank2) ? dtype1 : dtype2;
}

size_t tensor_numel(int* shape, int ndim) {
    size_t numel = 1;
    for (int i = 0; i < ndim; i++) {
        numel *= (size_t)shape[i];
    }
    return numel;
}

size_t* compute_contiguous_strides(int* shape, int ndim) {
    if (!shape || ndim < 0)
        return NULL;

    if (ndim == 0) {
        // Scalar tensor has no strides, but some functions might expect a non-NULL pointer
        // Return a dummy pointer that can be freed
        return (size_t*)malloc(sizeof(size_t));
    }

    size_t* strides = (size_t*)malloc((size_t)ndim * sizeof(size_t));
    if (!strides) {
        LOG_ERROR("Failed to allocate memory for strides");
        return NULL;
    }

    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * (size_t)shape[i + 1];
    }

    return strides;
}

bool tensor_check_is_contiguous(int* shape, size_t* strides, int ndim) {
    if (!shape || !strides || ndim <= 0)
        return false;

    size_t expected_stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        if (strides[i] != expected_stride)
            return false;
        expected_stride *= (size_t)shape[i];
    }

    return true;
}

size_t tensor_compute_storage_size(int* shape, size_t* strides, int ndim) {
    if (!shape || !strides || ndim <= 0)
        return 0;

    size_t max_offset = 0;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] > 1) {
            max_offset += (size_t)(shape[i] - 1) * strides[i];
        }
    }

    return max_offset + 1;
}

int* tensor_shape_copy(int* shape, int ndim) {
    int* new_shape = (int*)malloc((size_t)ndim * sizeof(int));
    if (!new_shape) {
        LOG_ERROR("Failed to allocate memory for tensor shape copy");
        return NULL;
    }
    memcpy(new_shape, shape, (size_t)ndim * sizeof(int));
    return new_shape;
}

float tensor_get_float(Tensor* t, size_t idx) {
    // Trigger lazy execution if needed
    if (t && !t->is_executed) {
        void* data = tensor_data_ptr(t);
        if (!data)
            return 0.0f;
    }
    if (!t || !t->data || idx >= t->numel)
        return 0.0f;

    size_t offset = t->storage_offset;
    if (!t->is_contiguous) {
        size_t temp = idx;
        for (int d = t->ndim - 1; d >= 0; d--) {
            size_t coord = temp % (size_t)t->shape[d];
            temp /= (size_t)t->shape[d];
            offset += coord * t->strides[d];
        }
    } else {
        offset += idx;
    }

    switch (t->dtype) {
    case DTYPE_FLOAT32:
        return ((float*)t->data)[offset];
    case DTYPE_FLOAT64:
        return (float)((double*)t->data)[offset];
    case DTYPE_INT32:
        return (float)((int32_t*)t->data)[offset];
    case DTYPE_INT64:
        return (float)((int64_t*)t->data)[offset];
    case DTYPE_BOOL:
        return (float)((uint8_t*)t->data)[offset];
    default:
        return 0.0f;
    }
}

void tensor_set_float(Tensor* t, size_t idx, float value) {
    if (!t || idx >= t->numel)
        return;
    // Trigger lazy execution if needed
    if (!t->is_executed) {
        void* data = tensor_data_ptr(t);
        if (!data)
            return;
    }
    if (!t->data)
        return;

    size_t offset = t->storage_offset;
    if (!t->is_contiguous) {
        size_t temp = idx;
        for (int d = t->ndim - 1; d >= 0; d--) {
            size_t coord = temp % (size_t)t->shape[d];
            temp /= (size_t)t->shape[d];
            offset += coord * t->strides[d];
        }
    } else {
        offset += idx;
    }

    switch (t->dtype) {
    case DTYPE_FLOAT32:
        ((float*)t->data)[offset] = value;
        break;
    case DTYPE_FLOAT64:
        ((double*)t->data)[offset] = (double)value;
        break;
    case DTYPE_INT32:
        ((int32_t*)t->data)[offset] = (int32_t)value;
        break;
    case DTYPE_INT64:
        ((int64_t*)t->data)[offset] = (int64_t)value;
        break;
    case DTYPE_BOOL:
        ((uint8_t*)t->data)[offset] = (uint8_t)(fabsf(value) > 1e-9f);
        break;
    }
}

Tensor* tensor_from_ir_node(struct IRNode* node, CMLGraph_t ir_context) {
    if (!node || !ir_context)
        return NULL;

    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t)
        return NULL;

    // Set IR references
    t->ir_node    = node;
    t->ir_context = ir_context;

    // Link output tensor back to IR node
    node->output = t;

    // Set shape from IR node (computed by broadcasting)
    if (node->output_shape) {
        t->shape = tensor_shape_copy(node->output_shape, node->output_ndim);
        t->ndim  = node->output_ndim;
        t->numel = tensor_numel(node->output_shape, node->output_ndim);
    } else {
        // Fallback: use first input's shape if available
        if (node->inputs && node->inputs[0]) {
            t->shape = tensor_shape_copy(node->inputs[0]->shape, node->inputs[0]->ndim);
            t->ndim  = node->inputs[0]->ndim;
            t->numel = node->inputs[0]->numel;
        } else {
            free(t);
            return NULL;
        }
    }

    // Set dtype and device from first input
    if (node->inputs && node->inputs[0]) {
        t->dtype  = node->inputs[0]->dtype;
        t->device = node->inputs[0]->device;
    } else {
        t->dtype  = DTYPE_FLOAT32;
        t->device = DEVICE_CPU;
    }

    // Execution state (lazy)
    t->is_executed = false;
    t->data        = NULL; // NULL until executed
    t->owns_data   = true; // Will own data when executed

    // Autograd
    t->requires_grad = node->requires_grad;
    t->grad          = NULL; // Gradient tensor (also lazy)

    // Memory management
    t->ref_count = 1;
    t->base      = NULL;

    t->strides        = compute_contiguous_strides(t->shape, t->ndim);
    t->storage_offset = 0;
    t->is_contiguous  = true;
    t->buffer_handle  = NULL;
    t->user_data      = NULL;

    // Link node to tensor
    node->output = t;

    return t;
}

int tensor_ensure_executed(Tensor* t) {
    if (!t)
        return -1;
    if (t->is_executed)
        return 0;

    // Trigger execution
    void* data = tensor_data_ptr(t);
    return data ? 0 : -1;
}

CMLGraph_t tensor_get_ir_context(Tensor* t) { return t ? t->ir_context : NULL; }

bool tensor_is_scalar(Tensor* t) { return t && t->ndim == 0; }

bool tensor_is_contiguous(Tensor* t) {
    if (!t)
        return false;
    return t->is_contiguous;
}

Tensor* tensor_empty(int* shape, int ndim, const TensorConfig* config) {
    if (!shape || ndim < 0) {
        error_stack_push(CM_INVALID_ARGUMENT,
                         "Invalid arguments to tensor_empty: shape is NULL or ndim < 0", __FILE__,
                         __LINE__, __func__);
        return NULL;
    }

    // Resolve config to actual values
    DType dtype;
    DeviceType device;
    resolve_config(config, &dtype, &device);

    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) {
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR,
                         "Failed to allocate memory for Tensor structure", __FILE__, __LINE__,
                         __func__);
        return NULL;
    }

    t->shape = tensor_shape_copy(shape, ndim);
    if (!t->shape) {
        LOG_ERROR("Failed to allocate memory for tensor shape copy in tensor_empty");
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR, "Failed to allocate memory for tensor shape",
                         __FILE__, __LINE__, __func__);
        free(t);
        return NULL;
    }

    t->strides = compute_contiguous_strides(shape, ndim);
    if (!t->strides) {
        LOG_ERROR("Failed to allocate memory for tensor strides in tensor_empty");
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR, "Failed to allocate memory for tensor strides",
                         __FILE__, __LINE__, __func__);
        free(t->shape);
        free(t);
        return NULL;
    }

    t->ndim           = ndim;
    t->numel          = tensor_numel(shape, ndim);
    t->dtype          = dtype;
    t->device         = device;
    t->storage_offset = 0;

    // IR fields (leaf tensors are not IR nodes)
    t->ir_node     = NULL;
    t->ir_context  = NULL;
    t->is_executed = true; // Leaf tensors have data immediately
    t->data        = NULL; // Will be allocated below
    t->owns_data   = true;

    t->is_contiguous = true;
    t->buffer_handle = NULL;

    // Allocate backing buffer through backend buffer interface
    CMLBackendBufferType_t buft = cml_backend_buffer_type_for_device(device);
    if (buft) {
        size_t alloc_size = cml_backend_buffer_type_get_alloc_size(buft, t);
        if (alloc_size == 0) {
            alloc_size = t->numel * cml_dtype_size(dtype);
        }

        CMLBackendBuffer_t buffer = cml_backend_buffer_type_alloc_buffer(buft, alloc_size);
        fflush(stderr);
        if (!buffer) {
            LOG_ERROR("Failed to allocate backend buffer of size %zu for device %s", alloc_size,
                      device_get_name(device));
            char error_msg[256];
            snprintf(error_msg, sizeof(error_msg),
                     "Failed to allocate %zu bytes for tensor data on device %s", alloc_size,
                     device_get_name(device));
            error_stack_push(CM_MEMORY_ALLOCATION_ERROR, error_msg, __FILE__, __LINE__, __func__);
            free(t->strides);
            free(t->shape);
            free(t);
            return NULL;
        }

        if (cml_backend_buffer_init_tensor(buffer, t) != 0) {
            LOG_ERROR("Backend buffer initialization failed for device %s",
                      device_get_name(device));
            cml_backend_buffer_free(buffer);
            free(t->strides);
            free(t->shape);
            free(t);
            return NULL;
        }

        t->buffer_handle = buffer;
    } else {
        size_t data_size = t->numel * cml_dtype_size(dtype);
        t->data          = malloc(data_size);
        if (!t->data) {
            LOG_ERROR("Failed to allocate fallback memory for tensor data");
            char error_msg[256];
            snprintf(error_msg, sizeof(error_msg),
                     "Failed to allocate %zu bytes for tensor data (fallback)", data_size);
            error_stack_push(CM_MEMORY_ALLOCATION_ERROR, error_msg, __FILE__, __LINE__, __func__);
            free(t->strides);
            free(t->shape);
            free(t);
            return NULL;
        }
    }

    t->grad          = NULL;
    t->requires_grad = false;
    t->ref_count     = 1;
    t->base          = NULL;

    return t;
}

Tensor* tensor_from_data(const void* data, int* shape, int ndim, const TensorConfig* config) {
    if (!data) {
        error_stack_push(CM_INVALID_ARGUMENT, "Invalid argument to tensor_from_data: data is NULL",
                         __FILE__, __LINE__, __func__);
        return NULL;
    }

    Tensor* t = tensor_empty(shape, ndim, config);
    if (!t)
        return NULL;

    DType dtype;
    DeviceType device;
    resolve_config(config, &dtype, &device);

    // Use device-specific copy (assumes source data is on CPU)
    size_t data_size = t->numel * cml_dtype_size(dtype);
    int result       = device_copy_to_device(t->data, data, data_size, device);
    if (result != 0) {
        LOG_ERROR("Failed to copy data to device %s", device_get_name(device));
        char error_msg[256];
        snprintf(error_msg, sizeof(error_msg), "Failed to copy data to device %s",
                 device_get_name(device));
        error_stack_push(CM_OPERATION_FAILED, error_msg, __FILE__, __LINE__, __func__);
        tensor_free(t);
        return NULL;
    }
    return t;
}

void tensor_free(Tensor* t) {
    if (!t)
        return;

    t->ref_count--;
    if (t->ref_count > 0)
        return;

    // CRITICAL: Clear IR node's output pointer BEFORE freeing the tensor
    // This prevents dangling pointers when cml_ir_free is called later
    if (t->ir_node) {
        struct IRNode* node = (struct IRNode*)t->ir_node;
        if (node->output == t) {
            node->output = NULL;
        }
        t->ir_node    = NULL;
        t->ir_context = NULL;
    }

    if (t->owns_data && t->data) {
        if (t->buffer_handle) {
            cml_backend_buffer_free(t->buffer_handle);
            t->buffer_handle = NULL;
        } else {
            if (t->device == DEVICE_CPU || t->device == DEVICE_AUTO) {
                free(t->data);
            } else {
                device_free(t->data, t->device);
            }
        }
    }

    if (t->shape)
        free(t->shape);
    if (t->strides)
        free(t->strides);

    if (t->grad) {
        tensor_free(t->grad);
        t->grad = NULL;
    }

    free(t);
}

Tensor* tensor_clone(Tensor* t) {
    if (!t)
        return NULL;

    TensorConfig config = (TensorConfig){t->dtype, t->device, true, true};
    Tensor* clone       = tensor_empty(t->shape, t->ndim, &config);
    if (!clone)
        return NULL;

    if (t->is_contiguous) {
        memcpy(clone->data, (char*)t->data + t->storage_offset * cml_dtype_size(t->dtype),
               t->numel * cml_dtype_size(t->dtype));
    } else {
        float* clone_data = (float*)clone->data;
        for (size_t i = 0; i < t->numel; i++) {
            clone_data[i] = tensor_get_float(t, i);
        }
    }

    clone->requires_grad = t->requires_grad;
    return clone;
}

Tensor* tensor_from_flat(const float* data, int rows, int cols) {
    if (!data || rows <= 0 || cols <= 0) {
        return NULL;
    }

    int shape[2] = {rows, cols};

    DType dtype         = cml_get_default_dtype();
    DeviceType device   = cml_get_default_device();
    TensorConfig config = (TensorConfig){dtype, device, true, true};

    Tensor* tensor = tensor_empty(shape, 2, &config);
    if (!tensor) {
        return NULL;
    }

    // Copy data
    float* tensor_data = (float*)tensor_data_ptr(tensor);
    if (!tensor_data) {
        tensor_free(tensor);
        return NULL;
    }

    memcpy(tensor_data, data, (size_t)(rows * cols) * sizeof(float));
    return tensor;
}

Tensor* tensor_from_array_2d(const float* data, int rows, int cols) {
    return tensor_from_flat(data, rows, cols);
}

Tensor* tensor_zeros_2d(int rows, int cols) {
    int shape[2] = {rows, cols};

    DType dtype         = cml_get_default_dtype();
    DeviceType device   = cml_get_default_device();
    TensorConfig config = (TensorConfig){dtype, device, true, true};
    return tensor_zeros(shape, 2, &config);
}

Tensor* tensor_ones_2d(int rows, int cols) {
    int shape[2] = {rows, cols};

    DType dtype         = cml_get_default_dtype();
    DeviceType device   = cml_get_default_device();
    TensorConfig config = (TensorConfig){dtype, device, true, true};
    return tensor_ones(shape, 2, &config);
}

Tensor* tensor_empty_2d(int rows, int cols) {
    int shape[2] = {rows, cols};

    DType dtype         = cml_get_default_dtype();
    DeviceType device   = cml_get_default_device();
    TensorConfig config = (TensorConfig){dtype, device, true, true};
    return tensor_empty(shape, 2, &config);
}

int* tensor_shape(int ndim, ...) {
    if (ndim <= 0) {
        return NULL;
    }

    int* shape = (int*)malloc((size_t)ndim * sizeof(int));
    if (!shape) {
        return NULL;
    }

    va_list args;
    va_start(args, ndim);

    for (int i = 0; i < ndim; i++) {
        shape[i] = va_arg(args, int);
    }

    va_end(args);
    return shape;
}

int tensor_to_device(Tensor* tensor, DeviceType device) {
    if (!tensor) {
        return -1;
    }
    return device_move_tensor(tensor, device);
}
