#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tensor/tensor.h"
#include "Core/logging.h"
#include "Core/memory_management.h"
#include "Core/device.h"
#include "Core/error_stack.h"

// ============================================================================
// Tensor Configuration Implementation
// ============================================================================

TensorConfig tensor_config_default(void) {
    TensorConfig config = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_AUTO, .has_dtype = false, .has_device = false};
    return config;
}

TensorConfig tensor_config_with_dtype(DType dtype) {
    TensorConfig config = {
        .dtype = dtype, .device = DEVICE_AUTO, .has_dtype = true, .has_device = false};
    return config;
}

TensorConfig tensor_config_with_device(DeviceType device) {
    TensorConfig config = {
        .dtype = DTYPE_FLOAT32, .device = device, .has_dtype = false, .has_device = true};
    return config;
}

TensorConfig tensor_config_with_dtype_device(DType dtype, DeviceType device) {
    TensorConfig config = {.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
    return config;
}

// Helper function to resolve config to actual values
static void resolve_config(const TensorConfig* config, DType* dtype, DeviceType* device) {
    if (config && config->has_dtype) {
        *dtype = config->dtype;
    } else {
        *dtype = DTYPE_FLOAT32; // Default dtype
    }

    if (config && config->has_device) {
        *device = config->device;
    } else {
        *device = device_get_default(); // Auto-detect device
    }

    // Handle DEVICE_AUTO
    if (*device == DEVICE_AUTO) {
        *device = device_get_best_available();
    }
}

size_t dtype_size(DType dtype) {
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

size_t tensor_numel(int* shape, int ndim) {
    size_t numel = 1;
    for (int i = 0; i < ndim; i++) {
        numel *= shape[i];
    }
    return numel;
}

size_t* compute_contiguous_strides(int* shape, int ndim) {
    if (!shape || ndim <= 0)
        return NULL;

    size_t* strides = CM_MALLOC(ndim * sizeof(size_t));
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

bool check_is_contiguous(int* shape, size_t* strides, int ndim) {
    if (!shape || !strides || ndim <= 0)
        return false;

    size_t expected_stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        if (strides[i] != expected_stride)
            return false;
        expected_stride *= shape[i];
    }

    return true;
}

size_t compute_storage_size(int* shape, size_t* strides, int ndim) {
    if (!shape || !strides || ndim <= 0)
        return 0;

    size_t max_offset = 0;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] > 1) {
            max_offset += (shape[i] - 1) * strides[i];
        }
    }

    return max_offset + 1;
}

int* tensor_shape_copy(int* shape, int ndim) {
    int* new_shape = CM_MALLOC(ndim * sizeof(int));
    if (!new_shape) {
        LOG_ERROR("Failed to allocate memory for tensor shape copy");
        return NULL;
    }
    memcpy(new_shape, shape, ndim * sizeof(int));
    return new_shape;
}

float tensor_get_float(Tensor* t, size_t idx) {
    if (!t || !t->data || idx >= t->numel)
        return 0.0f;

    size_t offset = t->storage_offset;
    if (!t->is_contiguous) {
        size_t temp = idx;
        for (int d = t->ndim - 1; d >= 0; d--) {
            size_t coord = temp % t->shape[d];
            temp /= t->shape[d];
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
    if (!t || !t->data || idx >= t->numel)
        return;

    size_t offset = t->storage_offset;
    if (!t->is_contiguous) {
        size_t temp = idx;
        for (int d = t->ndim - 1; d >= 0; d--) {
            size_t coord = temp % t->shape[d];
            temp /= t->shape[d];
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
        ((uint8_t*)t->data)[offset] = (uint8_t)(value != 0);
        break;
    }
}

void* tensor_data_ptr(Tensor* t) { return t ? t->data : NULL; }

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

    Tensor* t = CM_MALLOC(sizeof(Tensor));
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
        CM_FREE(t);
        return NULL;
    }

    t->strides = compute_contiguous_strides(shape, ndim);
    if (!t->strides) {
        LOG_ERROR("Failed to allocate memory for tensor strides in tensor_empty");
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR, "Failed to allocate memory for tensor strides",
                         __FILE__, __LINE__, __func__);
        CM_FREE(t->shape);
        CM_FREE(t);
        return NULL;
    }

    t->ndim           = ndim;
    t->numel          = tensor_numel(shape, ndim);
    t->dtype          = dtype;
    t->device         = device;
    t->storage_offset = 0;

    t->is_contiguous = true;
    t->owns_data     = true;

    size_t data_size = t->numel * dtype_size(dtype);
    // Use device-specific allocation
    t->data = device_alloc(data_size, device);
    if (!t->data) {
        LOG_ERROR("Failed to allocate memory for tensor data on device %s",
                  device_get_name(device));
        char error_msg[256];
        snprintf(error_msg, sizeof(error_msg),
                 "Failed to allocate %zu bytes for tensor data on device %s", data_size,
                 device_get_name(device));
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR, error_msg, __FILE__, __LINE__, __func__);
        CM_FREE(t->strides);
        CM_FREE(t->shape);
        CM_FREE(t);
        return NULL;
    }

    t->grad          = NULL;
    t->grad_fn       = NULL;
    t->requires_grad = false;
    t->ref_count     = 1;
    t->base          = NULL;

    return t;
}

Tensor* tensor_zeros(int* shape, int ndim, const TensorConfig* config) {
    Tensor* t = tensor_empty(shape, ndim, config);
    if (!t)
        return NULL;

    DType dtype;
    DeviceType device;
    resolve_config(config, &dtype, &device);
    memset(t->data, 0, t->numel * dtype_size(dtype));
    return t;
}

Tensor* tensor_ones(int* shape, int ndim, const TensorConfig* config) {
    Tensor* t = tensor_empty(shape, ndim, config);
    if (!t)
        return NULL;

    DType dtype;
    DeviceType device;
    resolve_config(config, &dtype, &device);

    for (size_t i = 0; i < t->numel; i++) {
        switch (dtype) {
        case DTYPE_FLOAT32:
            ((float*)t->data)[i] = 1.0f;
            break;
        case DTYPE_FLOAT64:
            ((double*)t->data)[i] = 1.0;
            break;
        case DTYPE_INT32:
            ((int32_t*)t->data)[i] = 1;
            break;
        case DTYPE_INT64:
            ((int64_t*)t->data)[i] = 1;
            break;
        case DTYPE_BOOL:
            ((uint8_t*)t->data)[i] = 1;
            break;
        }
    }
    return t;
}

Tensor* tensor_from_data(void* data, int* shape, int ndim, const TensorConfig* config) {
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
    size_t data_size = t->numel * dtype_size(dtype);
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

    if (t->owns_data && t->data) {
        // Use device-specific deallocation
        device_free(t->data, t->device);
    }

    if (t->shape)
        CM_FREE(t->shape);
    if (t->strides)
        CM_FREE(t->strides);

    if (t->grad) {
        tensor_free(t->grad);
        t->grad = NULL;
    }
    if (t->grad_fn) {
        CM_FREE(t->grad_fn);
        t->grad_fn = NULL;
    }

    CM_FREE(t);
}

Tensor* tensor_clone(Tensor* t) {
    if (!t)
        return NULL;

    TensorConfig config = tensor_config_with_dtype_device(t->dtype, t->device);
    Tensor* clone       = tensor_empty(t->shape, t->ndim, &config);
    if (!clone)
        return NULL;

    if (t->is_contiguous) {
        memcpy(clone->data, (char*)t->data + t->storage_offset * dtype_size(t->dtype),
               t->numel * dtype_size(t->dtype));
    } else {
        float* clone_data = (float*)clone->data;
        for (size_t i = 0; i < t->numel; i++) {
            clone_data[i] = tensor_get_float(t, i);
        }
    }

    clone->requires_grad = t->requires_grad;
    return clone;
}
