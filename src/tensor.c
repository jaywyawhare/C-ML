#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tensor/tensor.h"
#include "Core/logging.h"
#include "Core/memory_management.h"

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

Tensor* tensor_empty(int* shape, int ndim, DType dtype, DeviceType device) {
    if (!shape || ndim < 0)
        return NULL;

    Tensor* t = CM_MALLOC(sizeof(Tensor));
    if (!t)
        return NULL;

    t->shape = tensor_shape_copy(shape, ndim);
    if (!t->shape) {
        LOG_ERROR("Failed to allocate memory for tensor shape copy in tensor_empty");
        CM_FREE(t);
        return NULL;
    }

    t->strides = compute_contiguous_strides(shape, ndim);
    if (!t->strides) {
        LOG_ERROR("Failed to allocate memory for tensor strides in tensor_empty");
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
    t->data          = CM_MALLOC(data_size);
    if (!t->data) {
        LOG_ERROR("Failed to allocate memory for tensor data in tensor_empty");
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

Tensor* tensor_zeros(int* shape, int ndim, DType dtype, DeviceType device) {
    Tensor* t = tensor_empty(shape, ndim, dtype, device);
    if (!t)
        return NULL;

    memset(t->data, 0, t->numel * dtype_size(dtype));
    return t;
}

Tensor* tensor_ones(int* shape, int ndim, DType dtype, DeviceType device) {
    Tensor* t = tensor_empty(shape, ndim, dtype, device);
    if (!t)
        return NULL;

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

Tensor* tensor_from_data(void* data, int* shape, int ndim, DType dtype, DeviceType device) {
    if (!data)
        return NULL;

    Tensor* t = tensor_empty(shape, ndim, dtype, device);
    if (!t)
        return NULL;

    memcpy(t->data, data, t->numel * dtype_size(dtype));
    return t;
}

void tensor_free(Tensor* t) {
    if (!t)
        return;

    t->ref_count--;
    if (t->ref_count > 0)
        return;

    if (t->owns_data && t->data) {
        CM_FREE(t->data);
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

    Tensor* clone = tensor_empty(t->shape, t->ndim, t->dtype, t->device);
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
