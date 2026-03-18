#ifndef CML_TENSOR_VIEWS_H
#define CML_TENSOR_VIEWS_H

#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

Tensor* tensor_view(Tensor* t, int* new_shape, int new_ndim);
Tensor* tensor_as_strided(Tensor* t, int* shape, int ndim, size_t* strides, size_t storage_offset);
Tensor* tensor_reshape(Tensor* t, int* new_shape, int new_ndim);
Tensor* tensor_contiguous(Tensor* t);

#ifdef __cplusplus
}
#endif

#endif // CML_TENSOR_VIEWS_H
