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

Tensor* tensor_permute(Tensor* t, int* dims, int ndim);
Tensor* tensor_expand(Tensor* t, int* new_shape, int new_ndim);
Tensor* tensor_narrow(Tensor* t, int dim, int start, int length);
Tensor* tensor_index_select(Tensor* t, int dim, Tensor* index);
Tensor* tensor_unfold(Tensor* t, int dim, int size, int step);
Tensor* tensor_movedim(Tensor* t, int source, int destination);
Tensor* tensor_swapaxes(Tensor* t, int axis0, int axis1);

#ifdef __cplusplus
}
#endif

#endif // CML_TENSOR_VIEWS_H
