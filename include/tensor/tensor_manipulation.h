#ifndef CML_TENSOR_MANIPULATION_H
#define CML_TENSOR_MANIPULATION_H

#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

Tensor* tensor_concat(Tensor** tensors, int num_tensors, int dim);
Tensor* tensor_stack(Tensor** tensors, int num_tensors, int dim);
Tensor** tensor_split(Tensor* tensor, int num_splits, int dim, int* split_sizes);
Tensor* tensor_gather(Tensor* input, Tensor* indices, int dim);
Tensor* tensor_scatter(Tensor* input, int dim, Tensor* index, Tensor* src);

#ifdef __cplusplus
}
#endif

#endif // CML_TENSOR_MANIPULATION_H
