#ifndef CML_TENSOR_COMPARE_H
#define CML_TENSOR_COMPARE_H

#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

Tensor* tensor_lt(Tensor* a, Tensor* b);
Tensor* tensor_le(Tensor* a, Tensor* b);
Tensor* tensor_gt(Tensor* a, Tensor* b);
Tensor* tensor_ge(Tensor* a, Tensor* b);
Tensor* tensor_eq(Tensor* a, Tensor* b);
Tensor* tensor_ne(Tensor* a, Tensor* b);

Tensor* tensor_logical_and(Tensor* a, Tensor* b);
Tensor* tensor_logical_or(Tensor* a, Tensor* b);
Tensor* tensor_logical_not(Tensor* a);
Tensor* tensor_logical_xor(Tensor* a, Tensor* b);

Tensor* tensor_bitwise_and(Tensor* a, Tensor* b);
Tensor* tensor_bitwise_or(Tensor* a, Tensor* b);
Tensor* tensor_bitwise_xor(Tensor* a, Tensor* b);
Tensor* tensor_bitwise_not(Tensor* a);

Tensor* tensor_minimum(Tensor* a, Tensor* b);
Tensor* tensor_maximum(Tensor* a, Tensor* b);

#ifdef __cplusplus
}
#endif

#endif // CML_TENSOR_COMPARE_H
