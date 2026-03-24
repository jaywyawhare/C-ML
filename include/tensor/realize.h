

#ifndef CML_TENSOR_REALIZE_H
#define CML_TENSOR_REALIZE_H

#include "tensor/tensor.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

int tensor_realize(Tensor* t);

int tensor_realize_all(Tensor** tensors, int num_tensors);

bool tensor_is_realized(const Tensor* t);

void tensor_unrealize(Tensor* t);

int tensor_realize_with_grads(Tensor* t);

int tensor_schedule(Tensor* t);

int tensor_sync(Tensor* t);

#ifdef __cplusplus
}
#endif

#endif 
