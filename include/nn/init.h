#ifndef CML_NN_INIT_H
#define CML_NN_INIT_H

#include "tensor/tensor.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void nn_init_uniform(Tensor* tensor, float low, float high);
void nn_init_xavier(Tensor* tensor, int fan_in, int fan_out);
void nn_init_kaiming(Tensor* tensor, int fan_in, int kernel_volume);

#ifdef __cplusplus
}
#endif

#endif /* CML_NN_INIT_H */
