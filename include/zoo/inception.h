#ifndef CML_ZOO_INCEPTION_H
#define CML_ZOO_INCEPTION_H

#include "nn.h"
#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

Module* cml_zoo_inception_v3_create(int num_classes, DType dtype, DeviceType device);

#ifdef __cplusplus
}
#endif

#endif /* CML_ZOO_INCEPTION_H */
