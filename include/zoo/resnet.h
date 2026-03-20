#ifndef CML_ZOO_RESNET_H
#define CML_ZOO_RESNET_H

#include "nn.h"
#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

Module* cml_zoo_resnet50_create(int num_classes, DType dtype, DeviceType device);
Module* cml_zoo_resnet34_create(int num_classes, DType dtype, DeviceType device);
Module* cml_zoo_resnet18_create(int num_classes, DType dtype, DeviceType device);

#ifdef __cplusplus
}
#endif

#endif /* CML_ZOO_RESNET_H */
