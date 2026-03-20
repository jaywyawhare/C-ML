#ifndef CML_ZOO_CONVNEXT_H
#define CML_ZOO_CONVNEXT_H

#include "nn.h"
#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int dims[4];
    int depths[4];
} ConvNeXtConfig;

ConvNeXtConfig cml_zoo_convnext_config_tiny(void);
ConvNeXtConfig cml_zoo_convnext_config_small(void);
ConvNeXtConfig cml_zoo_convnext_config_base(void);
ConvNeXtConfig cml_zoo_convnext_config_large(void);

Module* cml_zoo_convnext_create(const ConvNeXtConfig* cfg, int num_classes,
                                 DType dtype, DeviceType device);

#ifdef __cplusplus
}
#endif

#endif /* CML_ZOO_CONVNEXT_H */
