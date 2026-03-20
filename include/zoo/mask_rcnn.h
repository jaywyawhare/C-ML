#ifndef CML_ZOO_MASK_RCNN_H
#define CML_ZOO_MASK_RCNN_H

#include "nn.h"
#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int num_classes;
    int num_anchors;
    int fpn_channels;
    int roi_output_size;
    int mask_output_size;
} MaskRCNNConfig;

MaskRCNNConfig cml_zoo_mask_rcnn_default_config(void);

Module* cml_zoo_mask_rcnn_create(const MaskRCNNConfig* cfg, DType dtype, DeviceType device);

#ifdef __cplusplus
}
#endif

#endif /* CML_ZOO_MASK_RCNN_H */
