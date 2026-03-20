#ifndef CML_ZOO_H
#define CML_ZOO_H

#include "nn.h"
#include "tensor/tensor.h"
#include "zoo/resnet.h"
#include "zoo/gpt2.h"
#include "zoo/bert.h"
#include "zoo/vit.h"
#include "zoo/clip.h"
#include "zoo/t5.h"
#include "zoo/unet.h"
#include "zoo/unet3d.h"
#include "zoo/rnnt.h"
#include "zoo/convnext.h"
#include "zoo/inception.h"
#include "zoo/retinanet.h"
#include "zoo/mask_rcnn.h"
#include "zoo/efficientnet.h"
#include "zoo/whisper.h"
#include "zoo/stable_diffusion.h"
#include "zoo/yolov8.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CML_ZOO_MLP_MNIST = 0,
    CML_ZOO_MLP_CIFAR10,
    CML_ZOO_RESNET18,
    CML_ZOO_RESNET34,
    CML_ZOO_RESNET50,
    CML_ZOO_VGG11,
    CML_ZOO_VGG16,
    CML_ZOO_GPT2_SMALL,
    CML_ZOO_GPT2_MEDIUM,
    CML_ZOO_GPT2_LARGE,
    CML_ZOO_GPT2_XL,
    CML_ZOO_BERT_TINY,
    CML_ZOO_BERT_MINI,
    CML_ZOO_BERT_SMALL,
    CML_ZOO_BERT_BASE,
    CML_ZOO_BERT_LARGE,
    CML_ZOO_VIT_TINY,
    CML_ZOO_VIT_SMALL,
    CML_ZOO_VIT_BASE,
    CML_ZOO_VIT_LARGE,
    CML_ZOO_CLIP_VIT_B32,
    CML_ZOO_CLIP_VIT_B16,
    CML_ZOO_CLIP_VIT_L14,
    CML_ZOO_T5_SMALL,
    CML_ZOO_T5_BASE,
    CML_ZOO_T5_LARGE,
    CML_ZOO_UNET_DEFAULT,
    CML_ZOO_CONVNEXT_TINY,
    CML_ZOO_CONVNEXT_SMALL,
    CML_ZOO_CONVNEXT_BASE,
    CML_ZOO_CONVNEXT_LARGE,
    CML_ZOO_INCEPTION_V3,
    CML_ZOO_RETINANET,
    CML_ZOO_MASK_RCNN,
    CML_ZOO_UNET3D,
    CML_ZOO_RNNT,
    CML_ZOO_MODEL_COUNT
} CMLZooModel;

typedef struct {
    bool pretrained;        /* Load pretrained weights */
    int num_classes;        /* Output classes (0 = default) */
    DType dtype;            /* Data type */
    DeviceType device;      /* Device */
    const char* weights_dir;/* Weights directory (NULL = ~/.cml/weights/) */
} CMLZooConfig;

CMLZooConfig cml_zoo_default_config(void);

Module* cml_zoo_mlp_mnist(const CMLZooConfig* config);

Module* cml_zoo_mlp_cifar10(const CMLZooConfig* config);

Module* cml_zoo_resnet18(const CMLZooConfig* config);

Module* cml_zoo_resnet34(const CMLZooConfig* config);

Module* cml_zoo_resnet50(const CMLZooConfig* config);

Module* cml_zoo_vgg11(const CMLZooConfig* config);

Module* cml_zoo_vgg16(const CMLZooConfig* config);

Module* cml_zoo_gpt2_small(const CMLZooConfig* config);

Module* cml_zoo_gpt2_medium(const CMLZooConfig* config);

Module* cml_zoo_gpt2_large(const CMLZooConfig* config);

Module* cml_zoo_gpt2_xl(const CMLZooConfig* config);

Module* cml_zoo_bert_tiny(const CMLZooConfig* config);

Module* cml_zoo_bert_mini(const CMLZooConfig* config);

Module* cml_zoo_bert_small(const CMLZooConfig* config);

Module* cml_zoo_bert_base(const CMLZooConfig* config);

Module* cml_zoo_bert_large(const CMLZooConfig* config);

Module* cml_zoo_vit_tiny(const CMLZooConfig* config);

Module* cml_zoo_vit_small(const CMLZooConfig* config);

Module* cml_zoo_vit_base(const CMLZooConfig* config);

Module* cml_zoo_vit_large(const CMLZooConfig* config);

Module* cml_zoo_clip_vit_b32(const CMLZooConfig* config);

Module* cml_zoo_clip_vit_b16(const CMLZooConfig* config);

Module* cml_zoo_clip_vit_l14(const CMLZooConfig* config);

Module* cml_zoo_t5_small(const CMLZooConfig* config);

Module* cml_zoo_t5_base(const CMLZooConfig* config);

Module* cml_zoo_t5_large(const CMLZooConfig* config);

Module* cml_zoo_unet_default(const CMLZooConfig* config);

Module* cml_zoo_convnext_tiny(const CMLZooConfig* config);

Module* cml_zoo_convnext_small(const CMLZooConfig* config);

Module* cml_zoo_convnext_base(const CMLZooConfig* config);

Module* cml_zoo_convnext_large(const CMLZooConfig* config);

Module* cml_zoo_inception_v3(const CMLZooConfig* config);

Module* cml_zoo_retinanet(const CMLZooConfig* config);

Module* cml_zoo_mask_rcnn(const CMLZooConfig* config);

Module* cml_zoo_unet3d(const CMLZooConfig* config);

Module* cml_zoo_rnnt(const CMLZooConfig* config);

Module* cml_zoo_create(CMLZooModel model, const CMLZooConfig* config);

const char* cml_zoo_model_name(CMLZooModel model);

const char* cml_zoo_download_weights(CMLZooModel model, const char* weights_dir);

int cml_zoo_load_weights(Module* module, const char* weights_path);

void cml_zoo_set_weights_url(const char* base_url);

const char* cml_zoo_get_weights_dir(void);

#ifdef __cplusplus
}
#endif

#endif /* CML_ZOO_H */
