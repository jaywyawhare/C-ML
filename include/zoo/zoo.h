#ifndef CML_ZOO_H
#define CML_ZOO_H

#include "nn.h"
#include "tensor/tensor.h"
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
    CML_ZOO_BERT_TINY,
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

Module* cml_zoo_bert_tiny(const CMLZooConfig* config);

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
