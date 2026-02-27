/**
 * @file zoo.h
 * @brief CML Model Zoo - Pre-built architectures and pretrained weights
 *
 * Provides convenience constructors for common architectures:
 * - MLP (MNIST, CIFAR-10)
 * - ResNet (18/34/50)
 * - VGG (11/16)
 * - GPT-2 small
 * - BERT-tiny
 *
 * Weights can be downloaded from a remote server and cached locally.
 */

#ifndef CML_ZOO_H
#define CML_ZOO_H

#include "nn.h"
#include "tensor/tensor.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Model zoo model identifier
 */
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

/**
 * @brief Model zoo configuration
 */
typedef struct {
    bool pretrained;        /* Load pretrained weights */
    int num_classes;        /* Output classes (0 = default) */
    DType dtype;            /* Data type */
    DeviceType device;      /* Device */
    const char* weights_dir;/* Weights directory (NULL = ~/.cml/weights/) */
} CMLZooConfig;

/**
 * @brief Create default zoo configuration
 */
CMLZooConfig cml_zoo_default_config(void);

/**
 * @brief Create MLP for MNIST (784 -> 256 -> 128 -> 10)
 */
Module* cml_zoo_mlp_mnist(const CMLZooConfig* config);

/**
 * @brief Create MLP for CIFAR-10 (3072 -> 512 -> 256 -> 10)
 */
Module* cml_zoo_mlp_cifar10(const CMLZooConfig* config);

/**
 * @brief Create ResNet-18
 */
Module* cml_zoo_resnet18(const CMLZooConfig* config);

/**
 * @brief Create ResNet-34
 */
Module* cml_zoo_resnet34(const CMLZooConfig* config);

/**
 * @brief Create ResNet-50
 */
Module* cml_zoo_resnet50(const CMLZooConfig* config);

/**
 * @brief Create VGG-11
 */
Module* cml_zoo_vgg11(const CMLZooConfig* config);

/**
 * @brief Create VGG-16
 */
Module* cml_zoo_vgg16(const CMLZooConfig* config);

/**
 * @brief Create GPT-2 small (124M params)
 */
Module* cml_zoo_gpt2_small(const CMLZooConfig* config);

/**
 * @brief Create BERT-tiny (4M params)
 */
Module* cml_zoo_bert_tiny(const CMLZooConfig* config);

/**
 * @brief Create a model from the zoo by enum
 */
Module* cml_zoo_create(CMLZooModel model, const CMLZooConfig* config);

/**
 * @brief Get model name string
 */
const char* cml_zoo_model_name(CMLZooModel model);

/**
 * @brief Download pretrained weights for a model
 *
 * Downloads from the CML weight server, verifies SHA256, stores in weights_dir.
 * Uses libcurl (via dlopen) or falls back to system curl/wget.
 *
 * @param model Model identifier
 * @param weights_dir Directory to store weights (NULL = ~/.cml/weights/)
 * @return Path to downloaded weights file, or NULL on failure
 */
const char* cml_zoo_download_weights(CMLZooModel model, const char* weights_dir);

/**
 * @brief Load pretrained weights into a model
 *
 * @param module Model to load weights into
 * @param weights_path Path to weights file
 * @return 0 on success, -1 on failure
 */
int cml_zoo_load_weights(Module* module, const char* weights_path);

/**
 * @brief Set the weights base URL
 */
void cml_zoo_set_weights_url(const char* base_url);

/**
 * @brief Get default weights directory
 */
const char* cml_zoo_get_weights_dir(void);

#ifdef __cplusplus
}
#endif

#endif /* CML_ZOO_H */
