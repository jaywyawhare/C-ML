#ifndef CML_ZOO_RNNT_H
#define CML_ZOO_RNNT_H

#include "nn.h"
#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int input_features;
    int encoder_layers;
    int encoder_dim;
    int pred_layers;
    int pred_dim;
    int joint_dim;
    int vocab_size;
    int subsample_stride;
    int subsample_kernel;
} CMLRNNTConfig;

typedef struct {
    Module base;
    CMLRNNTConfig config;
    DType dtype;
    DeviceType device;

    Module*  subsample_conv;
    Module** encoder_lstms;
    Module** encoder_lnorms;
    int      num_enc_layers;

    Module*  pred_embedding;
    Module** pred_lstms;
    Module** pred_lnorms;
    int      num_pred_layers;

    Module*  joint_linear1;
    Module*  joint_relu;
    Module*  joint_linear2;
} CMLRNNT;

Module* cml_zoo_rnnt_create(const CMLRNNTConfig* config, DType dtype, DeviceType device);

CMLRNNTConfig cml_zoo_rnnt_config_default(void);

Tensor* cml_rnnt_encode(Module* module, Tensor* audio_features);
Tensor* cml_rnnt_predict(Module* module, Tensor* prev_tokens);
Tensor* cml_rnnt_joint(Module* module, Tensor* enc_out, Tensor* pred_out);

#ifdef __cplusplus
}
#endif

#endif /* CML_ZOO_RNNT_H */
