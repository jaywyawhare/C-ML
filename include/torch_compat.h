#ifndef CML_TORCH_COMPAT_H
#define CML_TORCH_COMPAT_H

/*
 * Legacy macro aliases for the PyTorch-like C API.
 * Prefer including "torch/torch_c.h" directly for new code.
 */
#include "torch/torch_c.h"

/* Backward-compatible macro aliases (deprecated — use torch_* functions). */
#define torch_tensor_create    tensor_empty
#define torch_from_numpy       tensor_from_data
#define torch_where            tensor_where
#define torch_einsum           tensor_einsum

#define torch_nn_Conv2d        cml_nn_conv2d
#define torch_nn_BatchNorm2d   cml_nn_batchnorm2d
#define torch_nn_LayerNorm     cml_nn_layernorm
#define torch_nn_Dropout       cml_nn_dropout
#define torch_nn_ReLU          cml_nn_relu
#define torch_nn_GELU          nn_gelu
#define torch_nn_Embedding     cml_nn_embedding
#define torch_nn_Sequential    cml_nn_sequential

#define torch_optim_Adam       cml_optim_adam
#define torch_optim_SGD        cml_optim_sgd
#define torch_optim_AdamW      cml_optim_adamw

/* Deprecated CUDA-only alias. Prefer torch_cuda_device_count() from torch_c.h. */
#define torch_device_count     device_cuda_get_count

#endif /* CML_TORCH_COMPAT_H */
