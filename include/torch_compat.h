#ifndef CML_TORCH_COMPAT_H
#define CML_TORCH_COMPAT_H

#include "cml.h"

/* Tensor creation */
#define torch_tensor_create    tensor_empty
#define torch_zeros            tensor_zeros
#define torch_ones             tensor_ones
#define torch_randn            tensor_randn
#define torch_rand             tensor_rand
#define torch_eye              tensor_eye
#define torch_arange           tensor_arange
#define torch_linspace         tensor_linspace
#define torch_from_numpy       tensor_from_data
#define torch_cat              uop_cat
#define torch_stack            uop_stack

/* Element-wise / reduction ops */
#define torch_matmul           uop_matmul
#define torch_add              uop_add
#define torch_sub              uop_sub
#define torch_mul              uop_mul
#define torch_div              uop_div
#define torch_sum              uop_sum
#define torch_mean             uop_mean
#define torch_max              uop_max_reduce
#define torch_relu             uop_relu
#define torch_sigmoid          uop_sigmoid
#define torch_tanh             uop_tanh
#define torch_softmax          cml_softmax
#define torch_where            tensor_where
#define torch_einsum           tensor_einsum

/* Module constructors */
#define torch_nn_Linear        cml_nn_linear
#define torch_nn_Conv2d        cml_nn_conv2d
#define torch_nn_BatchNorm2d   cml_nn_batchnorm2d
#define torch_nn_LayerNorm     cml_nn_layernorm
#define torch_nn_Dropout       cml_nn_dropout
#define torch_nn_ReLU          cml_nn_relu
#define torch_nn_GELU          nn_gelu
#define torch_nn_Embedding     cml_nn_embedding
#define torch_nn_Sequential    cml_nn_sequential

/* Optimizers */
#define torch_optim_Adam       cml_optim_adam
#define torch_optim_SGD        cml_optim_sgd
#define torch_optim_AdamW      cml_optim_adamw

/* Autograd */
#define torch_backward         cml_backward
#define torch_no_grad          autograd_no_grad_enter
#define torch_enable_grad      autograd_no_grad_exit

/* Device queries */
#define torch_cuda_is_available  cml_cuda_available
#define torch_device_count       device_cuda_get_count

#endif /* CML_TORCH_COMPAT_H */
