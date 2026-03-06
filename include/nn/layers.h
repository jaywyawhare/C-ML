/**
 * @file layers.h
 * @brief Unified header for all neural network layers
 *
 * This header provides access to all neural network layers in C-ML.
 * Include this header to get access to all layer types.
 *
 * Usage:
 *   #include "nn/layers.h"
 *
 *   Linear *fc = nn_linear(10, 20, DTYPE_FLOAT32, DEVICE_CPU, true);
 *   ReLU *relu = nn_relu(false);
 *   Dropout *dropout = nn_dropout(0.5, false);
 */

#ifndef CML_NN_LAYERS_H
#define CML_NN_LAYERS_H

// Include all layer headers
#include "nn/layers/linear.h"
#include "nn/layers/activations.h"
#include "nn/layers/conv2d.h"
#include "nn/layers/dropout.h"
#include "nn/layers/batchnorm2d.h"
#include "nn/layers/layernorm.h"
#include "nn/layers/pooling.h"
#include "nn/layers/sequential.h"
#include "nn/layers/conv1d.h"
#include "nn/layers/conv3d.h"
#include "nn/layers/embedding.h"
#include "nn/layers/groupnorm.h"
#include "nn/layers/rnn.h"
#include "nn/layers/transformer.h"
#include "nn/layers/containers.h"
#include "nn/layers/rmsnorm.h"
#include "nn/layers/conv_transpose2d.h"
#include "nn/layers/instancenorm.h"
#include "nn/layers/conv_transpose1d.h"
#include "nn/layers/batchnorm3d.h"
#include "nn/layers/layernorm2d.h"

#ifdef __cplusplus
extern "C" {
#endif

// Convenience macros
#define nn_Linear(in, out, dtype, device, bias) nn_linear(in, out, dtype, device, bias)

#define nn_ReLU(inplace) nn_relu(inplace)

#define nn_LeakyReLU(neg_slope, inplace) nn_leaky_relu(neg_slope, inplace)

#define nn_Sigmoid() nn_sigmoid()

#define nn_Tanh() nn_tanh()

#define nn_Dropout(p, inplace) nn_dropout(p, inplace)

#define nn_Conv2d(in_ch, out_ch, kernel, stride, padding, dilation, bias, dtype, device)           \
    nn_conv2d(in_ch, out_ch, kernel, stride, padding, dilation, bias, dtype, device)

#define nn_BatchNorm2d(num_feat, eps, momentum, affine, track, dtype, device)                      \
    nn_batchnorm2d(num_feat, eps, momentum, affine, track, dtype, device)

#define nn_LayerNorm(normalized_shape, eps, affine, dtype, device)                                 \
    nn_layernorm(normalized_shape, eps, affine, dtype, device)

#define nn_MaxPool2d(kernel, stride, padding, dilation, ceil_mode)                                 \
    nn_maxpool2d(kernel, stride, padding, dilation, ceil_mode)

#define nn_AvgPool2d(kernel, stride, padding, ceil_mode, count_pad)                                \
    nn_avgpool2d(kernel, stride, padding, ceil_mode, count_pad)

#define nn_Sequential() nn_sequential()

#define nn_Conv1d(...) nn_conv1d(__VA_ARGS__)
#define nn_Conv3d(...) nn_conv3d(__VA_ARGS__)
#define nn_Embedding(...) nn_embedding(__VA_ARGS__)
#define nn_GroupNorm(...) nn_groupnorm(__VA_ARGS__)
#define nn_RNNCell(...) nn_rnn_cell(__VA_ARGS__)
#define nn_LSTMCell(...) nn_lstm_cell(__VA_ARGS__)
#define nn_GRUCell(...) nn_gru_cell(__VA_ARGS__)
#define nn_MultiHeadAttention(...) nn_multihead_attention(__VA_ARGS__)
#define nn_TransformerEncoderLayer(...) nn_transformer_encoder_layer(__VA_ARGS__)
#define nn_ModuleList() nn_module_list()
#define nn_ModuleDict() nn_module_dict()
#define nn_RMSNorm(normalized_shape, eps, dtype, device)                                           \
    nn_rmsnorm(normalized_shape, eps, dtype, device)
#define nn_ConvTranspose2d(in_ch, out_ch, kernel, stride, padding, opad, bias, dtype, device)      \
    nn_conv_transpose2d(in_ch, out_ch, kernel, stride, padding, opad, bias, dtype, device)

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_H
