#ifndef CML_H
#define CML_H

#include "core/export.h"

#include "core/logging.h"
#include "alloc/memory_management.h"
#include "core/error_codes.h"
#include "core/error_stack.h"
#include "core/dataset.h"
#include "core/training_metrics.h"
#include "core/cleanup.h"
#include "core/config.h"
#include "core/quantization.h"
#include "core/training_loop.h"
#include "core/serialization.h"
#include "core/gguf.h"
#include "core/safetensors.h"
#include "backend/profiling.h"
#include "backend/backend_buffer.h"
#include "alloc/graph_allocator.h"
#include "core/computation_graph.h"
#include "core/graph_context.h"
#include "ops/uops.h"
#include "ops/ir/ir.h"
#include "ops/ir/export.h"
#include "ops/ir/optimization.h"
#include "ops/ir/aot.h"
#include "ops/simd_views.h"

#include "ops/ir/fusion_patterns.h"

#include "tensor/tensor.h"
#include "autograd/forward_ops.h"

#include "autograd/autograd.h"
#include "autograd/loss_functions.h"
#include "autograd/amp.h"
#include "autograd/checkpointing.h"
#include "tensor/sparse_tensor.h"

#include "nn.h"
#include "nn/layers.h"

#include "optim.h"
#include "nn/model_io.h"

#ifdef CML_HAS_DISTRIBUTED
#include "distributed/distributed.h"
#include "distributed/data_parallel.h"
#include "distributed/pipeline_parallel.h"
#endif

#include "symbolic/symbolic.h"

#include "ops/ir/schedule.h"
#include "ops/ir/graph_capture.h"
#include "ops/ir/compiler_viz.h"
#include "tensor/image_dtype.h"
#include "backend/null_device.h"
#include "backend/disk_backend.h"
#include "core/pth_loader.h"
#include "core/tinyfs.h"
#include "nn/llm_ops.h"
#include "alloc/tlsf_alloc.h"

#include "zoo/zoo.h"
#include "datasets/datasets.h"
#include "datasets/loaders.h"

void cml_summary(struct Module* module);

void cml_track_module(struct Module* module);
void cml_untrack_module(struct Module* module);
void cml_track_optimizer(struct Optimizer* optimizer);
void cml_track_dataset(struct Dataset* dataset);

Tensor* cml_empty(int* shape, int ndim, const TensorConfig* config);
Tensor* cml_zeros(int* shape, int ndim, const TensorConfig* config);
Tensor* cml_ones(int* shape, int ndim, const TensorConfig* config);
Tensor* cml_full(int* shape, int ndim, const TensorConfig* config, float value);
Tensor* cml_tensor(void* data, int* shape, int ndim, const TensorConfig* config);
Tensor* cml_zeros_2d(int rows, int cols);
Tensor* cml_ones_2d(int rows, int cols);
Tensor* cml_empty_2d(int rows, int cols);
Tensor* cml_tensor_2d(const float* data, int rows, int cols);
Tensor* cml_zeros_1d(int size);
Tensor* cml_ones_1d(int size);
Tensor* cml_empty_1d(int size);
Tensor* cml_tensor_1d(const float* data, int size);

Tensor* cml_add(Tensor* a, Tensor* b);
Tensor* cml_sub(Tensor* a, Tensor* b);
Tensor* cml_mul(Tensor* a, Tensor* b);
Tensor* cml_div(Tensor* a, Tensor* b);
Tensor* cml_exp(Tensor* a);
Tensor* cml_log(Tensor* a);
Tensor* cml_sqrt(Tensor* a);
Tensor* cml_sin(Tensor* a);
Tensor* cml_cos(Tensor* a);
Tensor* cml_tan(Tensor* a);
Tensor* cml_pow(Tensor* a, Tensor* b);
Tensor* cml_sign(Tensor* a);
Tensor* cml_floor(Tensor* a);
Tensor* cml_ceil(Tensor* a);
Tensor* cml_round(Tensor* a);
Tensor* cml_log2(Tensor* a);
Tensor* cml_exp2(Tensor* a);
Tensor* cml_asin(Tensor* a);
Tensor* cml_acos(Tensor* a);
Tensor* cml_atan(Tensor* a);
Tensor* cml_square(Tensor* a);
Tensor* cml_rsqrt(Tensor* a);
Tensor* cml_erf(Tensor* a);
Tensor* cml_clamp(Tensor* a, float min_val, float max_val);

Tensor* cml_relu(Tensor* a);
Tensor* cml_sigmoid(Tensor* a);
Tensor* cml_tanh(Tensor* a);
Tensor* cml_softmax(Tensor* a, int dim);
Tensor* cml_elu(Tensor* x, float alpha);
Tensor* cml_selu(Tensor* x);
Tensor* cml_mish(Tensor* x);
Tensor* cml_silu(Tensor* x);
Tensor* cml_hardswish(Tensor* x);
Tensor* cml_leaky_relu(Tensor* x, float negative_slope);

Tensor* cml_sum(Tensor* a, int dim, bool keepdim);
Tensor* cml_mean(Tensor* a, int dim, bool keepdim);
Tensor* cml_max(Tensor* a, int dim, bool keepdim);
Tensor* cml_min(Tensor* a, int dim, bool keepdim);
Tensor* cml_prod(Tensor* a, int dim, bool keepdim);
Tensor* cml_argmax(Tensor* a, int dim);
Tensor* cml_argmin(Tensor* a, int dim);
Tensor* cml_cumsum(Tensor* a, int dim);
Tensor* cml_var(Tensor* a, int dim, bool unbiased, bool keepdim);
Tensor* cml_std(Tensor* a, int dim, bool unbiased, bool keepdim);

Tensor* cml_matmul(Tensor* a, Tensor* b);
Tensor* cml_dot(Tensor* a, Tensor* b);
QRResult cml_qr(Tensor* a);
SVDResult cml_svd(Tensor* a);

Tensor* cml_transpose(Tensor* a, int dim1, int dim2);
Tensor* cml_reshape(Tensor* a, int* new_shape, int new_ndim);
Tensor* cml_clone(Tensor* a);
Tensor* cml_detach(Tensor* a);
Tensor* cml_concat(Tensor** tensors, int num_tensors, int dim);
Tensor* cml_stack(Tensor** tensors, int num_tensors, int dim);
Tensor* cml_squeeze(Tensor* a, int dim);
Tensor* cml_unsqueeze(Tensor* a, int dim);
Tensor* cml_flip(Tensor* a, int dim);
Tensor* cml_repeat(Tensor* a, int* repeats, int num_repeats);
Tensor** cml_split(Tensor* a, int num_splits, int dim, int* out_count);
Tensor** cml_chunk(Tensor* a, int chunks, int dim, int* out_count);
Tensor* cml_triu(Tensor* a, int diagonal);
Tensor* cml_tril(Tensor* a, int diagonal);
Tensor* cml_pad(Tensor* a, int* pad_widths, int num_dims, float value);
Tensor* cml_pad_reflect(Tensor* a, int* pad_widths, int num_dims);
Tensor* cml_pad_replicate(Tensor* a, int* pad_widths, int num_dims);
Tensor* cml_unfold(Tensor* a, int kernel_size, int stride);
Tensor* cml_sort(Tensor* a, int dim, bool descending);
Tensor* cml_topk(Tensor* a, int k, int dim, bool largest, bool sorted);
Tensor* cml_masked_select(Tensor* a, Tensor* mask);
Tensor** cml_meshgrid(Tensor** tensors, int num_tensors, int* num_outputs);
Tensor* cml_diagonal(Tensor* a, int offset, int dim1, int dim2);
Tensor* cml_lerp(Tensor* a, Tensor* b, float weight);
Tensor* cml_idiv(Tensor* a, Tensor* b);
Tensor* cml_mod(Tensor* a, Tensor* b);
Tensor* cml_contiguous(Tensor* a);
Tensor* cml_scatter_reduce(Tensor* self, int dim, Tensor* index, Tensor* src, ScatterReduceMode mode);
Tensor* cml_interpolate(Tensor* a, int* output_size, int num_dims, InterpMode mode);

Tensor* cml_arange(float start, float end, float step, const TensorConfig* config);
Tensor* cml_linspace(float start, float end, int steps, const TensorConfig* config);
Tensor* cml_eye(int n, const TensorConfig* config);
Tensor* cml_rand(int* shape, int ndim, const TensorConfig* config);
Tensor* cml_randn(int* shape, int ndim, const TensorConfig* config);
Tensor* cml_randint(int low, int high, int* shape, int ndim, const TensorConfig* config);
Tensor* cml_randperm(int n, const TensorConfig* config);
void cml_manual_seed(uint64_t seed);
Tensor* cml_zeros_like(Tensor* a);
Tensor* cml_ones_like(Tensor* a);
Tensor* cml_rand_like(Tensor* a);
Tensor* cml_randn_like(Tensor* a);
Tensor* cml_full_like(Tensor* a, float value);
Tensor* cml_from_blob(void* data, int* shape, int ndim, const TensorConfig* config);

Tensor* cml_cast(Tensor* a, DType dtype);
Tensor* cml_bitcast(Tensor* a, DType target_dtype);
Tensor* cml_half(Tensor* a);
Tensor* cml_double(Tensor* a);
Tensor* cml_int_(Tensor* a);
Tensor* cml_long(Tensor* a);
Tensor* cml_short(Tensor* a);
Tensor* cml_bool_(Tensor* a);
Tensor* cml_bfloat16(Tensor* a);

Tensor* cml_kaiming_uniform(int* shape, int ndim, int fan_in, const TensorConfig* config);
Tensor* cml_kaiming_normal(int* shape, int ndim, int fan_in, const TensorConfig* config);
Tensor* cml_glorot_uniform(int* shape, int ndim, int fan_in, int fan_out, const TensorConfig* config);
Tensor* cml_xavier_normal(int* shape, int ndim, int fan_in, int fan_out, const TensorConfig* config);

Sequential* cml_nn_sequential(void);
Sequential* cml_nn_sequential_add(Sequential* seq, Module* layer);
Tensor* cml_nn_sequential_forward(Sequential* seq, Tensor* input);
Tensor* cml_nn_module_forward(Module* module, Tensor* input);
void cml_nn_module_set_training(Module* module, bool training);
bool cml_nn_module_is_training(Module* module);
void cml_nn_module_eval(Module* module);
void cml_nn_module_train(Module* module);

Linear* cml_nn_linear(int in_features, int out_features, DType dtype, DeviceType device, bool bias);
ReLU* cml_nn_relu(bool inplace);
Sigmoid* cml_nn_sigmoid(void);
Tanh* cml_nn_tanh(void);
LeakyReLU* cml_nn_leaky_relu(float negative_slope, bool inplace);
Dropout* cml_nn_dropout(float p, bool inplace);
Conv1d* cml_nn_conv1d(int in_channels, int out_channels, int kernel_size, int stride, int padding,
                      int dilation, bool use_bias, DType dtype, DeviceType device);
Conv2d* cml_nn_conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding,
                      int dilation, bool bias, DType dtype, DeviceType device);
Conv3d* cml_nn_conv3d(int in_channels, int out_channels, int kernel_size, int stride, int padding,
                      int dilation, bool use_bias, DType dtype, DeviceType device);
ConvTranspose1d* cml_nn_conv_transpose1d(int in_channels, int out_channels, int kernel_size,
                                          int stride, int padding, int output_padding,
                                          bool use_bias, DType dtype, DeviceType device);
ConvTranspose3d* cml_nn_conv_transpose3d(int in_channels, int out_channels, int kernel_size,
                                          int stride, int padding, int output_padding,
                                          bool use_bias, DType dtype, DeviceType device);
BatchNorm1d* cml_nn_batchnorm1d(int num_features, float eps, float momentum, bool affine,
                                 bool track_running_stats, DType dtype, DeviceType device);
BatchNorm2d* cml_nn_batchnorm2d(int num_features, float eps, float momentum, bool affine,
                                bool track_running_stats, DType dtype, DeviceType device);
BatchNorm3d* cml_nn_batchnorm3d(int num_features, float eps, float momentum, bool affine,
                                 bool track_running_stats, DType dtype, DeviceType device);
LayerNorm* cml_nn_layernorm(int normalized_shape, float eps, bool affine, DType dtype,
                            DeviceType device);
LayerNorm2d* cml_nn_layernorm2d(int num_channels, float eps, bool affine, DType dtype,
                                 DeviceType device);
InstanceNorm2d* cml_nn_instancenorm2d(int num_features, float eps, bool affine, DType dtype,
                                       DeviceType device);
GroupNorm* cml_nn_groupnorm(int num_groups, int num_channels, float eps, bool affine, DType dtype,
                            DeviceType device);
Embedding* cml_nn_embedding(int num_embeddings, int embedding_dim, int padding_idx, DType dtype,
                            DeviceType device);
Flatten* cml_nn_flatten(int start_dim, int end_dim);
Identity* cml_nn_identity(void);
PReLU* cml_nn_prelu(int num_parameters, float init, DType dtype, DeviceType device);
ModuleList* cml_nn_module_list(void);
ModuleDict* cml_nn_module_dict(void);

MaxPool1d* cml_nn_maxpool1d(int kernel_size, int stride, int padding, int dilation, bool ceil_mode);
MaxPool2d* cml_nn_maxpool2d(int kernel_size, int stride, int padding, int dilation, bool ceil_mode);
MaxPool3d* cml_nn_maxpool3d(int kernel_size, int stride, int padding, int dilation, bool ceil_mode);
AvgPool1d* cml_nn_avgpool1d(int kernel_size, int stride, int padding, bool ceil_mode,
                            bool count_include_pad);
AvgPool2d* cml_nn_avgpool2d(int kernel_size, int stride, int padding, bool ceil_mode,
                            bool count_include_pad);
AvgPool3d* cml_nn_avgpool3d(int kernel_size, int stride, int padding, bool ceil_mode,
                            bool count_include_pad);
AdaptiveAvgPool1d* cml_nn_adaptive_avgpool1d(int output_size);
AdaptiveAvgPool2d* cml_nn_adaptive_avgpool2d(int output_h, int output_w);
AdaptiveMaxPool1d* cml_nn_adaptive_maxpool1d(int output_size);
AdaptiveMaxPool2d* cml_nn_adaptive_maxpool2d(int output_h, int output_w);

RNNCell* cml_nn_rnn_cell(int input_size, int hidden_size, bool use_bias, DType dtype,
                         DeviceType device);
LSTMCell* cml_nn_lstm_cell(int input_size, int hidden_size, bool use_bias, DType dtype,
                           DeviceType device);
GRUCell* cml_nn_gru_cell(int input_size, int hidden_size, bool use_bias, DType dtype,
                         DeviceType device);
RNN* cml_nn_rnn(int input_size, int hidden_size, int num_layers, bool bidirectional,
                bool batch_first, float dropout, bool use_bias, DType dtype, DeviceType device);
LSTM* cml_nn_lstm(int input_size, int hidden_size, int num_layers, bool bidirectional,
                  bool batch_first, float dropout, bool use_bias, DType dtype, DeviceType device);
GRU* cml_nn_gru(int input_size, int hidden_size, int num_layers, bool bidirectional,
                bool batch_first, float dropout, bool use_bias, DType dtype, DeviceType device);

MultiHeadAttention* cml_nn_multihead_attention(int embed_dim, int num_heads, float dropout,
                                               DType dtype, DeviceType device);
TransformerEncoderLayer* cml_nn_transformer_encoder_layer(int d_model, int nhead,
                                                          int dim_feedforward, float dropout,
                                                          DType dtype, DeviceType device);
TransformerEncoder* cml_nn_transformer_encoder(int d_model, int nhead, int dim_feedforward,
                                                float dropout, int num_layers,
                                                DType dtype, DeviceType device);
TransformerDecoderLayer* cml_nn_transformer_decoder_layer(int d_model, int nhead, int dim_feedforward,
                                                           float dropout, DType dtype, DeviceType device);
TransformerDecoder* cml_nn_transformer_decoder(int d_model, int nhead, int dim_feedforward,
                                                float dropout, int num_layers,
                                                DType dtype, DeviceType device);

Upsample* cml_nn_upsample(float scale_factor, const int* output_size, int num_output_dims,
                           UpsampleMode mode, bool align_corners);
PixelShuffle* cml_nn_pixel_shuffle(int upscale_factor);
PixelUnshuffle* cml_nn_pixel_unshuffle(int downscale_factor);

Tensor* cml_f_interpolate(Tensor* input, int* output_size, int num_dims,
                           UpsampleMode mode, bool align_corners);
Tensor* cml_f_pixel_shuffle(Tensor* input, int upscale_factor);
Tensor* cml_f_pixel_unshuffle(Tensor* input, int downscale_factor);

Optimizer* cml_optim_adam(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                          float beta1, float beta2, float eps);
Optimizer* cml_optim_sgd(Parameter** parameters, int num_parameters, float lr, float momentum,
                         float weight_decay);
Optimizer* cml_optim_rmsprop(Parameter** parameters, int num_parameters, float lr,
                             float weight_decay, float alpha, float eps);
Optimizer* cml_optim_adagrad(Parameter** parameters, int num_parameters, float lr,
                             float weight_decay, float eps);
Optimizer* cml_optim_adamw(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                            float beta1, float beta2, float epsilon);
Optimizer* cml_optim_nadam(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                            float beta1, float beta2, float epsilon);
Optimizer* cml_optim_adamax(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                             float beta1, float beta2, float epsilon);
Optimizer* cml_optim_adadelta(Parameter** parameters, int num_parameters, float rho,
                               float weight_decay, float epsilon);
Optimizer* cml_optim_lamb(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                           float beta1, float beta2, float epsilon);
Optimizer* cml_optim_lars(Parameter** parameters, int num_parameters, float lr, float momentum,
                           float weight_decay, float trust_coefficient);
Optimizer* cml_optim_muon(Parameter** parameters, int num_parameters, float lr, float momentum,
                           float weight_decay, bool nesterov);
Optimizer* cml_optim_adam_for_model(Module* model, float lr, float weight_decay, float beta1,
                                    float beta2, float eps);
Optimizer* cml_optim_sgd_for_model(Module* model, float lr, float momentum, float weight_decay);
void cml_optim_zero_grad(Optimizer* optimizer);
void cml_optim_step(Optimizer* optimizer);

LRScheduler* cml_lr_scheduler_step(Optimizer* opt, int step_size, float gamma);
LRScheduler* cml_lr_scheduler_reduce_on_plateau(Optimizer* opt, float factor, int patience, float min_lr);
LRScheduler* cml_lr_scheduler_exponential(Optimizer* opt, float gamma);
LRScheduler* cml_lr_scheduler_cosine(Optimizer* opt, int T_max, float eta_min);
LRScheduler* cml_lr_scheduler_one_cycle(Optimizer* opt, float max_lr, int total_steps,
                                         float pct_start, float div_factor, float final_div_factor);
LRScheduler* cml_lr_scheduler_multi_step(Optimizer* opt, int* milestones, int num_milestones,
                                          float gamma);
LRScheduler* cml_lr_scheduler_polynomial(Optimizer* opt, int total_iters, float power, float min_lr);
LRScheduler* cml_lr_scheduler_warmup(LRScheduler* inner, int warmup_steps, float warmup_start_factor);
float cml_lr_scheduler_update(LRScheduler* scheduler, float metric);
float cml_lr_scheduler_get_lr(LRScheduler* scheduler);
void cml_lr_scheduler_free(LRScheduler* scheduler);

Tensor* cml_nn_mse_loss(Tensor* input, Tensor* target);
Tensor* cml_nn_mae_loss(Tensor* input, Tensor* target);
Tensor* cml_nn_bce_loss(Tensor* input, Tensor* target);
Tensor* cml_nn_cross_entropy_loss(Tensor* input, Tensor* target);
Tensor* cml_nn_huber_loss(Tensor* input, Tensor* target, float delta);
Tensor* cml_nn_kl_div_loss(Tensor* input, Tensor* target);
Tensor* cml_nn_sparse_cross_entropy_loss(Tensor* input, Tensor* target);
Tensor* cml_nn_triplet_margin_loss(Tensor* anchor, Tensor* positive, Tensor* negative,
                                   float margin);
Tensor* cml_nn_cosine_embedding_loss(Tensor* x1, Tensor* x2, Tensor* target, float margin);
Tensor* cml_nn_nll_loss(Tensor* log_probs, Tensor* targets);

void cml_backward(Tensor* tensor, Tensor* gradient, bool retain_graph, bool create_graph);
void cml_zero_grad(Tensor* tensor);
void cml_no_grad(void);
void cml_enable_grad(void);
bool cml_is_grad_enabled(void);
bool cml_requires_grad(Tensor* t);
void cml_set_requires_grad(Tensor* t, bool requires_grad);
bool cml_is_leaf(Tensor* t);

/*
 * Free all IR nodes from forward/backward passes.
 * Safe to call after optimizer step since gradients are already materialized.
 */
void cml_reset_ir_context(void);

void cml_kernel_cache_clear(void);
void cml_kernel_cache_stats(size_t* hits, size_t* misses, size_t* count, size_t* memory);
double cml_kernel_cache_hit_rate(void);
void cml_kernel_cache_print_stats(void);

void cml_autocast_enter(DType target_dtype);
void cml_autocast_exit(void);
bool cml_autocast_is_enabled(void);
GradScaler* cml_grad_scaler_create(float init_scale, float growth_factor,
                                     float backoff_factor, int growth_interval);
void cml_grad_scaler_free(GradScaler* scaler);
Tensor* cml_grad_scaler_scale(GradScaler* scaler, Tensor* loss);
void cml_grad_scaler_unscale(GradScaler* scaler, Parameter** params, int num_params);
void cml_grad_scaler_step(GradScaler* scaler, void (*step_fn)(void*), void* optimizer);
void cml_grad_scaler_update(GradScaler* scaler);

SparseCOOData* cml_sparse_coo_tensor(Tensor* indices, Tensor* values,
                                      const int* dense_shape, int dense_ndim);
SparseCOOData* cml_sparse_from_dense(Tensor* dense);
Tensor* cml_sparse_to_dense(SparseCOOData* sparse, const TensorConfig* config);
Tensor* cml_sparse_matmul(SparseCOOData* sparse, Tensor* dense);
SparseCOOData* cml_sparse_coalesce(SparseCOOData* sparse);
void cml_sparse_free(SparseCOOData* sparse);

Tensor* cml_from_url(const char* url);
GGUFContext* cml_gguf_open_read(const char* path);
GGUFContext* cml_gguf_open_write(const char* path);
void cml_gguf_close(GGUFContext* ctx);
int cml_gguf_write_tensor(GGUFContext* ctx, const char* name, Tensor* tensor);
Tensor* cml_gguf_read_tensor(GGUFContext* ctx, const char* name);
int cml_module_save_gguf(Module* module, const char* path);
int cml_module_load_gguf(Module* module, const char* path);
SafeTensorsContext* cml_safetensors_open_read(const char* path);
SafeTensorsContext* cml_safetensors_open_write(const char* path);
void cml_safetensors_close(SafeTensorsContext* ctx);
int cml_safetensors_write_tensor(SafeTensorsContext* ctx, const char* name, Tensor* tensor);
Tensor* cml_safetensors_read_tensor(SafeTensorsContext* ctx, const char* name);
int cml_module_save_safetensors(Module* module, const char* path);
int cml_module_load_safetensors(Module* module, const char* path);

#ifdef __cplusplus
extern "C" {
#endif

int cml_init(void);
int cml_cleanup(void);
void cml_register_cleanup_context(CleanupContext* ctx);
void cml_get_version(int* major, int* minor, int* patch, const char** version_string);
const char* cml_get_build_info(void);
bool cml_is_initialized(void);
int cml_get_init_count(void);
int cml_force_cleanup(void);

typedef void* (*CMLGlobalErrorHandler)(int error_code, const char* error_msg, void* context);
void cml_set_error_handler(CMLGlobalErrorHandler handler);
CMLGlobalErrorHandler cml_get_error_handler(void);

#ifdef __cplusplus
}
#endif

#endif // CML_H
