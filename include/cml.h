/**
 * @file cml.h
 * @brief C-ML: C Machine Learning Library - Main Header
 * @version 0.0.2
 * @author Arinjay
 * @date 2025
 *
 * This is the main header file for the C-ML library. Include this file
 * to access all the major components.
 *
 * @example
 * ```c
 * #include "cml.h"
 *
 * // Create model
 * Sequential *model = nn_sequential();
 * sequential_add(model, (Module*)nn_linear(2, 4, DTYPE_FLOAT32, DEVICE_CPU, true));
 * sequential_add(model, (Module*)nn_relu(false));
 *
 * // Create optimizer
 * Parameter **params; int num_params;
 * module_collect_parameters((Module*)model, &params, &num_params, true);
 * Optimizer *optimizer = optim_adam(params, num_params, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);
 *
 * // Training loop
 * for (int epoch = 0; epoch < epochs; epoch++) {
 *     optimizer_zero_grad(optimizer);
 *     Tensor *outputs = module_forward((Module*)model, X);
 *     Tensor *loss = tensor_mse_loss(outputs, y);
 *     tensor_backward(loss, NULL, false, false);
 *     optimizer_step(optimizer);
 * }
 * ```
 */

#ifndef CML_H
#define CML_H

// Version information - managed by release process

// Core utilities
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

#include "nn.h"
#include "nn/layers.h"

#include "optim.h"
#include "nn/model_io.h"

#ifdef CML_HAS_DISTRIBUTED
#include "distributed/distributed.h"
#include "distributed/data_parallel.h"
#include "distributed/pipeline_parallel.h"
#endif

#include "zoo/zoo.h"
#include "datasets/datasets.h"

// Global utility functions
void cml_summary(struct Module* module);

// Resource tracking functions (for automatic cleanup)
void cml_track_module(struct Module* module);
void cml_untrack_module(struct Module* module);
void cml_track_optimizer(struct Optimizer* optimizer);
void cml_track_dataset(struct Dataset* dataset);

/**
 * @brief Create an empty tensor with specified shape and configuration
 * @param shape Array of dimensions
 * @param ndim Number of dimensions
 * @param config Tensor configuration (dtype, device)
 * @return New tensor, or NULL on failure
 */
Tensor* cml_empty(int* shape, int ndim, const TensorConfig* config);

/**
 * @brief Create a tensor filled with zeros
 * @param shape Array of dimensions
 * @param ndim Number of dimensions
 * @param config Tensor configuration (dtype, device)
 * @return New tensor, or NULL on failure
 */
Tensor* cml_zeros(int* shape, int ndim, const TensorConfig* config);

/**
 * @brief Create a tensor filled with ones
 * @param shape Array of dimensions
 * @param ndim Number of dimensions
 * @param config Tensor configuration (dtype, device)
 * @return New tensor, or NULL on failure
 */
Tensor* cml_ones(int* shape, int ndim, const TensorConfig* config);

/**
 * @brief Create a tensor filled with a constant value
 * @param shape Array of dimensions
 * @param ndim Number of dimensions
 * @param config Tensor configuration (dtype, device)
 * @param value Value to fill the tensor with
 * @return New tensor, or NULL on failure
 */
Tensor* cml_full(int* shape, int ndim, const TensorConfig* config, float value);

/**
 * @brief Create a tensor from existing data
 * @param data Pointer to data
 * @param shape Array of dimensions
 * @param ndim Number of dimensions
 * @param config Tensor configuration (dtype, device)
 * @return New tensor, or NULL on failure
 */
Tensor* cml_tensor(void* data, int* shape, int ndim, const TensorConfig* config);

/**
 * @brief Create a 2D tensor filled with zeros
 * @param rows Number of rows
 * @param cols Number of columns
 * @return New tensor, or NULL on failure
 */
Tensor* cml_zeros_2d(int rows, int cols);

/**
 * @brief Create a 2D tensor filled with ones
 * @param rows Number of rows
 * @param cols Number of columns
 * @return New tensor, or NULL on failure
 */
Tensor* cml_ones_2d(int rows, int cols);

/**
 * @brief Create an empty 2D tensor
 * @param rows Number of rows
 * @param cols Number of columns
 * @return New tensor, or NULL on failure
 */
Tensor* cml_empty_2d(int rows, int cols);

/**
 * @brief Create a 2D tensor from data
 * @param data Pointer to flat data array
 * @param rows Number of rows
 * @param cols Number of columns
 * @return New tensor, or NULL on failure
 */
Tensor* cml_tensor_2d(const float* data, int rows, int cols);

/**
 * @brief Create a 1D tensor filled with zeros
 * @param size Number of elements
 * @return New tensor, or NULL on failure
 */
Tensor* cml_zeros_1d(int size);

/**
 * @brief Create a 1D tensor filled with ones
 * @param size Number of elements
 * @return New tensor, or NULL on failure
 */
Tensor* cml_ones_1d(int size);

/**
 * @brief Create an empty 1D tensor
 * @param size Number of elements
 * @return New tensor, or NULL on failure
 */
Tensor* cml_empty_1d(int size);

/**
 * @brief Create a 1D tensor from data
 * @param data Pointer to data array
 * @param size Number of elements
 * @return New tensor, or NULL on failure
 */
Tensor* cml_tensor_1d(const float* data, int size);

/**
 * @brief Element-wise addition
 * @param a First tensor
 * @param b Second tensor
 * @return Result tensor
 */
Tensor* cml_add(Tensor* a, Tensor* b);

/**
 * @brief Element-wise subtraction
 * @param a First tensor
 * @param b Second tensor
 * @return Result tensor
 */
Tensor* cml_sub(Tensor* a, Tensor* b);

/**
 * @brief Element-wise multiplication
 * @param a First tensor
 * @param b Second tensor
 * @return Result tensor
 */
Tensor* cml_mul(Tensor* a, Tensor* b);

/**
 * @brief Element-wise division
 * @param a First tensor
 * @param b Second tensor
 * @return Result tensor
 */
Tensor* cml_div(Tensor* a, Tensor* b);

/**
 * @brief Element-wise exponential
 * @param a Input tensor
 * @return Result tensor
 */
Tensor* cml_exp(Tensor* a);

/**
 * @brief Element-wise natural logarithm
 * @param a Input tensor
 * @return Result tensor
 */
Tensor* cml_log(Tensor* a);

/**
 * @brief Element-wise square root
 * @param a Input tensor
 * @return Result tensor
 */
Tensor* cml_sqrt(Tensor* a);

/**
 * @brief Element-wise sine
 * @param a Input tensor
 * @return Result tensor
 */
Tensor* cml_sin(Tensor* a);

/**
 * @brief Element-wise cosine
 * @param a Input tensor
 * @return Result tensor
 */
Tensor* cml_cos(Tensor* a);

/**
 * @brief Element-wise tangent
 * @param a Input tensor
 * @return Result tensor
 */
Tensor* cml_tan(Tensor* a);

/**
 * @brief Element-wise power function
 * @param a Base tensor
 * @param b Exponent tensor
 * @return Result tensor
 */
Tensor* cml_pow(Tensor* a, Tensor* b);

/**
 * @brief Rectified Linear Unit activation
 * @param a Input tensor
 * @return Result tensor
 */
Tensor* cml_relu(Tensor* a);

/**
 * @brief Sigmoid activation
 * @param a Input tensor
 * @return Result tensor
 */
Tensor* cml_sigmoid(Tensor* a);

/**
 * @brief Hyperbolic tangent activation
 * @param a Input tensor
 * @return Result tensor
 */
Tensor* cml_tanh(Tensor* a);

/**
 * @brief Softmax activation along a dimension
 * @param a Input tensor
 * @param dim Dimension to apply softmax
 * @return Result tensor
 */
Tensor* cml_softmax(Tensor* a, int dim);

/**
 * @brief Sum reduction
 * @param a Input tensor
 * @param dim Dimension to reduce
 * @param keepdim Whether to keep the reduced dimension
 * @return Result tensor
 */
Tensor* cml_sum(Tensor* a, int dim, bool keepdim);

/**
 * @brief Mean reduction
 * @param a Input tensor
 * @param dim Dimension to reduce
 * @param keepdim Whether to keep the reduced dimension
 * @return Result tensor
 */
Tensor* cml_mean(Tensor* a, int dim, bool keepdim);

/**
 * @brief Max reduction
 * @param a Input tensor
 * @param dim Dimension to reduce
 * @param keepdim Whether to keep the reduced dimension
 * @return Result tensor
 */
Tensor* cml_max(Tensor* a, int dim, bool keepdim);

/**
 * @brief Min reduction
 * @param a Input tensor
 * @param dim Dimension to reduce
 * @param keepdim Whether to keep the reduced dimension
 * @return Result tensor
 */
Tensor* cml_min(Tensor* a, int dim, bool keepdim);

/**
 * @brief Matrix multiplication
 * @param a First matrix
 * @param b Second matrix
 * @return Result tensor
 */
Tensor* cml_matmul(Tensor* a, Tensor* b);

/**
 * @brief Transpose tensor dimensions
 * @param a Input tensor
 * @param dim1 First dimension
 * @param dim2 Second dimension
 * @return Result tensor
 */
Tensor* cml_transpose(Tensor* a, int dim1, int dim2);

/**
 * @brief Reshape tensor
 * @param a Input tensor
 * @param new_shape New shape array
 * @param new_ndim New number of dimensions
 * @return Result tensor
 */
Tensor* cml_reshape(Tensor* a, int* new_shape, int new_ndim);

/**
 * @brief Clone tensor
 * @param a Input tensor
 * @return Cloned tensor
 */
Tensor* cml_clone(Tensor* a);

/**
 * @brief Detach tensor from computation graph
 * @param a Input tensor
 * @return Detached tensor
 */
Tensor* cml_detach(Tensor* a);

/**
 * @brief Concatenate tensors
 * @param tensors Array of tensors
 * @param num_tensors Number of tensors
 * @param dim Dimension to concatenate along
 * @return Result tensor
 */
Tensor* cml_concat(Tensor** tensors, int num_tensors, int dim);

/**
 * @brief Stack tensors
 * @param tensors Array of tensors
 * @param num_tensors Number of tensors
 * @param dim Dimension to stack along
 * @return Result tensor
 */
Tensor* cml_stack(Tensor** tensors, int num_tensors, int dim);

/**
 * @brief Create a sequential container
 * @return New sequential module
 */
Sequential* cml_nn_sequential(void);

/**
 * @brief Add a layer to a sequential container
 * @param seq Sequential container
 * @param layer Layer to add
 * @return The sequential container
 */
Sequential* cml_nn_sequential_add(Sequential* seq, Module* layer);

/**
 * @brief Forward pass through a sequential container
 * @param seq Sequential container
 * @param input Input tensor
 * @return Output tensor
 */
Tensor* cml_nn_sequential_forward(Sequential* seq, Tensor* input);

/**
 * @brief Create a linear layer
 * @param in_features Input features
 * @param out_features Output features
 * @param dtype Data type
 * @param device Device
 * @param bias Whether to include bias
 * @return New linear layer
 */
Linear* cml_nn_linear(int in_features, int out_features, DType dtype, DeviceType device, bool bias);

/**
 * @brief Create a ReLU activation layer
 * @param inplace Whether to perform operation in-place
 * @return New ReLU layer
 */
ReLU* cml_nn_relu(bool inplace);

/**
 * @brief Create a Sigmoid activation layer
 * @return New Sigmoid layer
 */
Sigmoid* cml_nn_sigmoid(void);

/**
 * @brief Create a Tanh activation layer
 * @return New Tanh layer
 */
Tanh* cml_nn_tanh(void);

/**
 * @brief Create a LeakyReLU activation layer
 * @param negative_slope Slope for negative values
 * @param inplace Whether to perform operation in-place
 * @return New LeakyReLU layer
 */
LeakyReLU* cml_nn_leaky_relu(float negative_slope, bool inplace);

/**
 * @brief Create a Dropout layer
 * @param p Dropout probability
 * @param inplace Whether to perform operation in-place
 * @return New Dropout layer
 */
Dropout* cml_nn_dropout(float p, bool inplace);

/**
 * @brief Create a 2D Convolution layer
 * @param in_channels Input channels
 * @param out_channels Output channels
 * @param kernel_size Kernel size
 * @param stride Stride
 * @param padding Padding
 * @param dilation Dilation
 * @param bias Whether to include bias
 * @param dtype Data type
 * @param device Device
 * @return New Conv2d layer
 */
Conv2d* cml_nn_conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding,
                      int dilation, bool bias, DType dtype, DeviceType device);

/**
 * @brief Create a 2D BatchNorm layer
 * @param num_features Number of features
 * @param eps Epsilon value
 * @param momentum Momentum value
 * @param affine Whether to include learnable affine parameters
 * @param track_running_stats Whether to track running statistics
 * @param dtype Data type
 * @param device Device
 * @return New BatchNorm2d layer
 */
BatchNorm2d* cml_nn_batchnorm2d(int num_features, float eps, float momentum, bool affine,
                                bool track_running_stats, DType dtype, DeviceType device);

/**
 * @brief Create a LayerNorm layer
 * @param normalized_shape Shape to normalize
 * @param eps Epsilon value
 * @param affine Whether to include learnable affine parameters
 * @param dtype Data type
 * @param device Device
 * @return New LayerNorm layer
 */
LayerNorm* cml_nn_layernorm(int normalized_shape, float eps, bool affine, DType dtype,
                            DeviceType device);

/**
 * @brief Create a 2D MaxPool layer
 * @param kernel_size Kernel size
 * @param stride Stride
 * @param padding Padding
 * @param dilation Dilation
 * @param ceil_mode Whether to use ceil for output size calculation
 * @return New MaxPool2d layer
 */
MaxPool2d* cml_nn_maxpool2d(int kernel_size, int stride, int padding, int dilation, bool ceil_mode);

/**
 * @brief Create a 2D AvgPool layer
 * @param kernel_size Kernel size
 * @param stride Stride
 * @param padding Padding
 * @param ceil_mode Whether to use ceil for output size calculation
 * @param count_include_pad Whether to include padding in average calculation
 * @return New AvgPool2d layer
 */
AvgPool2d* cml_nn_avgpool2d(int kernel_size, int stride, int padding, bool ceil_mode,
                            bool count_include_pad);

// New layer wrappers
Conv1d* cml_nn_conv1d(int in_channels, int out_channels, int kernel_size, int stride, int padding,
                      int dilation, bool use_bias, DType dtype, DeviceType device);
Conv3d* cml_nn_conv3d(int in_channels, int out_channels, int kernel_size, int stride, int padding,
                      int dilation, bool use_bias, DType dtype, DeviceType device);
Embedding* cml_nn_embedding(int num_embeddings, int embedding_dim, int padding_idx, DType dtype,
                            DeviceType device);
GroupNorm* cml_nn_groupnorm(int num_groups, int num_channels, float eps, bool affine, DType dtype,
                            DeviceType device);
RNNCell* cml_nn_rnn_cell(int input_size, int hidden_size, bool use_bias, DType dtype,
                         DeviceType device);
LSTMCell* cml_nn_lstm_cell(int input_size, int hidden_size, bool use_bias, DType dtype,
                           DeviceType device);
GRUCell* cml_nn_gru_cell(int input_size, int hidden_size, bool use_bias, DType dtype,
                         DeviceType device);
MultiHeadAttention* cml_nn_multihead_attention(int embed_dim, int num_heads, float dropout,
                                               DType dtype, DeviceType device);
TransformerEncoderLayer* cml_nn_transformer_encoder_layer(int d_model, int nhead,
                                                          int dim_feedforward, float dropout,
                                                          DType dtype, DeviceType device);
ModuleList* cml_nn_module_list(void);
ModuleDict* cml_nn_module_dict(void);

/**
 * @brief Create an Adam optimizer
 * @param parameters Array of parameters
 * @param num_parameters Number of parameters
 * @param lr Learning rate
 * @param weight_decay Weight decay
 * @param beta1 Beta1 coefficient
 * @param beta2 Beta2 coefficient
 * @param eps Epsilon value
 * @return New Adam optimizer
 */
Optimizer* cml_optim_adam(Parameter** parameters, int num_parameters, float lr, float weight_decay,
                          float beta1, float beta2, float eps);

/**
 * @brief Create an SGD optimizer
 * @param parameters Array of parameters
 * @param num_parameters Number of parameters
 * @param lr Learning rate
 * @param momentum Momentum factor
 * @param weight_decay Weight decay
 * @return New SGD optimizer
 */
Optimizer* cml_optim_sgd(Parameter** parameters, int num_parameters, float lr, float momentum,
                         float weight_decay);

/**
 * @brief Create an RMSprop optimizer
 * @param parameters Array of parameters
 * @param num_parameters Number of parameters
 * @param lr Learning rate
 * @param weight_decay Weight decay
 * @param alpha Smoothing constant
 * @param eps Epsilon value
 * @return New RMSprop optimizer
 */
Optimizer* cml_optim_rmsprop(Parameter** parameters, int num_parameters, float lr,
                             float weight_decay, float alpha, float eps);

/**
 * @brief Create an Adagrad optimizer
 * @param parameters Array of parameters
 * @param num_parameters Number of parameters
 * @param lr Learning rate
 * @param weight_decay Weight decay
 * @param eps Epsilon value
 * @return New Adagrad optimizer
 */
Optimizer* cml_optim_adagrad(Parameter** parameters, int num_parameters, float lr,
                             float weight_decay, float eps);

/**
 * @brief Create an Adam optimizer for a model
 * @param model Model to optimize
 * @param lr Learning rate
 * @param weight_decay Weight decay
 * @param beta1 Beta1 coefficient
 * @param beta2 Beta2 coefficient
 * @param eps Epsilon value
 * @return New Adam optimizer
 */
Optimizer* cml_optim_adam_for_model(Module* model, float lr, float weight_decay, float beta1,
                                    float beta2, float eps);

/**
 * @brief Create an SGD optimizer for a model
 * @param model Model to optimize
 * @param lr Learning rate
 * @param momentum Momentum factor
 * @param weight_decay Weight decay
 * @return New SGD optimizer
 */
Optimizer* cml_optim_sgd_for_model(Module* model, float lr, float momentum, float weight_decay);

/**
 * @brief Zero gradients of an optimizer
 * @param optimizer Optimizer instance
 */
void cml_optim_zero_grad(Optimizer* optimizer);

/**
 * @brief Perform an optimization step
 * @param optimizer Optimizer instance
 */
void cml_optim_step(Optimizer* optimizer);

/**
 * @brief MSE Loss
 * @param input Input tensor
 * @param target Target tensor
 * @return Loss tensor
 */
Tensor* cml_nn_mse_loss(Tensor* input, Tensor* target);

/**
 * @brief MAE Loss
 * @param input Input tensor
 * @param target Target tensor
 * @return Loss tensor
 */
Tensor* cml_nn_mae_loss(Tensor* input, Tensor* target);

/**
 * @brief BCE Loss
 * @param input Input tensor
 * @param target Target tensor
 * @return Loss tensor
 */
Tensor* cml_nn_bce_loss(Tensor* input, Tensor* target);

/**
 * @brief Cross Entropy Loss
 * @param input Input tensor
 * @param target Target tensor
 * @return Loss tensor
 */
Tensor* cml_nn_cross_entropy_loss(Tensor* input, Tensor* target);

/**
 * @brief Huber Loss
 * @param input Input tensor
 * @param target Target tensor
 * @param delta Delta value
 * @return Loss tensor
 */
Tensor* cml_nn_huber_loss(Tensor* input, Tensor* target, float delta);

/**
 * @brief KL Divergence Loss
 * @param input Input tensor
 * @param target Target tensor
 * @return Loss tensor
 */
Tensor* cml_nn_kl_div_loss(Tensor* input, Tensor* target);

/**
 * @brief Sparse Cross Entropy Loss (numerically stable)
 * @param input Predicted logits [N, C]
 * @param target Target class indices [N]
 * @return Loss tensor
 */
Tensor* cml_nn_sparse_cross_entropy_loss(Tensor* input, Tensor* target);

/**
 * @brief Perform backward pass
 * @param tensor Tensor to start backward pass from
 * @param gradient Gradient tensor (optional)
 * @param retain_graph Whether to retain the graph
 * @param create_graph Whether to create a graph for higher-order derivatives
 */
void cml_backward(Tensor* tensor, Tensor* gradient, bool retain_graph, bool create_graph);

/**
 * @brief Zero gradients of a tensor
 * @param tensor Tensor to zero gradients for
 */
void cml_zero_grad(Tensor* tensor);

/**
 * @brief Disable gradient calculation
 */
void cml_no_grad(void);

/**
 * @brief Enable gradient calculation
 */
void cml_enable_grad(void);

/**
 * @brief Check if gradient calculation is enabled
 * @return true if enabled, false otherwise
 */
bool cml_is_grad_enabled(void);

/**
 * @brief Check if tensor requires gradients
 * @param t Input tensor
 * @return true if requires gradients, false otherwise
 */
bool cml_requires_grad(Tensor* t);

/**
 * @brief Set whether tensor requires gradients
 * @param t Input tensor
 * @param requires_grad Whether to require gradients
 */
void cml_set_requires_grad(Tensor* t, bool requires_grad);

/**
 * @brief Check if tensor is a leaf node
 * @param t Input tensor
 * @return true if leaf node, false otherwise
 */
bool cml_is_leaf(Tensor* t);

/**
 * @brief Reset the global IR context
 *
 * This frees all IR nodes accumulated during forward/backward passes.
 * Call this after each training batch to prevent memory growth.
 * Gradients are already materialized in parameter tensors after optimizer step,
 * so it's safe to call this after cml_optim_step().
 */
void cml_reset_ir_context(void);

// JIT Kernel Cache Management

/**
 * @brief Clear the JIT kernel cache
 *
 * This frees all cached LLVM JIT execution engines. Call this if you want to
 * reclaim memory used by cached kernels. Note that this will cause recompilation
 * on next execution.
 *
 * The kernel cache uses LRU eviction by default (256 entry limit), so you
 * typically don't need to call this unless you want to force memory reclamation.
 */
void cml_kernel_cache_clear(void);

/**
 * @brief Get kernel cache statistics
 *
 * @param hits Pointer to store cache hit count (can be NULL)
 * @param misses Pointer to store cache miss count (can be NULL)
 * @param count Pointer to store current entry count (can be NULL)
 * @param memory Pointer to store estimated memory usage in bytes (can be NULL)
 */
void cml_kernel_cache_stats(size_t* hits, size_t* misses, size_t* count, size_t* memory);

/**
 * @brief Get kernel cache hit rate
 *
 * @return Hit rate as a value between 0.0 and 1.0
 */
double cml_kernel_cache_hit_rate(void);

/**
 * @brief Print kernel cache statistics to stdout
 *
 * Prints detailed cache statistics including hit/miss counts, hit rate,
 * memory usage, and eviction counts.
 */
void cml_kernel_cache_print_stats(void);

/**
 * @brief Forward pass through a module
 * @param module Module instance
 * @param input Input tensor
 * @return Output tensor
 */
Tensor* cml_nn_module_forward(Module* module, Tensor* input);

/**
 * @brief Set module training mode
 * @param module Module instance
 * @param training Whether to set to training mode
 */
void cml_nn_module_set_training(Module* module, bool training);

/**
 * @brief Check if module is in training mode
 * @param module Module instance
 * @return true if in training mode, false otherwise
 */
bool cml_nn_module_is_training(Module* module);

/**
 * @brief Set module to evaluation mode
 * @param module Module instance
 */
void cml_nn_module_eval(Module* module);

/**
 * @brief Set module to training mode
 * @param module Module instance
 */
void cml_nn_module_train(Module* module);

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the C-ML library
 *
 * This function should be called before using any C-ML functionality.
 * It initializes internal systems, sets up logging, and prepares
 * the library for use.
 *
 * @return 0 on success, negative value on failure
 */
int cml_init(void);

/**
 * @brief Cleanup the C-ML library
 *
 * This function should be called when the library is no longer needed.
 * It cleans up internal resources and ensures proper shutdown.
 * Note: This is automatically called on program exit via atexit().
 *
 * @return 0 on success, negative value on failure
 */
int cml_cleanup(void);

/**
 * @brief Register a cleanup context for automatic cleanup
 *
 * This function registers a cleanup context to be automatically freed
 * when the program exits. Cleanup contexts created with cleanup_context_create()
 * are automatically registered.
 *
 * @param ctx Cleanup context to register
 */
void cml_register_cleanup_context(CleanupContext* ctx);

/**
 * @brief Get library version information
 *
 * @param major Pointer to store major version
 * @param minor Pointer to store minor version
 * @param patch Pointer to store patch version
 * @param version_string Pointer to store version string
 */
void cml_get_version(int* major, int* minor, int* patch, const char** version_string);

/**
 * @brief Get library build information
 *
 * @return String containing build information (compiler, flags, etc.)
 */
const char* cml_get_build_info(void);

/**
 * @brief Check if C-ML library is initialized
 *
 * @return true if initialized, false otherwise
 */
bool cml_is_initialized(void);

/**
 * @brief Get C-ML library initialization count
 *
 * @return Current initialization reference count
 */
int cml_get_init_count(void);

/**
 * @brief Force cleanup of C-ML library (ignores reference count)
 *
 * This function should be used with caution as it forces cleanup
 * regardless of the reference count.
 *
 * @return 0 on success, negative value on failure
 */
int cml_force_cleanup(void);

/**
 * @brief Global error handler function pointer type
 *
 * Called when a checked constructor would fail.
 * Should handle the error and return a sentinel object.
 *
 * @param error_code Error code
 * @param error_msg Error message
 * @param context Context information
 * @return Sentinel object to use instead of NULL
 */
typedef void* (*CMLGlobalErrorHandler)(int error_code, const char* error_msg, void* context);

/**
 * @brief Set global error handler for checked constructors
 *
 * @param handler Error handler function (NULL to use default)
 */
void cml_set_error_handler(CMLGlobalErrorHandler handler);

/**
 * @brief Get current error handler
 *
 * @return Current error handler function
 */
CMLGlobalErrorHandler cml_get_error_handler(void);

#ifdef __cplusplus
}
#endif

#endif // CML_H
