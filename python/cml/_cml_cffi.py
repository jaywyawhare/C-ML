"""
CFFI Definitions for CML (C Machine Learning Library)

This module defines the CFFI interface to the C-ML library.
It exposes the necessary C structures and functions for Python.
"""

from cffi import FFI

ffi = FFI()

ffi.cdef(
    """

    typedef enum {
        DTYPE_FLOAT32,
        DTYPE_FLOAT64,
        DTYPE_INT32,
        DTYPE_INT64,
        DTYPE_BOOL,
        DTYPE_FLOAT16,
        DTYPE_BFLOAT16,
        DTYPE_INT8,
        DTYPE_UINT8,
        DTYPE_INT16,
        DTYPE_UINT16,
        DTYPE_UINT32,
        DTYPE_UINT64,
        DTYPE_FLOAT8_E4M3,
        DTYPE_FLOAT8_E5M2,
    } DType;

    typedef enum {
        DEVICE_CPU,
        DEVICE_CUDA,
        DEVICE_METAL,
        DEVICE_ROCM,
        DEVICE_SIM_GPU,
        DEVICE_AUTO,
    } DeviceType;

    typedef enum {
        SCATTER_REDUCE_SUM,
        SCATTER_REDUCE_PROD,
        SCATTER_REDUCE_MEAN,
        SCATTER_REDUCE_AMAX,
        SCATTER_REDUCE_AMIN,
    } ScatterReduceMode;

    typedef enum {
        INTERP_NEAREST,
        INTERP_BILINEAR,
    } InterpMode;

    typedef enum {
        UPSAMPLE_NEAREST,
        UPSAMPLE_BILINEAR,
        UPSAMPLE_BICUBIC,
    } UpsampleMode;

    typedef enum {
        LOG_LEVEL_DEBUG   = 0,
        LOG_LEVEL_INFO    = 1,
        LOG_LEVEL_WARNING = 2,
        LOG_LEVEL_ERROR   = 3,
    } LogLevel;


    struct IRNode;
    struct CMLGraph;
    typedef struct CMLGraph* CMLGraph_t;
    struct CMLBackendBuffer;
    typedef struct CMLBackendBuffer* CMLBackendBuffer_t;

    // Tensor struct (full definition - Python accesses fields directly)

    typedef struct Tensor {
        int* shape;
        int ndim;
        size_t numel;
        DType dtype;
        DeviceType device;

        struct IRNode* ir_node;
        CMLGraph_t ir_context;

        bool is_executed;
        void* data;
        bool owns_data;
        bool from_buffer_cache;

        bool requires_grad;
        struct Tensor* grad;

        int ref_count;
        struct Tensor* base;

        size_t* strides;
        size_t storage_offset;
        bool is_contiguous;
        CMLBackendBuffer_t buffer_handle;

        void* user_data;
    } Tensor;

    // TensorConfig struct (full definition - used in creation functions)

    typedef struct TensorConfig {
        DType dtype;
        DeviceType device;
        bool has_dtype;
        bool has_device;
    } TensorConfig;


    typedef struct {
        Tensor* Q;
        Tensor* R;
    } QRResult;

    typedef struct {
        Tensor* U;
        Tensor* S;
        Tensor* Vt;
    } SVDResult;

    typedef struct SparseCOOData {
        Tensor* indices;
        Tensor* values;
        int* dense_shape;
        int dense_ndim;
        int nnz;
    } SparseCOOData;

    // Parameter and Module structs (full definitions for field access)

    typedef struct Parameter {
        Tensor* tensor;
        bool requires_grad;
        char* name;
    } Parameter;

    typedef Tensor* (*ForwardFn)(struct Module* module, Tensor* input);
    typedef void (*FreeFn)(struct Module* module);

    typedef struct Module {
        char* name;
        ForwardFn forward;
        FreeFn free;

        Parameter** parameters;
        int num_parameters;
        int parameters_capacity;

        struct Module* next;

        bool training;
        void* user_data;

        const char* version;
        const char* description;
    } Module;

    // Opaque struct typedefs (Python only uses pointers to these)

    typedef struct Sequential { ...; } Sequential;
    typedef struct Linear { ...; } Linear;
    typedef struct Conv1d { ...; } Conv1d;
    typedef struct Conv2d { ...; } Conv2d;
    typedef struct Conv3d { ...; } Conv3d;
    typedef struct ConvTranspose1d { ...; } ConvTranspose1d;
    typedef struct ConvTranspose3d { ...; } ConvTranspose3d;
    typedef struct ReLU { ...; } ReLU;
    typedef struct Sigmoid { ...; } Sigmoid;
    typedef struct Tanh { ...; } Tanh;
    typedef struct LeakyReLU { ...; } LeakyReLU;
    typedef struct Dropout { ...; } Dropout;
    typedef struct BatchNorm1d { ...; } BatchNorm1d;
    typedef struct BatchNorm2d { ...; } BatchNorm2d;
    typedef struct BatchNorm3d { ...; } BatchNorm3d;
    typedef struct LayerNorm { ...; } LayerNorm;
    typedef struct LayerNorm2d { ...; } LayerNorm2d;
    typedef struct InstanceNorm2d { ...; } InstanceNorm2d;
    typedef struct GroupNorm { ...; } GroupNorm;
    typedef struct Embedding { ...; } Embedding;
    typedef struct Flatten { ...; } Flatten;
    typedef struct Identity { ...; } Identity;
    typedef struct PReLU { ...; } PReLU;
    typedef struct ModuleList { ...; } ModuleList;
    typedef struct ModuleDict { ...; } ModuleDict;
    typedef struct MaxPool1d { ...; } MaxPool1d;
    typedef struct MaxPool2d { ...; } MaxPool2d;
    typedef struct MaxPool3d { ...; } MaxPool3d;
    typedef struct AvgPool1d { ...; } AvgPool1d;
    typedef struct AvgPool2d { ...; } AvgPool2d;
    typedef struct AvgPool3d { ...; } AvgPool3d;
    typedef struct AdaptiveAvgPool1d { ...; } AdaptiveAvgPool1d;
    typedef struct AdaptiveAvgPool2d { ...; } AdaptiveAvgPool2d;
    typedef struct AdaptiveMaxPool1d { ...; } AdaptiveMaxPool1d;
    typedef struct AdaptiveMaxPool2d { ...; } AdaptiveMaxPool2d;
    typedef struct RNNCell { ...; } RNNCell;
    typedef struct LSTMCell { ...; } LSTMCell;
    typedef struct GRUCell { ...; } GRUCell;
    typedef struct RNN { ...; } RNN;
    typedef struct LSTM { ...; } LSTM;
    typedef struct GRU { ...; } GRU;
    typedef struct MultiHeadAttention { ...; } MultiHeadAttention;
    typedef struct TransformerEncoderLayer { ...; } TransformerEncoderLayer;
    typedef struct TransformerEncoder { ...; } TransformerEncoder;
    typedef struct TransformerDecoderLayer { ...; } TransformerDecoderLayer;
    typedef struct TransformerDecoder { ...; } TransformerDecoder;
    typedef struct Upsample { ...; } Upsample;
    typedef struct PixelShuffle { ...; } PixelShuffle;
    typedef struct PixelUnshuffle { ...; } PixelUnshuffle;
    typedef struct Optimizer { ...; } Optimizer;
    typedef struct Dataset { ...; } Dataset;

    // Pointer-only opaque types
    typedef struct GGUFContext GGUFContext;
    typedef struct SafeTensorsContext SafeTensorsContext;
    typedef struct GradScaler GradScaler;
    typedef struct LRScheduler LRScheduler;
    typedef struct CleanupContext CleanupContext;

    // Tensor utility functions (from tensor/tensor.h)

    size_t cml_dtype_size(DType dtype);
    size_t tensor_numel(int* shape, int ndim);

    void tensor_free(Tensor* t);
    Tensor* tensor_clone(Tensor* t);
    float tensor_get_float(Tensor* t, size_t idx);
    void tensor_set_float(Tensor* t, size_t idx, float value);
    void* tensor_data_ptr(Tensor* t);
    int tensor_ensure_executed(Tensor* t);
    bool tensor_is_scalar(Tensor* t);
    bool tensor_is_contiguous(Tensor* t);
    Tensor* tensor_from_data(const void* data, int* shape, int ndim, const TensorConfig* config);

    // Module functions (from nn.h)

    void module_free(Module* module);
    Tensor* module_forward(Module* module, Tensor* input);
    void module_set_training(Module* module, bool training);
    bool module_is_training(Module* module);
    int module_collect_parameters(Module* module, Parameter*** params_out, int* num_params_out,
                                  bool recursive);

    // Optimizer functions (from optim.h)

    void optimizer_free(Optimizer* optimizer);

    // Autograd functions (from autograd/autograd.h)

    void autograd_no_grad_enter(void);
    void autograd_no_grad_exit(void);
    bool autograd_is_grad_enabled(void);
    bool tensor_requires_grad(struct Tensor* t);
    void tensor_set_requires_grad(struct Tensor* t, bool requires_grad);

    // Dataset functions (from datasets/datasets.h)

    Dataset* cml_dataset_load(const char* name);
    int dataset_normalize(Dataset* dataset, const char* method);

    // Logging (from core/logging.h)

    void cml_set_log_level(LogLevel level);

    // Error stack (from core/error_stack.h)
    void error_stack_init(void);
    void error_stack_cleanup(void);
    bool error_stack_has_errors(void);
    const char* error_stack_get_last_message(void);
    int error_stack_get_last_code(void);

    // Config (from core/config.h)

    DeviceType cml_get_default_device(void);
    void cml_set_default_device(DeviceType device);
    void cml_set_default_dtype(DType dtype);
    DType cml_get_default_dtype(void);
    void cml_seed(uint64_t seed);

    // Device detection (from backend/device.h)
    bool device_cuda_available(void);
    bool device_metal_available(void);
    bool device_rocm_available(void);

    // cml.h: Initialization and cleanup

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

    // cml.h: Tracking

    void cml_summary(Module* module);
    void cml_track_module(Module* module);
    void cml_untrack_module(Module* module);
    void cml_track_optimizer(Optimizer* optimizer);
    void cml_track_dataset(Dataset* dataset);

    // cml.h: Tensor creation - general

    Tensor* cml_empty(int* shape, int ndim, const TensorConfig* config);
    Tensor* cml_zeros(int* shape, int ndim, const TensorConfig* config);
    Tensor* cml_ones(int* shape, int ndim, const TensorConfig* config);
    Tensor* cml_full(int* shape, int ndim, const TensorConfig* config, float value);
    Tensor* cml_tensor(void* data, int* shape, int ndim, const TensorConfig* config);

    // cml.h: Tensor creation - 2D convenience

    Tensor* cml_zeros_2d(int rows, int cols);
    Tensor* cml_ones_2d(int rows, int cols);
    Tensor* cml_empty_2d(int rows, int cols);
    Tensor* cml_tensor_2d(const float* data, int rows, int cols);

    // cml.h: Tensor creation - 1D convenience

    Tensor* cml_zeros_1d(int size);
    Tensor* cml_ones_1d(int size);
    Tensor* cml_empty_1d(int size);
    Tensor* cml_tensor_1d(const float* data, int size);

    // cml.h: Element-wise operations

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

    // cml.h: Activation functions

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

    // cml.h: Reduction operations

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

    // cml.h: Linear algebra

    Tensor* cml_matmul(Tensor* a, Tensor* b);
    Tensor* cml_dot(Tensor* a, Tensor* b);
    QRResult cml_qr(Tensor* a);
    SVDResult cml_svd(Tensor* a);

    // cml.h: Shape manipulation

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
    Tensor* cml_scatter_reduce(Tensor* self, int dim, Tensor* index, Tensor* src,
                               ScatterReduceMode mode);
    Tensor* cml_interpolate(Tensor* a, int* output_size, int num_dims, InterpMode mode);

    // cml.h: Random / factory functions

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

    // cml.h: Type casting

    Tensor* cml_cast(Tensor* a, DType dtype);
    Tensor* cml_bitcast(Tensor* a, DType target_dtype);
    Tensor* cml_half(Tensor* a);
    Tensor* cml_double(Tensor* a);
    Tensor* cml_int_(Tensor* a);
    Tensor* cml_long(Tensor* a);
    Tensor* cml_short(Tensor* a);
    Tensor* cml_bool_(Tensor* a);
    Tensor* cml_bfloat16(Tensor* a);

    // cml.h: Weight initialization

    Tensor* cml_kaiming_uniform(int* shape, int ndim, int fan_in, const TensorConfig* config);
    Tensor* cml_kaiming_normal(int* shape, int ndim, int fan_in, const TensorConfig* config);
    Tensor* cml_glorot_uniform(int* shape, int ndim, int fan_in, int fan_out,
                               const TensorConfig* config);
    Tensor* cml_xavier_normal(int* shape, int ndim, int fan_in, int fan_out,
                              const TensorConfig* config);

    // cml.h: Neural network - Sequential

    Sequential* cml_nn_sequential(void);
    Sequential* cml_nn_sequential_add(Sequential* seq, Module* layer);
    Tensor* cml_nn_sequential_forward(Sequential* seq, Tensor* input);

    // cml.h: Neural network - Module operations

    Tensor* cml_nn_module_forward(Module* module, Tensor* input);
    void cml_nn_module_set_training(Module* module, bool training);
    bool cml_nn_module_is_training(Module* module);
    void cml_nn_module_eval(Module* module);
    void cml_nn_module_train(Module* module);

    // cml.h: Neural network - Layers

    Linear* cml_nn_linear(int in_features, int out_features, DType dtype, DeviceType device,
                          bool bias);
    ReLU* cml_nn_relu(bool inplace);
    Sigmoid* cml_nn_sigmoid(void);
    Tanh* cml_nn_tanh(void);
    LeakyReLU* cml_nn_leaky_relu(float negative_slope, bool inplace);
    Dropout* cml_nn_dropout(float p, bool inplace);

    Conv1d* cml_nn_conv1d(int in_channels, int out_channels, int kernel_size, int stride,
                          int padding, int dilation, bool use_bias, DType dtype, DeviceType device);
    Conv2d* cml_nn_conv2d(int in_channels, int out_channels, int kernel_size, int stride,
                          int padding, int dilation, bool bias, DType dtype, DeviceType device);
    Conv3d* cml_nn_conv3d(int in_channels, int out_channels, int kernel_size, int stride,
                          int padding, int dilation, bool use_bias, DType dtype, DeviceType device);
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
    GroupNorm* cml_nn_groupnorm(int num_groups, int num_channels, float eps, bool affine,
                                DType dtype, DeviceType device);

    Embedding* cml_nn_embedding(int num_embeddings, int embedding_dim, int padding_idx,
                                DType dtype, DeviceType device);
    Flatten* cml_nn_flatten(int start_dim, int end_dim);
    Identity* cml_nn_identity(void);
    PReLU* cml_nn_prelu(int num_parameters, float init, DType dtype, DeviceType device);
    ModuleList* cml_nn_module_list(void);
    ModuleDict* cml_nn_module_dict(void);

    // cml.h: Neural network - Pooling layers

    MaxPool1d* cml_nn_maxpool1d(int kernel_size, int stride, int padding, int dilation,
                                bool ceil_mode);
    MaxPool2d* cml_nn_maxpool2d(int kernel_size, int stride, int padding, int dilation,
                                bool ceil_mode);
    MaxPool3d* cml_nn_maxpool3d(int kernel_size, int stride, int padding, int dilation,
                                bool ceil_mode);
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

    // cml.h: Neural network - RNN layers

    RNNCell* cml_nn_rnn_cell(int input_size, int hidden_size, bool use_bias, DType dtype,
                             DeviceType device);
    LSTMCell* cml_nn_lstm_cell(int input_size, int hidden_size, bool use_bias, DType dtype,
                               DeviceType device);
    GRUCell* cml_nn_gru_cell(int input_size, int hidden_size, bool use_bias, DType dtype,
                             DeviceType device);
    RNN* cml_nn_rnn(int input_size, int hidden_size, int num_layers, bool bidirectional,
                    bool batch_first, float dropout, bool use_bias, DType dtype, DeviceType device);
    LSTM* cml_nn_lstm(int input_size, int hidden_size, int num_layers, bool bidirectional,
                      bool batch_first, float dropout, bool use_bias, DType dtype,
                      DeviceType device);
    GRU* cml_nn_gru(int input_size, int hidden_size, int num_layers, bool bidirectional,
                    bool batch_first, float dropout, bool use_bias, DType dtype, DeviceType device);

    // cml.h: Neural network - Transformer layers

    MultiHeadAttention* cml_nn_multihead_attention(int embed_dim, int num_heads, float dropout,
                                                   DType dtype, DeviceType device);
    TransformerEncoderLayer* cml_nn_transformer_encoder_layer(int d_model, int nhead,
                                                              int dim_feedforward, float dropout,
                                                              DType dtype, DeviceType device);
    TransformerEncoder* cml_nn_transformer_encoder(int d_model, int nhead, int dim_feedforward,
                                                    float dropout, int num_layers,
                                                    DType dtype, DeviceType device);
    TransformerDecoderLayer* cml_nn_transformer_decoder_layer(int d_model, int nhead,
                                                               int dim_feedforward, float dropout,
                                                               DType dtype, DeviceType device);
    TransformerDecoder* cml_nn_transformer_decoder(int d_model, int nhead, int dim_feedforward,
                                                    float dropout, int num_layers,
                                                    DType dtype, DeviceType device);

    // cml.h: Neural network - Upsampling / pixel shuffle

    Upsample* cml_nn_upsample(float scale_factor, const int* output_size, int num_output_dims,
                               UpsampleMode mode, bool align_corners);
    PixelShuffle* cml_nn_pixel_shuffle(int upscale_factor);
    PixelUnshuffle* cml_nn_pixel_unshuffle(int downscale_factor);

    Tensor* cml_f_interpolate(Tensor* input, int* output_size, int num_dims,
                               UpsampleMode mode, bool align_corners);
    Tensor* cml_f_pixel_shuffle(Tensor* input, int upscale_factor);
    Tensor* cml_f_pixel_unshuffle(Tensor* input, int downscale_factor);

    // cml.h: Optimizers

    Optimizer* cml_optim_adam(Parameter** parameters, int num_parameters, float lr,
                             float weight_decay, float beta1, float beta2, float eps);
    Optimizer* cml_optim_sgd(Parameter** parameters, int num_parameters, float lr, float momentum,
                             float weight_decay);
    Optimizer* cml_optim_rmsprop(Parameter** parameters, int num_parameters, float lr,
                                 float weight_decay, float alpha, float eps);
    Optimizer* cml_optim_adagrad(Parameter** parameters, int num_parameters, float lr,
                                 float weight_decay, float eps);
    Optimizer* cml_optim_adamw(Parameter** parameters, int num_parameters, float lr,
                               float weight_decay, float beta1, float beta2, float epsilon);
    Optimizer* cml_optim_nadam(Parameter** parameters, int num_parameters, float lr,
                               float weight_decay, float beta1, float beta2, float epsilon);
    Optimizer* cml_optim_adamax(Parameter** parameters, int num_parameters, float lr,
                                float weight_decay, float beta1, float beta2, float epsilon);
    Optimizer* cml_optim_adadelta(Parameter** parameters, int num_parameters, float rho,
                                   float weight_decay, float epsilon);
    Optimizer* cml_optim_lamb(Parameter** parameters, int num_parameters, float lr,
                              float weight_decay, float beta1, float beta2, float epsilon);
    Optimizer* cml_optim_lars(Parameter** parameters, int num_parameters, float lr, float momentum,
                              float weight_decay, float trust_coefficient);
    Optimizer* cml_optim_muon(Parameter** parameters, int num_parameters, float lr, float momentum,
                              float weight_decay, bool nesterov);
    Optimizer* cml_optim_adam_for_model(Module* model, float lr, float weight_decay, float beta1,
                                        float beta2, float eps);
    Optimizer* cml_optim_sgd_for_model(Module* model, float lr, float momentum, float weight_decay);
    void cml_optim_zero_grad(Optimizer* optimizer);
    void cml_optim_step(Optimizer* optimizer);

    // cml.h: Learning rate schedulers

    LRScheduler* cml_lr_scheduler_step(Optimizer* opt, int step_size, float gamma);
    LRScheduler* cml_lr_scheduler_reduce_on_plateau(Optimizer* opt, float factor, int patience,
                                                     float min_lr);
    LRScheduler* cml_lr_scheduler_exponential(Optimizer* opt, float gamma);
    LRScheduler* cml_lr_scheduler_cosine(Optimizer* opt, int T_max, float eta_min);
    LRScheduler* cml_lr_scheduler_one_cycle(Optimizer* opt, float max_lr, int total_steps,
                                             float pct_start, float div_factor,
                                             float final_div_factor);
    LRScheduler* cml_lr_scheduler_multi_step(Optimizer* opt, int* milestones, int num_milestones,
                                              float gamma);
    LRScheduler* cml_lr_scheduler_polynomial(Optimizer* opt, int total_iters, float power,
                                              float min_lr);
    LRScheduler* cml_lr_scheduler_warmup(LRScheduler* inner, int warmup_steps,
                                          float warmup_start_factor);
    float cml_lr_scheduler_update(LRScheduler* scheduler, float metric);
    float cml_lr_scheduler_get_lr(LRScheduler* scheduler);
    void cml_lr_scheduler_free(LRScheduler* scheduler);

    // cml.h: Loss functions

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

    // cml.h: Autograd (cml_ wrappers)

    void cml_backward(Tensor* tensor, Tensor* gradient, bool retain_graph, bool create_graph);
    void cml_zero_grad(Tensor* tensor);
    void cml_no_grad(void);
    void cml_enable_grad(void);
    bool cml_is_grad_enabled(void);
    bool cml_requires_grad(Tensor* t);
    void cml_set_requires_grad(Tensor* t, bool requires_grad);
    bool cml_is_leaf(Tensor* t);

    // cml.h: IR context management

    void cml_reset_ir_context(void);

    // cml.h: Kernel cache

    void cml_kernel_cache_clear(void);
    void cml_kernel_cache_stats(size_t* hits, size_t* misses, size_t* count, size_t* memory);
    double cml_kernel_cache_hit_rate(void);
    void cml_kernel_cache_print_stats(void);

    // cml.h: Automatic mixed precision (AMP)

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

    // cml.h: Sparse tensor operations

    SparseCOOData* cml_sparse_coo_tensor(Tensor* indices, Tensor* values,
                                          const int* dense_shape, int dense_ndim);
    SparseCOOData* cml_sparse_from_dense(Tensor* dense);
    Tensor* cml_sparse_to_dense(SparseCOOData* sparse, const TensorConfig* config);
    Tensor* cml_sparse_matmul(SparseCOOData* sparse, Tensor* dense);
    SparseCOOData* cml_sparse_coalesce(SparseCOOData* sparse);
    void cml_sparse_free(SparseCOOData* sparse);

    // cml.h: Serialization (GGUF)

    Tensor* cml_from_url(const char* url);
    GGUFContext* cml_gguf_open_read(const char* path);
    GGUFContext* cml_gguf_open_write(const char* path);
    void cml_gguf_close(GGUFContext* ctx);
    int cml_gguf_write_tensor(GGUFContext* ctx, const char* name, Tensor* tensor);
    Tensor* cml_gguf_read_tensor(GGUFContext* ctx, const char* name);
    int cml_module_save_gguf(Module* module, const char* path);
    int cml_module_load_gguf(Module* module, const char* path);

    // cml.h: Serialization (SafeTensors)

    SafeTensorsContext* cml_safetensors_open_read(const char* path);
    SafeTensorsContext* cml_safetensors_open_write(const char* path);
    void cml_safetensors_close(SafeTensorsContext* ctx);
    int cml_safetensors_write_tensor(SafeTensorsContext* ctx, const char* name, Tensor* tensor);
    Tensor* cml_safetensors_read_tensor(SafeTensorsContext* ctx, const char* name);
    int cml_module_save_safetensors(Module* module, const char* path);
    int cml_module_load_safetensors(Module* module, const char* path);

    void optimizer_set_lr(Optimizer* optimizer, float lr);
    void optimizer_set_group_lr(Optimizer* optimizer, int group_index, float lr);
    float optimizer_get_group_lr(Optimizer* optimizer, int group_index);
    void optimizer_set_grad_clip_norm(Optimizer* optimizer, float norm);
    void optimizer_set_amsgrad(Optimizer* optimizer, bool amsgrad);
    const char* optimizer_get_name(Optimizer* optimizer);

    int module_list_append(ModuleList* list, Module* module);
    int module_list_insert(ModuleList* list, int index, Module* module);
    int module_list_length(ModuleList* list);
    int module_dict_add(ModuleDict* dict, const char* key, Module* module);
    int module_dict_size(ModuleDict* dict);

    int model_load(Module* model, const char* filepath);

    typedef enum {
        DIST_BACKEND_NCCL = 0,
        DIST_BACKEND_MPI,
        DIST_BACKEND_GLOO,
        DIST_BACKEND_COUNT
    } DistBackendType;

    typedef enum {
        DIST_REDUCE_SUM = 0,
        DIST_REDUCE_PRODUCT,
        DIST_REDUCE_MAX,
        DIST_REDUCE_MIN,
        DIST_REDUCE_AVG
    } DistReduceOp;

    int cml_dist_init(DistBackendType backend, int world_size, int rank);
    void cml_dist_destroy(void);
    int cml_dist_get_rank(void);
    int cml_dist_get_world_size(void);
    bool cml_dist_is_initialized(void);
    int cml_dist_allreduce(Tensor* tensor, DistReduceOp op);
    int cml_dist_barrier(void);

    void* tensor_data_ptr(Tensor* t);
"""
)

ffi.set_source(
    "cml._cml_lib",
    """
    #include "cml.h"
    #include "tensor/tensor.h"
    """,
    libraries=["cml", "m", "dl", "pthread"],
    include_dirs=["../../include"],
    library_dirs=["../../lib", "../../build/lib"],
    extra_compile_args=["-std=c11", "-O2"],
    extra_link_args=[
        "-Wl,-rpath,$ORIGIN/../../lib",
        "-Wl,-rpath,$ORIGIN/../../build/lib",
    ],
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
