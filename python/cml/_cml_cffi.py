"""
CFFI Definitions for CML (C Machine Learning Library)

This module defines the CFFI interface to the C-ML library.
It exposes the necessary C structures and functions for Python.
"""

from cffi import FFI

# Create FFI instance
ffi = FFI()

# Define C types and structures based on actual cml.h API
ffi.cdef(
    """
    // Type definitions
    typedef enum {
        DTYPE_FLOAT32,
        DTYPE_FLOAT64,
        DTYPE_INT32,
        DTYPE_INT64
    } DType;

    typedef enum {
        DEVICE_CPU,
        DEVICE_CUDA,
        DEVICE_METAL,
        DEVICE_ROCM
    } DeviceType;

    // Forward declarations for opaque types used in Tensor
    struct IRNode;
    struct CMLGraph;
    typedef struct CMLGraph* CMLGraph_t;
    struct CMLBackendBuffer;
    typedef struct CMLBackendBuffer* CMLBackendBuffer_t;

    // Structure declarations - Tensor exposes key fields for Python access
    typedef struct Tensor {
        int* shape;                   // Shape array
        int ndim;                     // Number of dimensions
        size_t numel;                 // Total number of elements
        DType dtype;                  // Data type
        DeviceType device;            // Device
        struct IRNode* ir_node;       // IR node (opaque)
        CMLGraph_t ir_context;           // IR context (opaque)
        bool is_executed;             // Has this been executed?
        void* data;                   // Data pointer (NULL until executed)
        bool owns_data;               // Owns data flag
        bool requires_grad;           // Requires gradient
        struct Tensor* grad;          // Gradient tensor
        int ref_count;                // Reference count
        struct Tensor* base;          // Base tensor for views
        size_t* strides;              // Stride array
        size_t storage_offset;        // Storage offset
        bool is_contiguous;           // Contiguous flag
        CMLBackendBuffer_t buffer_handle; // Backend buffer handle
        void* user_data;              // User data
    } Tensor;
    typedef struct Module Module;
    typedef struct Sequential Sequential;
    typedef struct Linear Linear;
    typedef struct Conv2d Conv2d;
    typedef struct ReLU ReLU;
    typedef struct Sigmoid Sigmoid;
    typedef struct Tanh Tanh;
    typedef struct Softmax Softmax;
    typedef struct Dropout Dropout;
    typedef struct BatchNorm2d BatchNorm2d;
    typedef struct LayerNorm LayerNorm;
    typedef struct MaxPool2d MaxPool2d;
    typedef struct AvgPool2d AvgPool2d;
    typedef struct Optimizer Optimizer;
    typedef struct Dataset Dataset;
    typedef struct Parameter Parameter;
    typedef struct TensorConfig TensorConfig;
    typedef struct CleanupContext CleanupContext;

    // Initialization and cleanup
    int cml_init(void);
    int cml_cleanup(void);
    bool cml_is_initialized(void);
    int cml_get_init_count(void);
    int cml_force_cleanup(void);
    void cml_get_version(int* major, int* minor, int* patch, const char** version_string);

    // Device management
    DeviceType cml_get_default_device(void);
    DType cml_get_default_dtype(void);
    void cml_set_default_dtype(DType dtype);

    // Tensor creation functions - 2D
    Tensor* cml_zeros_2d(int rows, int cols);
    Tensor* cml_ones_2d(int rows, int cols);
    Tensor* cml_empty_2d(int rows, int cols);
    Tensor* cml_tensor_2d(const float* data, int rows, int cols);

    // Tensor creation functions - 1D
    Tensor* cml_zeros_1d(int size);
    Tensor* cml_ones_1d(int size);
    Tensor* cml_empty_1d(int size);
    Tensor* cml_tensor_1d(const float* data, int size);

    // Tensor operations
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
    Tensor* cml_matmul(Tensor* a, Tensor* b);

    // Tensor activations
    Tensor* cml_relu(Tensor* a);
    Tensor* cml_sigmoid(Tensor* a);
    Tensor* cml_tanh(Tensor* a);
    Tensor* cml_softmax(Tensor* a, int dim);

    // Tensor reduction
    Tensor* cml_sum(Tensor* a, int dim, bool keepdim);
    Tensor* cml_mean(Tensor* a, int dim, bool keepdim);
    Tensor* cml_max(Tensor* a, int dim, bool keepdim);
    Tensor* cml_min(Tensor* a, int dim, bool keepdim);

    // Tensor manipulation
    Tensor* cml_transpose(Tensor* a, int dim1, int dim2);
    Tensor* cml_reshape(Tensor* a, int* new_shape, int new_ndim);
    Tensor* cml_clone(Tensor* a);
    Tensor* cml_detach(Tensor* a);

    // Tensor from tensor.h
    void tensor_free(Tensor* t);
    void* tensor_data_ptr(Tensor* t);
    float tensor_get_float(Tensor* t, size_t idx);
    void tensor_set_float(Tensor* t, size_t idx, float value);
    bool tensor_is_scalar(Tensor* t);
    bool tensor_is_contiguous(Tensor* t);
    Tensor* tensor_clone(Tensor* t);
    Tensor* tensor_from_data(void* data, int* shape, int ndim, void* config);
    int tensor_ensure_executed(Tensor* t);

    // Autograd
    void cml_backward(Tensor* tensor, Tensor* gradient, bool retain_graph, bool create_graph);
    void cml_zero_grad(Tensor* tensor);
    void cml_no_grad(void);
    void cml_enable_grad(void);
    bool cml_is_grad_enabled(void);
    bool cml_requires_grad(Tensor* t);
    void cml_set_requires_grad(Tensor* t, bool requires_grad);
    bool cml_is_leaf(Tensor* t);

    // Neural Network - Sequential
    Sequential* cml_nn_sequential(void);
    Sequential* cml_nn_sequential_add(Sequential* seq, Module* layer);
    Tensor* cml_nn_sequential_forward(Sequential* seq, Tensor* input);

    // Neural Network - Linear
    Linear* cml_nn_linear(int in_features, int out_features, DType dtype, DeviceType device, bool bias);

    // Neural Network - Activations
    ReLU* cml_nn_relu(bool inplace);
    Sigmoid* cml_nn_sigmoid(void);
    Tanh* cml_nn_tanh(void);

    // Neural Network - Regularization
    Dropout* cml_nn_dropout(float p, bool inplace);

    // Neural Network - Module operations
    Tensor* cml_nn_module_forward(Module* module, Tensor* input);
    void cml_nn_module_set_training(Module* module, bool training);
    bool cml_nn_module_is_training(Module* module);
    void cml_nn_module_eval(Module* module);
    void cml_nn_module_train(Module* module);
    void module_free(Module* module);

    // Loss functions
    Tensor* cml_nn_mse_loss(Tensor* input, Tensor* target);
    Tensor* cml_nn_mae_loss(Tensor* input, Tensor* target);
    Tensor* cml_nn_bce_loss(Tensor* input, Tensor* target);
    Tensor* cml_nn_cross_entropy_loss(Tensor* input, Tensor* target);
    Tensor* cml_nn_huber_loss(Tensor* input, Tensor* target, float delta);
    Tensor* cml_nn_kl_div_loss(Tensor* input, Tensor* target);

    // Optimizers
    Optimizer* cml_optim_adam_for_model(Module* model, float lr, float weight_decay, float beta1, float beta2, float epsilon);
    Optimizer* cml_optim_sgd_for_model(Module* model, float lr, float momentum, float weight_decay);

    // Optimizer operations
    void cml_optim_step(Optimizer* optimizer);
    void cml_optim_zero_grad(Optimizer* optimizer);
    void optimizer_free(Optimizer* opt);

    // Summary
    void cml_summary(Module* module);

    // Logging
    void cml_set_log_level(int level);
"""
)

# Set source files location
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
