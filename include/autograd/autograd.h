#ifndef CML_AUTOGRAD_H
#define CML_AUTOGRAD_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "core/logging.h"
#include "alloc/memory_management.h"
#include "tensor/tensor.h"
#include <stdbool.h>
#include <stddef.h>
#include <pthread.h>

struct Tensor;

typedef enum {
    OP_NONE = 0,

    // Binary operations
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_POW,
    OP_MATMUL,

    // Unary operations
    OP_NEG,
    OP_EXP,
    OP_LOG,
    OP_SQRT,
    OP_SIN,
    OP_COS,
    OP_TAN,
    OP_TANH,

    // Activations
    OP_RELU,
    OP_SIGMOID,
    OP_SOFTMAX,
    OP_LOG_SOFTMAX,
    OP_LEAKY_RELU,
    OP_ELU,
    OP_SELU,
    OP_SWISH,
    OP_MISH,
    OP_HARD_SWISH,
    OP_GELU,

    // Reductions
    OP_SUM,
    OP_MEAN,
    OP_MAX,
    OP_MIN,

    // Shape operations
    OP_TRANSPOSE,
    OP_RESHAPE,
    OP_VIEW,
    OP_PERMUTE,
    OP_SQUEEZE,
    OP_UNSQUEEZE,

    // Loss functions
    OP_MSE_LOSS,
    OP_MAE_LOSS,
    OP_BCE_LOSS,
    OP_CROSS_ENTROPY_LOSS,
    OP_HUBER_LOSS,
    OP_KL_DIV_LOSS,

    // Utility
    OP_CLONE,
    OP_DETACH,

    // Custom
    OP_CUSTOM
} OpType;

typedef struct AutogradEngine {
    bool enabled;
    bool grad_mode;         // Global gradient tracking flag
    bool anomaly_detection; // Check for NaN/Inf in gradients
    bool deterministic;     // Use deterministic algorithms

    // For higher-order derivatives
    bool create_graph;

    // Gradient accumulation
    bool accumulate_grad;

    // Thread safety
    pthread_mutex_t lock;
    bool lock_initialized;
} AutogradEngine;

extern AutogradEngine* global_autograd_engine;

typedef void (*TensorHookFn)(struct Tensor* grad);

void autograd_init(void);
void autograd_shutdown(void);
AutogradEngine* autograd_get_engine(void);
void autograd_set_grad_mode(bool enabled);
bool autograd_is_grad_enabled(void);
void autograd_no_grad_enter(void);
void autograd_no_grad_exit(void);
void tensor_backward(struct Tensor* tensor, struct Tensor* gradient, bool retain_graph,
                     bool create_graph);
void tensor_zero_grad(struct Tensor* tensor);
bool tensor_requires_grad(struct Tensor* t);
void tensor_set_requires_grad(struct Tensor* t, bool requires_grad);
bool tensor_is_leaf(struct Tensor* t);
struct Tensor* tensor_detach(struct Tensor* t);
void tensor_detach_inplace(struct Tensor* t);
void tensor_retain_grad(struct Tensor* t);

typedef void (*TensorBackwardHook)(struct Tensor* grad);
struct Module;
typedef void (*ModuleBackwardHook)(struct Module* module, struct Tensor* grad);

int tensor_register_backward_hook(struct Tensor* t, TensorBackwardHook hook);
void tensor_remove_hooks(struct Tensor* t);
int module_register_backward_hook(struct Module* module, ModuleBackwardHook hook);
void tensor_accumulate_grad(struct Tensor* tensor, struct Tensor* new_grad);
struct Tensor* tensor_get_grad(struct Tensor* tensor);
void tensor_register_hook(struct Tensor* tensor, TensorHookFn hook_fn);
#define DEFINE_UNARY_OP(name, op_type) struct Tensor* tensor_##name(struct Tensor* input);

#define DEFINE_BINARY_OP(name, op_type)                                                            \
    struct Tensor* tensor_##name(struct Tensor* a, struct Tensor* b);

bool tensor_can_broadcast_shapes(int* shape1, int ndim1, int* shape2, int ndim2);
int* broadcast_shapes(int* shape1, int ndim1, int* shape2, int ndim2, int* out_ndim);
int* broadcast_multi_shapes(int** shapes, int* ndims, int num_shapes, int* out_ndim);
void tensor_compute_grad_for_broadcast(struct Tensor* grad_output, int* original_shape, int ndim,
                                       struct Tensor** grad_input);
void autograd_check_anomaly(struct Tensor* tensor, const char* operation);
void autograd_set_anomaly_detection(bool enabled);
void autograd_print_graph(struct Tensor* tensor);
int autograd_export_json(struct Tensor* root, const char* path);

#endif // CML_AUTOGRAD_H
