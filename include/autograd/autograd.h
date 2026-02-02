#ifndef CML_AUTOGRAD_H
#define CML_AUTOGRAD_H

#define _POSIX_C_SOURCE 200809L

#include "core/logging.h"
#include "alloc/memory_management.h"
#include "tensor/tensor.h"
#include <stdbool.h>
#include <stddef.h>

// Forward declarations
struct Tensor;

// Operation Types (used for auto-capture and unified API)
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
} AutogradEngine;

// Global autograd engine
extern AutogradEngine* global_autograd_engine;

typedef void (*TensorHookFn)(struct Tensor* grad);

/**
 * @brief Initialize the autograd engine
 */
void autograd_init(void);

/**
 * @brief Shutdown the autograd engine
 */
void autograd_shutdown(void);

/**
 * @brief Get the global autograd engine instance
 * @return Pointer to AutogradEngine
 */
AutogradEngine* autograd_get_engine(void);

/**
 * @brief Set gradient calculation mode
 * @param enabled Whether to enable gradient calculation
 */
void autograd_set_grad_mode(bool enabled);

/**
 * @brief Check if gradient calculation is enabled
 * @return true if enabled, false otherwise
 */
bool autograd_is_grad_enabled(void);

/**
 * @brief Enter no-grad context (disable gradient calculation)
 */
void autograd_no_grad_enter(void);

/**
 * @brief Exit no-grad context (restore previous state)
 */
void autograd_no_grad_exit(void);

/**
 * @brief Perform backward pass from a tensor
 * @param tensor Tensor to start backward pass from
 * @param gradient Gradient tensor (optional, defaults to 1.0)
 * @param retain_graph Whether to retain the graph for multiple backward passes
 * @param create_graph Whether to create a graph for higher-order derivatives
 */
void tensor_backward(struct Tensor* tensor, struct Tensor* gradient, bool retain_graph,
                     bool create_graph);

/**
 * @brief Zero gradients of a tensor
 * @param tensor Tensor to zero gradients for
 */
void tensor_zero_grad(struct Tensor* tensor);

/**
 * @brief Check if tensor requires gradients
 * @param t Input tensor
 * @return true if requires gradients, false otherwise
 */
bool tensor_requires_grad(struct Tensor* t);

/**
 * @brief Set whether tensor requires gradients
 * @param t Input tensor
 * @param requires_grad Whether to require gradients
 */
void tensor_set_requires_grad(struct Tensor* t, bool requires_grad);

/**
 * @brief Check if tensor is a leaf node
 * @param t Input tensor
 * @return true if leaf node, false otherwise
 */
bool tensor_is_leaf(struct Tensor* t);

/**
 * @brief Detach tensor from computation graph
 * @param t Input tensor
 * @return New detached tensor
 */
struct Tensor* tensor_detach(struct Tensor* t);

/**
 * @brief Detach tensor in-place
 * @param t Input tensor
 */
void tensor_detach_inplace(struct Tensor* t);

/**
 * @brief Retain gradient for non-leaf tensor
 * @param t Input tensor
 */
void tensor_retain_grad(struct Tensor* t);

typedef void (*TensorBackwardHook)(struct Tensor* grad);
// Forward declare Module for hook
struct Module;
typedef void (*ModuleBackwardHook)(struct Module* module, struct Tensor* grad);

/**
 * @brief Register backward hook on tensor
 * @param t Input tensor
 * @param hook Hook function
 * @return Hook ID (or negative on failure)
 */
int tensor_register_backward_hook(struct Tensor* t, TensorBackwardHook hook);

/**
 * @brief Remove all hooks from tensor
 * @param t Input tensor
 */
void tensor_remove_hooks(struct Tensor* t);

/**
 * @brief Register backward hook on module
 * @param module Input module
 * @param hook Hook function
 * @return Hook ID (or negative on failure)
 */
int module_register_backward_hook(struct Module* module, ModuleBackwardHook hook);

/**
 * @brief Accumulate gradient into tensor
 * @param tensor Target tensor
 * @param new_grad Gradient to accumulate
 */
void tensor_accumulate_grad(struct Tensor* tensor, struct Tensor* new_grad);

/**
 * @brief Get gradient of tensor
 * @param tensor Input tensor
 * @return Gradient tensor (or NULL if none)
 */
struct Tensor* tensor_get_grad(struct Tensor* tensor);

/**
 * @brief Register hook on tensor (generic)
 * @param tensor Input tensor
 * @param hook_fn Hook function
 */
void tensor_register_hook(struct Tensor* tensor, TensorHookFn hook_fn);
// Note: tensor_remove_hooks is declared above to avoid redundant declaration

#define DEFINE_UNARY_OP(name, op_type) struct Tensor* tensor_##name(struct Tensor* input);

#define DEFINE_BINARY_OP(name, op_type)                                                            \
    struct Tensor* tensor_##name(struct Tensor* a, struct Tensor* b);

/**
 * @brief Check if shapes can be broadcasted
 * @param shape1 First shape
 * @param ndim1 First number of dimensions
 * @param shape2 Second shape
 * @param ndim2 Second number of dimensions
 * @return true if broadcastable, false otherwise
 */
bool tensor_can_broadcast_shapes(int* shape1, int ndim1, int* shape2, int ndim2);

/**
 * @brief Compute broadcasted shape for two shapes
 * @param shape1 First shape
 * @param ndim1 First number of dimensions
 * @param shape2 Second shape
 * @param ndim2 Second number of dimensions
 * @param out_ndim Pointer to store output number of dimensions
 * @return Allocated broadcasted shape array (caller must free)
 */
int* broadcast_shapes(int* shape1, int ndim1, int* shape2, int ndim2, int* out_ndim);

/**
 * @brief Compute broadcasted shape for multiple shapes
 * @param shapes Array of shape arrays
 * @param ndims Array of dimension counts
 * @param num_shapes Number of shapes
 * @param out_ndim Pointer to store output number of dimensions
 * @return Allocated broadcasted shape array (caller must free)
 */
int* broadcast_multi_shapes(int** shapes, int* ndims, int num_shapes, int* out_ndim);

/**
 * @brief Compute gradients for broadcasted operation
 * @param grad_output Output gradient
 * @param original_shape Original input shape
 * @param ndim Original input dimensions
 * @param grad_input Pointer to store computed input gradient
 */
void tensor_compute_grad_for_broadcast(struct Tensor* grad_output, int* original_shape, int ndim,
                                       struct Tensor** grad_input);

/**
 * @brief Check for anomalies (NaN/Inf) in tensor
 * @param tensor Tensor to check
 * @param operation Operation name for reporting
 */
void autograd_check_anomaly(struct Tensor* tensor, const char* operation);

/**
 * @brief Enable/disable anomaly detection
 * @param enabled Whether to enable anomaly detection
 */
void autograd_set_anomaly_detection(bool enabled);

/**
 * @brief Print computation graph
 * @param tensor Root tensor
 */
void autograd_print_graph(struct Tensor* tensor);

/**
 * @brief Export computation graph to JSON
 * @param root Root tensor
 * @param path File path
 * @return 0 on success, negative on failure
 */
int autograd_export_json(struct Tensor* root, const char* path);

#endif // CML_AUTOGRAD_H
