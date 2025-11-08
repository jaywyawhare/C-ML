#ifndef CML_AUTOGRAD_H
#define CML_AUTOGRAD_H

#define _POSIX_C_SOURCE 200809L

#include "Core/logging.h"
#include "Core/memory_management.h"
#include "tensor/tensor.h"
#include <stdbool.h>
#include <stddef.h>

// Autograd System

// Forward declarations
struct Tensor;
struct Function;
struct AutogradContext;

// Operation Types

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

// Backward Context for Saving Tensors

typedef struct AutogradContext {
    struct Tensor** saved_tensors;
    int num_saved_tensors;
    int capacity;

    // Saved values (scalars, etc.)
    void* saved_data;
    size_t saved_data_size;

    // For shape operations
    int* saved_shape;
    int saved_ndim;

    // For custom operations
    void* custom_data;
    void (*custom_free_fn)(void*);
} AutogradContext;

// Backward function signature
typedef void (*BackwardFn)(struct Function* fn, struct Tensor* grad_output);

// Function struct: represents an operation in the computation graph
typedef struct Function {
    OpType op_type;
    char* op_name;

    // Parent tensors (inputs to this operation)
    struct Tensor** inputs;
    int num_inputs;

    // Context for backward pass
    AutogradContext* ctx;

    // Backward function
    BackwardFn backward_fn;

    // For graph traversal
    int sequence_nr;
    bool needs_input_grad[8]; // Which inputs need gradients (max 8 inputs)

    // Reference counting
    int ref_count;
} Function;

// Autograd Engine (Tape-based)

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

// Hook System

typedef void (*TensorHookFn)(struct Tensor* grad);
typedef void (*BackwardHookFn)(struct Function* fn, struct Tensor* grad_input,
                               struct Tensor* grad_output);

typedef struct TensorHook {
    TensorHookFn fn;
    struct TensorHook* next;
} TensorHook;

// Core Autograd Functions

// Engine management
void autograd_init(void);
void autograd_shutdown(void);
AutogradEngine* autograd_get_engine(void);

// Gradient mode context
void autograd_set_grad_mode(bool enabled);
bool autograd_is_grad_enabled(void);
void autograd_no_grad_enter(void);
void autograd_no_grad_exit(void);

// Backward pass
void tensor_backward(struct Tensor* tensor, struct Tensor* gradient, bool retain_graph,
                     bool create_graph);
void tensor_zero_grad(struct Tensor* tensor);

// Gradient utilities
bool tensor_requires_grad(struct Tensor* t);
void tensor_set_requires_grad(struct Tensor* t, bool requires_grad);
bool tensor_is_leaf(struct Tensor* t);
struct Tensor* tensor_detach(struct Tensor* t);
void tensor_detach_(struct Tensor* t);
void tensor_retain_grad(struct Tensor* t);

// Context Management

AutogradContext* autograd_context_create(void);
void autograd_context_free(AutogradContext* ctx);
void autograd_context_save_for_backward(AutogradContext* ctx, struct Tensor** tensors,
                                        int num_tensors);
void autograd_context_save_shape(AutogradContext* ctx, int* shape, int ndim);
void autograd_context_save_data(AutogradContext* ctx, void* data, size_t size);
struct Tensor* autograd_context_get_saved_tensor(AutogradContext* ctx, int index);

// Function Management

Function* autograd_function_create(OpType op_type, const char* name);
void autograd_function_free(Function* fn);
void autograd_function_set_backward(Function* fn, BackwardFn backward_fn);
void autograd_function_set_inputs(Function* fn, struct Tensor** inputs, int num_inputs);
void autograd_function_mark_dirty(Function* fn, struct Tensor* tensor);

// Apply function and create computation graph edge
struct Tensor* autograd_apply_function(Function* fn, struct Tensor** outputs, int num_outputs);

// Backward Pass Implementation

// Topological sort for backward pass
typedef struct BackwardNode {
    struct Tensor* tensor;
    struct Tensor* grad;
    Function* grad_fn;
    int depth;
    bool visited;
    struct BackwardNode* next;
} BackwardNode;

typedef struct BackwardGraph {
    BackwardNode** nodes;
    int num_nodes;
    int capacity;
} BackwardGraph;

BackwardGraph* backward_graph_create(void);
void backward_graph_free(BackwardGraph* graph);
void backward_graph_add_node(BackwardGraph* graph, struct Tensor* tensor, struct Tensor* grad);
void backward_graph_execute(BackwardGraph* graph, bool retain_graph);

// Gradient Accumulation

void tensor_accumulate_grad(struct Tensor* tensor, struct Tensor* new_grad);
struct Tensor* tensor_get_grad(struct Tensor* tensor);

// Hook Management

void tensor_register_hook(struct Tensor* tensor, TensorHookFn hook_fn);
void tensor_remove_hooks(struct Tensor* tensor);
void function_register_hook(Function* fn, BackwardHookFn hook_fn);

// Operation Registration Helpers

// Helper macros for creating operations
#define DEFINE_UNARY_OP(name, op_type) struct Tensor* tensor_##name(struct Tensor* input);

#define DEFINE_BINARY_OP(name, op_type)                                                            \
    struct Tensor* tensor_##name(struct Tensor* a, struct Tensor* b);

// Broadcasting and Shape Utilities

bool can_broadcast_shapes(int* shape1, int ndim1, int* shape2, int ndim2);
int* broadcast_shapes(int* shape1, int ndim1, int* shape2, int ndim2, int* out_ndim);
int* broadcast_multi_shapes(int** shapes, int* ndims, int num_shapes, int* out_ndim);
void compute_grad_for_broadcast(struct Tensor* grad_output, int* original_shape, int ndim,
                                struct Tensor** grad_input);

// Error Handling

void autograd_check_anomaly(struct Tensor* tensor, const char* operation);
void autograd_set_anomaly_detection(bool enabled);

// Utility Functions

const char* op_type_to_string(OpType op);
void autograd_print_graph(struct Tensor* tensor);
void autograd_visualize_backward_graph(struct Tensor* tensor);

int autograd_export_json(struct Tensor* root, const char* path);

// Advanced Features

// Double backward (higher-order derivatives)
void tensor_backward_backward(struct Tensor* tensor);

// Gradient checkpointing
struct Tensor* checkpoint_forward(Function* fn, struct Tensor** inputs, int num_inputs);

// Jacobian-vector product
struct Tensor* jacobian_vector_product(struct Tensor* output, struct Tensor* input,
                                       struct Tensor* vector);

// Vector-jacobian product (used in backward pass)
struct Tensor* vector_jacobian_product(struct Tensor* output, struct Tensor* input,
                                       struct Tensor* vector);

// Forward Operation Declarations (implemented in forward_ops.c)

// Binary operations
struct Tensor* tensor_add(struct Tensor* a, struct Tensor* b);
struct Tensor* tensor_sub(struct Tensor* a, struct Tensor* b);
struct Tensor* tensor_mul(struct Tensor* a, struct Tensor* b);
struct Tensor* tensor_div(struct Tensor* a, struct Tensor* b);
struct Tensor* tensor_pow(struct Tensor* a, struct Tensor* b);

// Unary operations
struct Tensor* tensor_neg(struct Tensor* a);
struct Tensor* tensor_exp(struct Tensor* a);
struct Tensor* tensor_log(struct Tensor* a);
struct Tensor* tensor_sqrt(struct Tensor* a);
struct Tensor* tensor_sin(struct Tensor* a);
struct Tensor* tensor_cos(struct Tensor* a);
struct Tensor* tensor_tan(struct Tensor* a);
struct Tensor* tensor_tanh(struct Tensor* a);

// Activation functions
struct Tensor* tensor_relu(struct Tensor* a);
struct Tensor* tensor_sigmoid(struct Tensor* a);
struct Tensor* tensor_leaky_relu(struct Tensor* a, float negative_slope);

// Reduction operations
struct Tensor* tensor_sum(struct Tensor* a, int dim, bool keepdim);
struct Tensor* tensor_mean(struct Tensor* a, int dim, bool keepdim);

// Matrix operations
struct Tensor* tensor_transpose(struct Tensor* a, int dim0, int dim1);
struct Tensor* tensor_matmul(struct Tensor* a, struct Tensor* b);

// Loss functions have been moved to autograd/loss_functions.h
// Include that header for loss function declarations

#endif // CML_AUTOGRAD_H
