#ifndef C_ML_AUTOGRAD_H
#define C_ML_AUTOGRAD_H

#include <stddef.h>

// Operation types
typedef enum Operation
{
    OP_NONE = 0,
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_POW,
    OP_EXP,
    OP_LOG,
    OP_TANH,
    OP_SIGMOID,
    OP_RELU,
    OP_SOFTMAX,
    OP_ELU,
    OP_GELU,
    OP_LEAKY_RELU,
    OP_LINEAR
} Operation;

// Forward declarations of structs
typedef struct Function Function;
typedef struct Node Node;
typedef struct Hook Hook;
typedef struct TensorOptions TensorOptions;
typedef struct StorageImpl StorageImpl;
typedef struct TensorImpl TensorImpl;
typedef struct SavedVariable SavedVariable;
typedef struct GradContext GradContext;

// Core tensor operations
Node *tensor(float value, int requires_grad);
Node *tensor_with_options(float value, TensorOptions options);

// Basic arithmetic operations
Node *add(Node *a, Node *b);
Node *sub(Node *a, Node *b);
Node *mul(Node *a, Node *b);
Node *div(Node *a, Node *b);
Node *pow(Node *a, Node *b);
Node *exp(Node *x);
Node *log(Node *x);

// Matrix operations
// Remove transpose and cat functions since they are not activation functions

// Tensor creation and manipulation
Node *empty(int *sizes, int ndim);
Node *empty_like(Node *other);
Node *zeros_like(Node *other);
Node *ones_like(Node *other);
void resize_(Node *self, int *sizes, int ndim);
void set_(Node *self, Node *source);

// View operations
Node *as_strided(Node *self, int *shape, int *strides, int ndim);
Node *reshape(Node *self, int *shape, int ndim);
Node *view(Node *self, int *shape, int ndim);
Node *contiguous(Node *self);
Node *cat(Node **tensors, int n, int dim);

// Broadcasting support
Node *expand(Node *self, int *sizes, int ndim);
Node *broadcast_to(Node *self, int *sizes, int ndim);

// Gradient utilities
void set_requires_grad(Node *self, int requires_grad);
void acc_grad(Node *self, float grad);
int is_leaf(Node *self);
int is_contiguous(Node *self);
void set_checkpoint(Node *self, int enabled);

// Autograd state management
void save_for_backward(Node *self, Node *var);
SavedVariable *get_saved_variable(Node *self, int index);

// Struct definitions
struct Function
{
    void (*forward)(Node *result, Node **inputs, int ninputs);
    void (*backward)(float grad_output, Node **inputs, int ninputs);
    Node **saved_tensors;
    int num_saved;
};

struct TensorOptions
{
    int requires_grad;
    int retain_grad;
    float grad_scale;
};

typedef struct Node
{
    // Core data
    float value;
    float grad;

    // PyTorch-style attributes
    Function *grad_fn; // Records operation that created this node
    int requires_grad; // Whether to compute gradients
    int is_leaf;       // True if node was created by user
    int retain_grad;   // Keep gradients for non-leaf nodes

    // Graph tracking
    struct Node **next; // Forward edges
    struct Node **prev; // Backward edges
    int num_next;
    int num_prev;
    int ref_count;

    // Hook system
    Hook *backward_hooks;

    // Gradient accumulation
    int grad_accumulated;
    float grad_scale;

    // Additional PyTorch features
    int is_variable;       // PyTorch Variable compatibility
    TensorOptions options; // Tensor options
    struct Node *grad_acc; // Gradient accumulator
    int needs_reset;       // Flag for zero_grad behavior

    // PyTorch view system
    TensorImpl *tensor;         // Replace direct value storage
    int version_counter;        // Track modifications
    struct Node *autograd_meta; // For gradient tracking

    // View management
    struct Node *base; // Base tensor for views

    // Graph state
    int in_backward_pass;
} Node;

typedef struct Hook
{
    void (*fn)(struct Node *self, float grad);
    struct Hook *next;
} Hook;

typedef struct StorageImpl
{
    float *data;
    size_t size;
    int ref_count;
    int device_id; // For future GPU support
} StorageImpl;

typedef struct TensorImpl
{
    StorageImpl *storage;
    int *sizes;    // Dimensions
    int *strides;  // Stride for each dimension
    int ndim;      // Number of dimensions
    size_t offset; // Offset into storage
    int is_contiguous;
} TensorImpl;

typedef struct SavedVariable
{
    struct Node *tensor;
    int version;
} SavedVariable;

typedef struct GradContext
{
    int enabled;
    int grad_mode_stack[32];
    int stack_size;
} GradContext;

// Validation helper for activation functions
int validate_activation_input(float x);

// Activation helper functions
void create_activation_node(Node *output, Node *input, Operation op, Node *saved_var);

#endif
