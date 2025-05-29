#ifndef C_ML_AUTOGRAD_H
#define C_ML_AUTOGRAD_H

#include <stddef.h>
#include <stdbool.h>

// Operation types (PyTorch-like)
typedef enum Operation
{
    OP_NONE = 0,
    // Arithmetic operations
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_POW,
    OP_MATMUL,
    OP_NEG,
    OP_ABS,
    
    // Mathematical functions
    OP_EXP,
    OP_LOG,
    OP_SQRT,
    OP_SIN,
    OP_COS,
    OP_TAN,
    
    // Activation functions
    OP_TANH,
    OP_SIGMOID,
    OP_RELU,
    OP_LEAKY_RELU,
    OP_ELU,
    OP_GELU,
    OP_SOFTMAX,
    OP_LOG_SOFTMAX,
    OP_LINEAR,
    
    // Reduction operations
    OP_SUM,
    OP_MEAN,
    OP_MAX,
    OP_MIN,
    
    // Loss functions
    OP_MSE_LOSS,
    OP_BCE_LOSS,
    OP_CE_LOSS,
    
    // Tensor operations
    OP_RESHAPE,
    OP_TRANSPOSE,
    OP_SQUEEZE,
    OP_UNSQUEEZE,
    OP_CAT,
    OP_SPLIT,
    
    // Gradient operations
    OP_GRAD,
    OP_DETACH,
    OP_REQUIRES_GRAD
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

// Forward declarations of structs
typedef struct Function Function;
typedef struct Node Node;
typedef struct Hook Hook;
typedef struct TensorOptions TensorOptions;
typedef struct StorageImpl StorageImpl;
typedef struct TensorImpl TensorImpl;
typedef struct SavedVariable SavedVariable;
typedef struct GradContext GradContext;
typedef struct AutogradMeta AutogradMeta;

// PyTorch-like tensor creation functions
Node *tensor(float value, int requires_grad);
Node *tensor_from_data(float *data, int *sizes, int ndim, int requires_grad);
Node *tensor_with_options(float value, TensorOptions options);
Node *empty(int *sizes, int ndim);
Node *zeros(int *sizes, int ndim);
Node *ones(int *sizes, int ndim);
Node *randn(int *sizes, int ndim);
Node *rand_uniform(int *sizes, int ndim, float low, float high);
Node *tensor_from_array(float *data, int size, int requires_grad);

// Tensor manipulation
Node *empty_like(Node *other);
Node *zeros_like(Node *other);
Node *ones_like(Node *other);
void resize_(Node *self, int *sizes, int ndim);
void set_(Node *self, Node *source);
Node *clone(Node *self);
Node *detach(Node *self);

// Basic arithmetic operations (PyTorch-like)
Node *add(Node *a, Node *b);
Node *sub(Node *a, Node *b);
Node *mul(Node *a, Node *b);
Node *div_tensor(Node *a, Node *b);
Node *pow_tensor(Node *a, Node *b);
Node *matmul(Node *a, Node *b);
Node *neg(Node *a);
Node *abs_tensor(Node *a);

// Scalar operations
Node *add_scalar(Node *a, float scalar);
Node *sub_scalar(Node *a, float scalar);
Node *mul_scalar(Node *a, float scalar);
Node *div_scalar(Node *a, float scalar);

// Mathematical functions
Node *exp_tensor(Node *a);
Node *log_tensor(Node *a);
Node *sqrt_tensor(Node *a);
Node *sin_tensor(Node *a);
Node *cos_tensor(Node *a);
Node *tan_tensor(Node *a);

// Element-wise comparison and clamp functions
Node *max_elementwise(Node *a, Node *b);
Node *min_elementwise(Node *a, Node *b);
Node *clamp(Node *a, float min_val, float max_val);

// Activation functions (PyTorch-style)
Node *relu(Node *x);
Node *leaky_relu(Node *x, float negative_slope);
Node *sigmoid(Node *x);
Node *tanh_tensor(Node *x);
Node *elu(Node *x, float alpha);
Node *gelu(Node *x);
Node *softmax(Node *x, int dim);
Node *log_softmax(Node *x, int dim);

// Reduction operations
Node *sum_tensor(Node *x, int *dims, int ndims, int keepdim);
Node *mean_tensor(Node *x, int *dims, int ndims, int keepdim);
Node *max_tensor(Node *x, int dim, int keepdim);
Node *min_tensor(Node *x, int dim, int keepdim);

// Loss functions
Node *mse_loss(Node *input, Node *target, int reduction);
Node *binary_cross_entropy(Node *input, Node *target, Node *weight, int reduction);
Node *cross_entropy(Node *input, Node *target, Node *weight, int reduction);

// View operations
Node *reshape(Node *self, int *shape, int ndim);
Node *view(Node *self, int *shape, int ndim);
Node *transpose(Node *self, int dim0, int dim1);
Node *permute(Node *self, int *dims, int ndim);
Node *squeeze(Node *self, int dim);
Node *unsqueeze(Node *self, int dim);
Node *contiguous(Node *self);

// Concatenation and splitting
Node *cat(Node **tensors, int n, int dim);
Node **split(Node *tensor, int split_size, int dim, int *num_splits);

// Broadcasting support
Node *expand(Node *self, int *sizes, int ndim);
Node *broadcast_to(Node *self, int *sizes, int ndim);

// Gradient utilities (PyTorch-like)
void set_requires_grad(Node *self, int requires_grad);
void retain_grad(Node *self);
void zero_grad(Node *self);
void backward(Node *self, Node *gradient, int retain_graph, int create_graph);
int is_leaf(Node *self);
int is_contiguous(Node *self);

// Autograd context management
void enable_grad(void);
void disable_grad(void);
int is_grad_enabled(void);
void no_grad_push(void);
void no_grad_pop(void);

// Hook system (PyTorch-like)
int register_hook(Node *self, void (*hook_fn)(Node *self, Node *grad));
void remove_hook(Node *self, int hook_id);

// Advanced autograd features
void set_detect_anomaly(int enabled);
void autograd_set_grad_mode(int enabled);
void grad_set_checkpoint(Node *self, int enabled);

// Struct definitions
struct Function
{
    void (*forward)(Node *result, Node **inputs, int ninputs);
    void (*backward)(float grad_output, Node **inputs, int ninputs);
    Node **saved_tensors;
    int num_saved;
    
    // Input tracking for gradient computation
    Node **inputs;
    int ninputs;
    
    // Additional parameters for specific operations
    float alpha;      // For ELU, Leaky ReLU, etc.
    Operation op_type; // To track operation type
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
    struct Node ***next; // Changed from Node** to Node***
    struct Node **prev;  // Backward edges
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
    
    // Input nodes for activation functions and other operations
    struct Node **input_nodes;
    int num_inputs;
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

// Gradient accumulation helpers
void accumulate_grad(Node *node, float grad);

// Node creation helper
Node *create_node(void);
Node *create_node_with_value(float value, int requires_grad);

// Graph dependency helper
void set_graph_dependencies(Node *output, Node **inputs, int num_inputs);

// Add these declarations  
void backward_from_root(Node *root);
Node **build_topo(Node *root, int *size);
void execute_backward_function(Node *node);

#endif
