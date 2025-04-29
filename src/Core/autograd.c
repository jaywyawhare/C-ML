#include "../../include/Core/autograd.h"
#include "../../include/Core/memory_management.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"
#include "../../include/Activations/elu.h"
#include "../../include/Activations/gelu.h"
#include "../../include/Activations/leaky_relu.h"
#include "../../include/Activations/linear.h"
#include "../../include/Activations/relu.h"
#include "../../include/Activations/sigmoid.h"
#include "../../include/Activations/softmax.h"
#include "../../include/Activations/tanh.h"
#include <math.h>
#include <stdio.h>

// Forward declarations for helper functions
static void accumulate_grad(Node *node, float grad);
static Function *create_function(Operation op);
static void add_edge(Node *from, Node *to);
static int *compute_strides(int *sizes, int ndim);

// Forward declarations of backward functions
static void add_backward(float grad_output, Node **inputs, int ninputs);
static void mul_backward(float grad_output, Node **inputs, int ninputs);
static void sub_backward(float grad_output, Node **inputs, int ninputs);
static void div_backward(float grad_output, Node **inputs, int ninputs);
static void pow_backward(float grad_output, Node **inputs, int ninputs);
static void exp_backward(float grad_output, Node **inputs, int ninputs);
static void log_backward(float grad_output, Node **inputs, int ninputs);

static GradContext grad_ctx = {1, {0}, 0};

void no_grad_push(void)
{
    if (grad_ctx.stack_size < 32)
    {
        grad_ctx.grad_mode_stack[grad_ctx.stack_size++] = grad_ctx.enabled;
        grad_ctx.enabled = 0;
    }
}

void no_grad_pop(void)
{
    if (grad_ctx.stack_size > 0)
    {
        grad_ctx.enabled = grad_ctx.grad_mode_stack[--grad_ctx.stack_size];
    }
}

static TensorImpl *create_tensor_impl(int *sizes, int ndim)
{
    TensorImpl *impl = (TensorImpl *)cm_safe_malloc(sizeof(TensorImpl), __FILE__, __LINE__);
    if (!impl)
        return NULL;

    // Setup storage and dimensions
    size_t total_size = 1;
    for (int i = 0; i < ndim; i++)
        total_size *= sizes[i];

    impl->storage = create_storage(total_size);
    impl->sizes = copy_sizes(sizes, ndim);
    impl->strides = compute_strides(sizes, ndim);
    impl->ndim = ndim;
    impl->offset = 0;
    impl->is_contiguous = 1;

    return impl;
}

Node *tensor(float value, int requires_grad)
{
    int size = 1;
    Node *result = empty(&size, 1);
    if (!result)
        return NULL;

    result->tensor->storage->data[0] = value;
    result->requires_grad = requires_grad && grad_ctx.enabled;
    result->is_leaf = 1;

    return result;
}

// PyTorch-style tensor creation with options
Node *tensor_with_options(float value, TensorOptions options)
{
    Node *result = tensor(value, options.requires_grad);
    if (!result)
        return NULL;

    result->retain_grad = options.retain_grad;
    result->grad_scale = options.grad_scale;
    result->options = options;
    return result;
}

Node *empty(int *sizes, int ndim)
{
    Node *result = (Node *)cm_safe_malloc(sizeof(Node), __FILE__, __LINE__);
    if (!result)
        return NULL;

    result->tensor = create_tensor_impl(sizes, ndim);
    result->requires_grad = 0;
    result->is_leaf = 1;
    result->version_counter = 0;
    return result;
}

Node *empty_like(Node *other)
{
    return empty(other->tensor->sizes, other->tensor->ndim);
}

void resize_(Node *self, int *sizes, int ndim)
{
    if (!self || !sizes)
        return;

    size_t new_size = 1;
    for (int i = 0; i < ndim; i++)
    {
        new_size *= sizes[i];
    }

    // Reallocate storage if needed
    if (new_size > self->tensor->storage->size)
    {
        StorageImpl *new_storage = create_storage(new_size);
        if (!new_storage)
            return;

        // Copy existing data
        memcpy(new_storage->data, self->tensor->storage->data,
               self->tensor->storage->size * sizeof(float));

        // Free old storage
        cm_safe_free((void **)&self->tensor->storage->data);
        cm_safe_free((void **)&self->tensor->storage);
        self->tensor->storage = new_storage;
    }

    // Update tensor metadata
    cm_safe_free((void **)&self->tensor->sizes);
    cm_safe_free((void **)&self->tensor->strides);
    self->tensor->sizes = copy_sizes(sizes, ndim);
    self->tensor->strides = compute_strides(sizes, ndim);
    self->tensor->ndim = ndim;
    self->version_counter++;
}

void set_(Node *self, Node *source)
{
    if (!self || !source)
        return;

    // Copy storage data
    memcpy(self->tensor->storage->data, source->tensor->storage->data,
           source->tensor->storage->size * sizeof(float));

    // Update metadata
    resize_(self, source->tensor->sizes, source->tensor->ndim);
}

int is_contiguous(Node *self)
{
    if (!self || !self->tensor)
        return 0;
    return self->tensor->is_contiguous;
}

// Broadcasting support
Node *expand(Node *self, int *sizes, int ndim)
{
    if (!self || !sizes)
        return NULL;

    // Create new tensor with expanded dimensions
    Node *result = empty(sizes, ndim);
    if (!result)
        return NULL;

    // Setup view into original data
    result->tensor->storage = self->tensor->storage;
    result->tensor->storage->ref_count++;
    result->tensor->is_contiguous = 0;
    result->base = self;

    // Calculate broadcast strides
    int *new_strides = compute_broadcast_strides(self->tensor->sizes,
                                                 self->tensor->strides,
                                                 self->tensor->ndim,
                                                 sizes, ndim);
    if (!new_strides)
    {
        cm_safe_free((void **)&result);
        return NULL;
    }

    cm_safe_free((void **)&result->tensor->strides);
    result->tensor->strides = new_strides;

    return result;
}

Node *broadcast_to(Node *self, int *sizes, int ndim)
{
    return expand(self, sizes, ndim);
}

// View operation implementations
Node *reshape(Node *self, int *shape, int ndim)
{
    if (!is_contiguous(self))
    {
        Node *cont = contiguous(self);
        if (!cont)
            return NULL;
        Node *result = reshape(cont, shape, ndim);
        cm_safe_free((void **)&cont);
        return result;
    }

    size_t new_size = 1;
    for (int i = 0; i < ndim; i++)
    {
        new_size *= shape[i];
    }

    if (new_size != self->tensor->storage->size)
    {
        return NULL; // Invalid reshape
    }

    return view(self, shape, ndim);
}

Node *view(Node *self, int *shape, int ndim)
{
    if (!is_contiguous(self))
    {
        Node *cont = contiguous(self);
        if (!cont)
            return NULL;
        Node *result = view(cont, shape, ndim);
        cm_safe_free((void **)&cont);
        return result;
    }

    Node *result = empty(shape, ndim);
    if (!result)
        return NULL;

    result->tensor->storage = self->tensor->storage;
    result->tensor->storage->ref_count++;
    result->base = self;
    result->requires_grad = self->requires_grad;

    return result;
}

Node *contiguous(Node *self)
{
    if (is_contiguous(self))
    {
        return self;
    }

    Node *result = empty(self->tensor->sizes, self->tensor->ndim);
    if (!result)
        return NULL;

    // Copy data to contiguous layout
    size_t total_size = self->tensor->storage->size;
    for (size_t i = 0; i < total_size; i++)
    {
        size_t src_idx = compute_index(i, self->tensor);
        result->tensor->storage->data[i] = self->tensor->storage->data[src_idx];
    }

    return result;
}

// Utility functions
void set_requires_grad(Node *self, int requires_grad)
{
    if (self->is_leaf)
    {
        self->requires_grad = requires_grad;
    }
}

void acc_grad(Node *self, float grad)
{
    if (!self->requires_grad)
        return;
    self->grad += grad;
    self->grad_accumulated = 1;
}

int is_leaf(Node *self)
{
    return self->is_leaf;
}

// Memory management helpers
static int *copy_sizes(int *sizes, int ndim)
{
    int *result = (int *)cm_safe_malloc(ndim * sizeof(int), __FILE__, __LINE__);
    if (result)
    {
        memcpy(result, sizes, ndim * sizeof(int));
    }
    return result;
}

static StorageImpl *create_storage(size_t size)
{
    StorageImpl *storage = (StorageImpl *)cm_safe_malloc(sizeof(StorageImpl),
                                                         __FILE__, __LINE__);
    if (!storage)
        return NULL;

    storage->data = (float *)cm_safe_malloc(size * sizeof(float),
                                            __FILE__, __LINE__);
    if (!storage->data)
    {
        cm_safe_free((void **)&storage);
        return NULL;
    }

    storage->size = size;
    storage->ref_count = 1;
    storage->device_id = 0;

    return storage;
}

// Tensor creation functions
Node *zeros_like(Node *other)
{
    Node *result = empty(other->tensor->sizes, other->tensor->ndim);
    if (!result)
        return NULL;

    memset(result->tensor->storage->data, 0,
           result->tensor->storage->size * sizeof(float));
    return result;
}

Node *ones_like(Node *other)
{
    Node *result = empty(other->tensor->sizes, other->tensor->ndim);
    if (!result)
        return NULL;

    for (size_t i = 0; i < result->tensor->storage->size; i++)
    {
        result->tensor->storage->data[i] = 1.0f;
    }
    return result;
}

// Matrix operations
Node *matmul(Node *a, Node *b)
{
    if (a->tensor->ndim != 2 || b->tensor->ndim != 2)
        return NULL;

    int m = a->tensor->sizes[0];
    int k = a->tensor->sizes[1];
    int n = b->tensor->sizes[1];

    if (k != b->tensor->sizes[0])
        return NULL;

    int out_sizes[2] = {m, n};
    Node *result = empty(out_sizes, 2);
    if (!result)
        return NULL;

    // Perform matrix multiplication
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            float sum = 0.0f;
            for (int p = 0; p < k; p++)
            {
                float a_val = a->tensor->storage->data[i * k + p];
                float b_val = b->tensor->storage->data[p * n + j];
                sum += a_val * b_val;
            }
            result->tensor->storage->data[i * n + j] = sum;
        }
    }

    return result;
}

Node *transpose(Node *x)
{
    if (x->tensor->ndim != 2)
        return NULL;

    int new_sizes[2] = {x->tensor->sizes[1], x->tensor->sizes[0]};
    int new_strides[2] = {x->tensor->strides[1], x->tensor->strides[0]};

    Node *result = as_strided(x, new_sizes, new_strides, 2);
    if (!result)
        return NULL;

    return result;
}

Node *cat(Node **tensors, int n, int dim)
{
    if (!tensors || n <= 0)
        return NULL;

    // Calculate new size along concatenation dimension
    int new_size = 0;
    for (int i = 0; i < n; i++)
    {
        new_size += tensors[i]->tensor->sizes[dim];
    }

    // Create output tensor with updated size
    int *new_sizes = copy_sizes(tensors[0]->tensor->sizes,
                                tensors[0]->tensor->ndim);
    new_sizes[dim] = new_size;

    Node *result = empty(new_sizes, tensors[0]->tensor->ndim);
    if (!result)
    {
        cm_safe_free((void **)&new_sizes);
        return NULL;
    }

    // Copy data from input tensors
    size_t offset = 0;
    for (int i = 0; i < n; i++)
    {
        size_t size = tensors[i]->tensor->storage->size;
        memcpy(result->tensor->storage->data + offset,
               tensors[i]->tensor->storage->data,
               size * sizeof(float));
        offset += size;
    }

    cm_safe_free((void **)&new_sizes);
    return result;
}

// Additional view operations
Node *as_strided(Node *self, int *shape, int *strides, int ndim)
{
    Node *result = empty(shape, ndim);
    if (!result)
        return NULL;

    // Share storage with input tensor
    result->tensor->storage = self->tensor->storage;
    result->tensor->storage->ref_count++;

    // Update metadata
    cm_safe_free((void **)&result->tensor->strides);
    result->tensor->strides = copy_sizes(strides, ndim);
    result->tensor->is_contiguous = 0;
    result->base = self;

    return result;
}

// Helper function to compute index in tensor storage
static size_t compute_index(size_t linear_idx, TensorImpl *tensor)
{
    size_t index = 0;
    size_t remaining = linear_idx;

    for (int i = 0; i < tensor->ndim; i++)
    {
        size_t coord = remaining / tensor->strides[i];
        remaining %= tensor->strides[i];
        index += coord * tensor->strides[i];
    }

    return index;
}

// Gradient accumulation helpers
static void accumulate_grad(Node *node, float grad)
{
    if (!node->grad_acc)
    {
        node->grad_acc = tensor(0.0f, 0);
    }
    node->grad_acc->value += grad;
}

// Function creation helpers
static Function *create_function(Operation op)
{
    Function *fn = (Function *)cm_safe_malloc(sizeof(Function), __FILE__, __LINE__);
    if (!fn)
        return NULL;

    fn->saved_tensors = NULL;
    fn->num_saved = 0;

    // Map operation to its backward function
    switch (op)
    {
    case OP_ADD:
        fn->backward = add_backward;
        break;
    case OP_MUL:
        fn->backward = mul_backward;
        break;
    case OP_SUB:
        fn->backward = sub_backward;
        break;
    case OP_DIV:
        fn->backward = div_backward;
        break;
    case OP_POW:
        fn->backward = pow_backward;
        break;
    case OP_EXP:
        fn->backward = exp_backward;
        break;
    case OP_LOG:
        fn->backward = log_backward;
        break;
    case OP_TANH:
        fn->backward = tanh_backward;
        break;
    case OP_RELU:
        fn->backward = relu_backward;
        break;
    case OP_SIGMOID:
        fn->backward = sigmoid_backward;
        break;
    case OP_SOFTMAX:
        fn->backward = softmax_backward;
        break;
    case OP_ELU:
        fn->backward = elu_backward;
        break;
    case OP_GELU:
        fn->backward = gelu_backward;
        break;
    case OP_LEAKY_RELU:
        fn->backward = leaky_relu_backward;
        break;
    case OP_LINEAR:
        fn->backward = linear_backward;
        break;
    default:
        cm_safe_free((void **)&fn);
        return NULL;
    }
    return fn;
}

static void execute_backward_function(Node *node)
{
    if (!node->grad_fn)
        return;
    node->grad_fn->backward(node->grad_acc->value, node->prev, node->num_prev);
}

// Build topological sort for backward pass
static void dfs_topo_sort(Node *node, int *visited, Node **result, int *size, int max_size)
{
    if (!node || *size >= max_size)
        return;

    visited[node->version_counter] = 1;

    // Visit all children first
    for (int i = 0; i < node->num_prev; i++)
    {
        if (!visited[node->prev[i]->version_counter])
        {
            dfs_topo_sort(node->prev[i], visited, result, size, max_size);
        }
    }

    // Add current node after all children
    result[(*size)++] = node;
}

static Node **build_topo(Node *root, int *size)
{
    if (!root || !size)
        return NULL;

    const int max_nodes = 1000; // Reasonable limit for graph size
    *size = 0;

    // Allocate result array and visited tracking
    Node **result = (Node **)cm_safe_malloc(max_nodes * sizeof(Node *),
                                            __FILE__, __LINE__);
    int *visited = (int *)calloc(max_nodes, sizeof(int));

    if (!result || !visited)
    {
        cm_safe_free((void **)&result);
        free(visited);
        return NULL;
    }

    // Perform DFS-based topological sort
    dfs_topo_sort(root, visited, result, size, max_nodes);

    free(visited);
    return result;
}

int validate_activation_input(float x)
{
    if (isnan(x) || isinf(x) || x == -INFINITY)
    {
        LOG_ERROR("Invalid input (NaN or Inf)");
        return CM_INVALID_INPUT_ERROR;
    }
    return 0;
}

void create_activation_node(Node *output, Node *input, Operation op, Node *saved_var)
{
    if (output && output->requires_grad)
    {
        output->grad_fn = create_function(op);
        add_edge(output, input);
        if (saved_var)
        {
            save_for_backward(output, saved_var);
        }
    }
}