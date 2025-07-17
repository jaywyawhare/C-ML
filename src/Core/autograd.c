#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../include/Core/autograd.h"
#include "../../include/Core/memory_management.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"
#include <stdint.h>
// Activation headers removed to avoid conflicts with autograd function declarations

// Forward declarations of static functions
static void add_edge(Node *from, Node *to);
static int *compute_strides(int *sizes, int ndim);
static int *compute_broadcast_strides(int *src_sizes, int *src_strides,
                                      int src_ndim, int *dst_sizes, int dst_ndim);
static StorageImpl *create_storage(size_t size);
static int *copy_sizes(int *sizes, int ndim);
static size_t compute_index(size_t linear_idx, TensorImpl *tensor);
static void dfs_topo_sort(Node *node, int *visited, Node **result, int *size, int max_size);
static Node *as_strided(Node *self, int *shape, int *strides, int ndim);

// Backward pass function declarations
static void add_backward(float grad_output, Node **inputs, int ninputs);
static void mul_backward(float grad_output, Node **inputs, int ninputs);
static void sub_backward(float grad_output, Node **inputs, int ninputs);
static void div_backward(float grad_output, Node **inputs, int ninputs);
static void pow_backward(float grad_output, Node **inputs, int ninputs);
static void exp_backward(float grad_output, Node **inputs, int ninputs);
static void log_backward(float grad_output, Node **inputs, int ninputs);
static void neg_backward(float grad_output, Node **inputs, int ninputs);
static void abs_backward(float grad_output, Node **inputs, int ninputs);
static void sqrt_backward(float grad_output, Node **inputs, int ninputs);
static void max_backward(float grad_output, Node **inputs, int ninputs);
static void min_backward(float grad_output, Node **inputs, int ninputs);
static void tanh_backward(float grad_output, Node **inputs, int ninputs);
static void relu_backward(float grad_output, Node **inputs, int ninputs);
static void sigmoid_backward(float grad_output, Node **inputs, int ninputs);
static void softmax_backward(float grad_output, Node **inputs, int ninputs);
static void elu_backward(float grad_output, Node **inputs, int ninputs);
static void gelu_backward(float grad_output, Node **inputs, int ninputs);
static void leaky_relu_backward(float grad_output, Node **inputs, int ninputs);
static void linear_backward(float grad_output, Node **inputs, int ninputs);

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

// Helper for dtype size
size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DTYPE_FLOAT32: return sizeof(float);
        case DTYPE_FLOAT64: return sizeof(double);
        case DTYPE_INT32:   return sizeof(int32_t);
        case DTYPE_INT64:   return sizeof(int64_t);
        case DTYPE_BOOL:    return sizeof(uint8_t);
        default: return sizeof(float);
    }
}

// Update StorageImpl, TensorImpl, Node creation
static StorageImpl *create_storage_typed(size_t size, DType dtype, DeviceType device) {
    StorageImpl *storage = (StorageImpl *)cm_safe_malloc(sizeof(StorageImpl), __FILE__, __LINE__);
    if (!storage) return NULL;
    storage->size = size;
    storage->dtype = dtype;
    storage->device = device;
    storage->data = cm_safe_malloc(size * dtype_size(dtype), __FILE__, __LINE__);
    storage->ref_count = 1;
    storage->device_id = 0;
    return storage;
}

static TensorImpl *create_tensor_impl_typed(int *sizes, int ndim, DType dtype, DeviceType device) {
    TensorImpl *impl = (TensorImpl *)cm_safe_malloc(sizeof(TensorImpl), __FILE__, __LINE__);
    if (!impl) return NULL;
    size_t total_size = 1;
    for (int i = 0; i < ndim; i++) total_size *= sizes[i];
    impl->storage = create_storage_typed(total_size, dtype, device);
    impl->sizes = copy_sizes(sizes, ndim);
    impl->strides = compute_strides(sizes, ndim);
    impl->ndim = ndim;
    impl->offset = 0;
    impl->is_contiguous = 1;
    impl->dtype = dtype;
    impl->device = device;
    return impl;
}

// Update existing functions to use default dtype/device
Node *tensor(float value, int requires_grad) {
    TensorOptions options = {requires_grad, 0, 1.0f, DTYPE_FLOAT32, DEVICE_CPU};
    return tensor_with_options(value, options);
}

// PyTorch-style tensor creation with options
Node *tensor_with_options(float value, TensorOptions options)
{
    Node *result = empty_typed(&(int){1}, 1, options.dtype, options.device);
    if (!result) return NULL;
    if (options.dtype == DTYPE_FLOAT32) ((float *)result->tensor->storage->data)[0] = value;
    else if (options.dtype == DTYPE_FLOAT64) ((double *)result->tensor->storage->data)[0] = value;
    result->requires_grad = options.requires_grad;
    result->retain_grad = options.retain_grad;
    result->grad_scale = options.grad_scale;
    result->dtype = options.dtype;
    result->device = options.device;
    result->options = options;
    return result;
}

Node *empty(int *sizes, int ndim)
{
    return empty_typed(sizes, ndim, DTYPE_FLOAT32, DEVICE_CPU);
}

Node *empty_like(Node *other)
{
    return empty(other->tensor->sizes, other->tensor->ndim);
}

// Additional tensor creation functions
Node *zeros(int *sizes, int ndim)
{
    Node *result = empty_typed(sizes, ndim, DTYPE_FLOAT32, DEVICE_CPU);
    if (!result)
        return NULL;
    
    size_t total_size = 1;
    for (int i = 0; i < ndim; i++)
        total_size *= sizes[i];
    
    memset(result->tensor->storage->data, 0, total_size * sizeof(float));
    return result;
}

Node *ones(int *sizes, int ndim)
{
    Node *result = empty_typed(sizes, ndim, DTYPE_FLOAT32, DEVICE_CPU);
    if (!result)
        return NULL;
    
    size_t total_size = 1;
    for (int i = 0; i < ndim; i++)
        total_size *= sizes[i];
    
    for (size_t i = 0; i < total_size; i++)
        ((float *)result->tensor->storage->data)[i] = 1.0f;
    
    return result;
}

Node *zeros_like(Node *other)
{
    return zeros(other->tensor->sizes, other->tensor->ndim);
}

Node *ones_like(Node *other)
{
    return ones(other->tensor->sizes, other->tensor->ndim);
}

Node *tensor_from_array(float *data, int size, int requires_grad)
{
    Node *result = empty(&size, 1);
    if (!result)
        return NULL;
    
    memcpy(result->tensor->storage->data, data, size * sizeof(float));
    result->requires_grad = requires_grad && grad_ctx.enabled;
    result->is_leaf = 1;
    
    return result;
}

Node *tensor_from_data(float *data, int *sizes, int ndim, int requires_grad)
{
    Node *result = empty(sizes, ndim);
    if (!result)
        return NULL;
    
    size_t total_size = 1;
    for (int i = 0; i < ndim; i++)
        total_size *= sizes[i];
    
    memcpy(result->tensor->storage->data, data, total_size * sizeof(float));
    result->requires_grad = requires_grad && grad_ctx.enabled;
    result->is_leaf = 1;
    
    return result;
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
Node *tensor_randn(int m, int n)
{
    int sizes[2] = {m, n};
    Node *result = empty(sizes, 2);
    if (!result)
        return NULL;

    for (size_t i = 0; i < result->tensor->storage->size; i++)
    {
        // Simple random initialization between -1 and 1
        result->tensor->storage->data[i] = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
    }
    return result;
}

Node *tensor_zeros(int size)
{
    Node *result = empty(&size, 1);
    if (!result)
        return NULL;

    memset(result->tensor->storage->data, 0,
           result->tensor->storage->size * sizeof(float));
    return result;
}

Node *tensor_ones(int size)
{
    Node *result = empty(&size, 1);
    if (!result)
        return NULL;

    for (size_t i = 0; i < result->tensor->storage->size; i++)
    {
        result->tensor->storage->data[i] = 1.0f;
    }
    return result;
}

Node *zeros_typed(int *sizes, int ndim, DType dtype, DeviceType device) {
    Node *result = empty_typed(sizes, ndim, dtype, device);
    if (!result) return NULL;
    size_t total_size = 1;
    for (int i = 0; i < ndim; i++) total_size *= sizes[i];
    memset(result->tensor->storage->data, 0, total_size * dtype_size(dtype));
    return result;
}

Node *ones_typed(int *sizes, int ndim, DType dtype, DeviceType device) {
    Node *result = empty_typed(sizes, ndim, dtype, device);
    if (!result) return NULL;
    size_t total_size = 1;
    for (int i = 0; i < ndim; i++) total_size *= sizes[i];
    
    switch (dtype) {
        case DTYPE_FLOAT32:
            for (size_t i = 0; i < total_size; i++)
                ((float *)result->tensor->storage->data)[i] = 1.0f;
            break;
        case DTYPE_FLOAT64:
            for (size_t i = 0; i < total_size; i++)
                ((double *)result->tensor->storage->data)[i] = 1.0f;
            break;
        case DTYPE_INT32:
            for (size_t i = 0; i < total_size; i++)
                ((int32_t *)result->tensor->storage->data)[i] = 1;
            break;
        case DTYPE_INT64:
            for (size_t i = 0; i < total_size; i++)
                ((int64_t *)result->tensor->storage->data)[i] = 1;
            break;
        case DTYPE_BOOL:
            for (size_t i = 0; i < total_size; i++)
                ((uint8_t *)result->tensor->storage->data)[i] = 1;
            break;
    }
    return result;
}

Node *tensor_from_data_typed(void *data, int *sizes, int ndim, int requires_grad, DType dtype, DeviceType device) {
    Node *result = empty_typed(sizes, ndim, dtype, device);
    if (!result) return NULL;
    
    size_t total_size = 1;
    for (int i = 0; i < ndim; i++) total_size *= sizes[i];
    
    memcpy(result->tensor->storage->data, data, total_size * dtype_size(dtype));
    result->requires_grad = requires_grad && grad_ctx.enabled;
    result->is_leaf = 1;
    result->dtype = dtype;
    result->device = device;
    
    return result;
}

// Helper function to get value as float (for backward compatibility)
static float get_value_as_float(Node *node) {
    if (!node || !node->tensor || !node->tensor->storage) return 0.0f;
    
    switch (node->dtype) {
        case DTYPE_FLOAT32: return ((float *)node->tensor->storage->data)[0];
        case DTYPE_FLOAT64: return (float)((double *)node->tensor->storage->data)[0];
        case DTYPE_INT32:   return (float)((int32_t *)node->tensor->storage->data)[0];
        case DTYPE_INT64:   return (float)((int64_t *)node->tensor->storage->data)[0];
        case DTYPE_BOOL:    return (float)((uint8_t *)node->tensor->storage->data)[0];
        default: return 0.0f;
    }
}

// Helper function to set value from float
static void set_value_from_float(Node *node, float value) {
    if (!node || !node->tensor || !node->tensor->storage) return;
    
    switch (node->dtype) {
        case DTYPE_FLOAT32: ((float *)node->tensor->storage->data)[0] = value; break;
        case DTYPE_FLOAT64: ((double *)node->tensor->storage->data)[0] = (double)value; break;
        case DTYPE_INT32:  ((int32_t *)node->tensor->storage->data)[0] = (int32_t)value; break;
        case DTYPE_INT64:  ((int64_t *)node->tensor->storage->data)[0] = (int64_t)value; break;
        case DTYPE_BOOL:    ((uint8_t *)node->tensor->storage->data)[0] = (uint8_t)(value != 0); break;
    }
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

Node *transpose(Node *self, int dim0, int dim1)
{
    if (!self || !self->tensor || self->tensor->ndim < 2)
        return NULL;

    // For simplicity, handle 2D case where dim0=0, dim1=1 
    if (self->tensor->ndim != 2 || dim0 != 0 || dim1 != 1)
        return NULL;

    int new_sizes[2] = {self->tensor->sizes[1], self->tensor->sizes[0]};
    int new_strides[2] = {self->tensor->strides[1], self->tensor->strides[0]};

    Node *result = as_strided(self, new_sizes, new_strides, 2);
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
static Node *as_strided(Node *self, int *shape, int *strides, int ndim)
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

// Helper function to set up computational graph dependencies
void set_graph_dependencies(Node *result, Node **inputs, int ninputs)
{
    if (!result || !inputs || ninputs <= 0)
        return;
        
    result->prev = (Node **)cm_safe_malloc(ninputs * sizeof(Node *), __FILE__, __LINE__);
    if (!result->prev)
        return;
        
    for (int i = 0; i < ninputs; i++) {
        result->prev[i] = inputs[i];
    }
    result->num_prev = ninputs;
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
void accumulate_grad(Node *node, float grad)
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
    fn->inputs = NULL;
    fn->ninputs = 0;
    fn->op_type = op;

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
    case OP_NEG:
        fn->backward = neg_backward;
        break;
    case OP_ABS:
        fn->backward = abs_backward;
        break;
    case OP_SQRT:
        fn->backward = sqrt_backward;
        break;
    case OP_MAX:
        fn->backward = max_backward;
        break;
    case OP_MIN:
        fn->backward = min_backward;
        break;
    default:
        cm_safe_free((void **)&fn);
        return NULL;
    }
    return fn;
}

// Helper function to set function inputs
static void set_function_inputs(Function *fn, Node **inputs, int ninputs)
{
    if (!fn || !inputs || ninputs <= 0)
        return;
        
    fn->inputs = (Node **)cm_safe_malloc(ninputs * sizeof(Node *), __FILE__, __LINE__);
    if (!fn->inputs)
        return;
        
    fn->ninputs = ninputs;
    for (int i = 0; i < ninputs; i++) {
        fn->inputs[i] = inputs[i];
    }
}

// Helper function to save variables for backward pass
static void save_for_backward(Node *output, Node *var)
{
    if (!output || !output->grad_fn || !var)
        return;
        
    output->grad_fn->saved_tensors = (Node **)cm_safe_malloc(sizeof(Node *), __FILE__, __LINE__);
    if (!output->grad_fn->saved_tensors)
        return;
        
    output->grad_fn->saved_tensors[0] = var;
    output->grad_fn->num_saved = 1;
}

void execute_backward_function(Node *node)
{
    if (!node->grad_fn)
        return;
    node->grad_fn->backward(node->grad_acc->value, node->prev, node->num_prev);
}

void backward_from_root(Node *root)
{
    int size;
    Node **topo = build_topo(root, &size);
    if (!topo)
        return;

    // Initialize root gradient
    if (!root->grad_acc) {
        root->grad_acc = tensor(1.0f, 0);
    }

    for (int i = size - 1; i >= 0; i--)
    {
        execute_backward_function(topo[i]);
    }
    cm_safe_free((void **)&topo);
}

Node **build_topo(Node *root, int *size)
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

// Node creation helper
Node *create_node(void)
{
    Node *node = (Node *)cm_safe_malloc(sizeof(Node), __FILE__, __LINE__);
    if (!node)
        return NULL;
    
    // Initialize with scalar tensor (size 1)
    int size = 1;
    node->tensor = create_tensor_impl(&size, 1);
    if (!node->tensor) {
        cm_safe_free((void **)&node);
        return NULL;
    }
    
    // Initialize Node fields
    node->grad = 0.0f;
    node->requires_grad = 0;
    node->is_leaf = 1;
    node->retain_grad = 0;
    node->grad_fn = NULL;
    node->next = NULL;
    node->prev = NULL;
    node->num_next = 0;
    node->num_prev = 0;
    node->ref_count = 1;
    node->backward_hooks = NULL;
    node->grad_accumulated = 0;
    node->grad_scale = 1.0f;
    node->is_variable = 0;
    node->grad_acc = NULL;
    node->needs_reset = 0;
    node->version_counter = 0;
    node->autograd_meta = NULL;
    node->base = NULL;
    node->in_backward_pass = 0;
    node->input_nodes = NULL;
    node->num_inputs = 0;
    
    return node;
}

// Helper function to create node with specific value
Node *create_node_with_value(float value, int requires_grad)
{
    return tensor(value, requires_grad);
}

// Add implementation of static functions
static void add_edge(Node *from, Node *to)
{
    if (!from || !to)
        return;

    // Allocate or resize next array
    if (!from->next)
    {
        from->next = cm_safe_malloc(sizeof(Node **), __FILE__, __LINE__);
        from->num_next = 0;
    }
    else
    {
        from->next = cm_safe_realloc(from->next, (from->num_next + 1) * sizeof(Node **),
                                     __FILE__, __LINE__);
    }

    from->next[from->num_next++] = &to;
    to->prev[to->num_prev++] = from;
}

static int *compute_strides(int *sizes, int ndim)
{
    int *strides = cm_safe_malloc(ndim * sizeof(int), __FILE__, __LINE__);
    if (!strides)
        return NULL;

    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--)
    {
        strides[i] = stride;
        stride *= sizes[i];
    }
    return strides;
}

static int *compute_broadcast_strides(int *src_sizes, int *src_strides,
                                      int src_ndim, int *dst_sizes, int dst_ndim)
{
    int *strides = cm_safe_malloc(dst_ndim * sizeof(int), __FILE__, __LINE__);
    if (!strides)
        return NULL;

    int offset = dst_ndim - src_ndim;
    for (int i = 0; i < dst_ndim; i++)
    {
        if (i < offset)
        {
            strides[i] = 0;
        }
        else
        {
            int src_i = i - offset;
            strides[i] = (src_sizes[src_i] == 1) ? 0 : src_strides[src_i];
        }
    }
    return strides;
}

// Implement backward functions
static void add_backward(float grad_output, Node **inputs, int ninputs)
{
    if (ninputs != 2)
        return;
    if (inputs[0] && inputs[0]->requires_grad)
        accumulate_grad(inputs[0], grad_output);
    if (inputs[1] && inputs[1]->requires_grad)
        accumulate_grad(inputs[1], grad_output);
}

static void mul_backward(float grad_output, Node **inputs, int ninputs)
{
    if (ninputs != 2)
        return;
    if (inputs[0] && inputs[0]->requires_grad)
        accumulate_grad(inputs[0], grad_output * inputs[1]->tensor->storage->data[0]);
    if (inputs[1] && inputs[1]->requires_grad)
        accumulate_grad(inputs[1], grad_output * inputs[0]->tensor->storage->data[0]);
}

static void sub_backward(float grad_output, Node **inputs, int ninputs)
{
    if (ninputs != 2)
        return;
    if (inputs[0] && inputs[0]->requires_grad)
        accumulate_grad(inputs[0], grad_output);
    if (inputs[1] && inputs[1]->requires_grad)
        accumulate_grad(inputs[1], -grad_output);
}

static void div_backward(float grad_output, Node **inputs, int ninputs)
{
    if (ninputs != 2)
        return;
    float b = inputs[1]->tensor->storage->data[0];
    if (inputs[0] && inputs[0]->requires_grad)
        accumulate_grad(inputs[0], grad_output / b);
    if (inputs[1] && inputs[1]->requires_grad)
        accumulate_grad(inputs[1], -grad_output * inputs[0]->tensor->storage->data[0] / (b * b));
}

static void pow_backward(float grad_output, Node **inputs, int ninputs)
{
    if (ninputs != 2)
        return;
    if (inputs[0] && inputs[0]->requires_grad)
        accumulate_grad(inputs[0], grad_output * inputs[1]->tensor->storage->data[0] * logf(inputs[0]->tensor->storage->data[0]));
    if (inputs[1] && inputs[1]->requires_grad)
        accumulate_grad(inputs[1], grad_output * inputs[0]->tensor->storage->data[0] * logf(inputs[1]->tensor->storage->data[0]));
}

static void exp_backward(float grad_output, Node **inputs, int ninputs)
{
    if (ninputs != 1)
        return;
    if (inputs[0] && inputs[0]->requires_grad)
        accumulate_grad(inputs[0], grad_output * expf(inputs[0]->tensor->storage->data[0]));
}

static void log_backward(float grad_output, Node **inputs, int ninputs)
{
    if (ninputs != 1)
        return;
    if (inputs[0] && inputs[0]->requires_grad)
        accumulate_grad(inputs[0], grad_output / inputs[0]->tensor->storage->data[0]);
}

static void relu_backward(float grad_output, Node **inputs, int ninputs)
{
    if (ninputs != 1 || !inputs[0])
        return;
    if (inputs[0]->requires_grad) {
        float input_val = inputs[0]->tensor->storage->data[0];
        float grad = input_val > 0.0f ? grad_output : 0.0f;
        accumulate_grad(inputs[0], grad);
    }
}

static void sigmoid_backward(float grad_output, Node **inputs, int ninputs)
{
    if (ninputs != 1 || !inputs[0])
        return;
    if (inputs[0]->requires_grad) {
        float input_val = inputs[0]->tensor->storage->data[0];
        float sigmoid_val = 1.0f / (1.0f + expf(-input_val));
        float grad = grad_output * sigmoid_val * (1.0f - sigmoid_val);
        accumulate_grad(inputs[0], grad);
    }
}

static void tanh_backward(float grad_output, Node **inputs, int ninputs)
{
    if (ninputs != 1 || !inputs[0])
        return;
    if (inputs[0]->requires_grad) {
        float input_val = inputs[0]->tensor->storage->data[0];
        float tanh_val = tanhf(input_val);
        float grad = grad_output * (1.0f - tanh_val * tanh_val);
        accumulate_grad(inputs[0], grad);
    }
}

static void softmax_backward(float grad_output, Node **inputs, int ninputs)
{
    // Simplified softmax backward for single element
    if (ninputs != 1 || !inputs[0])
        return;
    if (inputs[0]->requires_grad) {
        accumulate_grad(inputs[0], grad_output);
    }
}

static void elu_backward(float grad_output, Node **inputs, int ninputs)
{
    if (ninputs != 1 || !inputs[0])
        return;
    if (inputs[0]->requires_grad) {
        float input_val = inputs[0]->tensor->storage->data[0];
        float alpha = 1.0f; // Default alpha
        float grad = input_val >= 0.0f ? grad_output : grad_output * alpha * expf(input_val);
        accumulate_grad(inputs[0], grad);
    }
}

static void gelu_backward(float grad_output, Node **inputs, int ninputs)
{
    if (ninputs != 1 || !inputs[0])
        return;
    if (inputs[0]->requires_grad) {
        float x = inputs[0]->tensor->storage->data[0];
        const float sqrt_2_over_pi = 0.7978845608f;
        float term = sqrt_2_over_pi * (x + 0.044715f * x * x * x);
        float tanh_term = tanhf(term);
        float grad = grad_output * 0.5f * (1.0f + tanh_term + x * (sqrt_2_over_pi * (1.0f - tanh_term * tanh_term) * (1.0f + 0.134145f * x * x)));
        accumulate_grad(inputs[0], grad);
    }
}

static void leaky_relu_backward(float grad_output, Node **inputs, int ninputs)
{
    if (ninputs != 1 || !inputs[0])
        return;
    if (inputs[0]->requires_grad) {
        float input_val = inputs[0]->tensor->storage->data[0];
        float negative_slope = 0.01f; // Default slope
        float grad = input_val > 0.0f ? grad_output : grad_output * negative_slope;
        accumulate_grad(inputs[0], grad);
    }
}

static void linear_backward(float grad_output, Node **inputs, int ninputs)
{
    if (ninputs != 1 || !inputs[0])
        return;
    if (inputs[0]->requires_grad) {
        accumulate_grad(inputs[0], grad_output);
    }
}

static void neg_backward(float grad_output, Node **inputs, int ninputs)
{
    if (ninputs != 1 || !inputs[0])
        return;
    if (inputs[0]->requires_grad) {
        accumulate_grad(inputs[0], -grad_output);
    }
}

static void abs_backward(float grad_output, Node **inputs, int ninputs)
{
    if (ninputs != 1 || !inputs[0])
        return;
    if (inputs[0]->requires_grad) {
        float input_val = inputs[0]->tensor->storage->data[0];
        float grad = input_val >= 0.0f ? grad_output : -grad_output;
        accumulate_grad(inputs[0], grad);
    }
}

static void sqrt_backward(float grad_output, Node **inputs, int ninputs)
{
    if (ninputs != 1 || !inputs[0])
        return;
    if (inputs[0]->requires_grad) {
        float input_val = inputs[0]->tensor->storage->data[0];
        float grad = grad_output / (2.0f * sqrtf(input_val));
        accumulate_grad(inputs[0], grad);
    }
}

static void max_backward(float grad_output, Node **inputs, int ninputs)
{
    if (ninputs < 1 || !inputs[0])
        return;
    
    if (ninputs == 1) {
        // For clamp operation
        if (inputs[0]->requires_grad) {
            accumulate_grad(inputs[0], grad_output);
        }
    } else if (ninputs == 2) {
        // For max_elementwise
        if (inputs[0] && inputs[1]) {
            float a_val = inputs[0]->tensor->storage->data[0];
            float b_val = inputs[1]->tensor->storage->data[0];
            
            if (inputs[0]->requires_grad) {
                float grad = (a_val >= b_val) ? grad_output : 0.0f;
                accumulate_grad(inputs[0], grad);
            }
            if (inputs[1]->requires_grad) {
                float grad = (b_val > a_val) ? grad_output : 0.0f;
                accumulate_grad(inputs[1], grad);
            }
        }
    }
}

static void min_backward(float grad_output, Node **inputs, int ninputs)
{
    if (ninputs != 2 || !inputs[0] || !inputs[1])
        return;
        
    float a_val = inputs[0]->tensor->storage->data[0];
    float b_val = inputs[1]->tensor->storage->data[0];
    
    if (inputs[0]->requires_grad) {
        float grad = (a_val <= b_val) ? grad_output : 0.0f;
        accumulate_grad(inputs[0], grad);
    }
    if (inputs[1]->requires_grad) {
        float grad = (b_val < a_val) ? grad_output : 0.0f;
        accumulate_grad(inputs[1], grad);
    }
}

// Basic arithmetic operations (PyTorch-like)
Node *add(Node *a, Node *b)
{
    if (!a || !b) return NULL;
    
    // For now, assume both operands have the same dtype
    DType result_dtype = a->dtype;
    DeviceType result_device = a->device;
    
    float a_val = get_value_as_float(a);
    float b_val = get_value_as_float(b);
    float result_val = a_val + b_val;
    
    Node *result = tensor_with_options(result_val, (TensorOptions){
        a->requires_grad || b->requires_grad, 0, result_dtype, result_device
    });
    
    if (result && result->requires_grad) {
        result->grad_fn = create_function(OP_ADD);
        if (result->grad_fn) {
            result->grad_fn->inputs = (Node **)cm_safe_malloc(2 * sizeof(Node *), __FILE__, __LINE__);
            if (result->grad_fn->inputs) {
                result->grad_fn->inputs[0] = a;
                result->grad_fn->inputs[1] = b;
                result->grad_fn->ninputs = 2;
            }
        }
    }
    
    return result;
}

Node *sub(Node *a, Node *b)
{
    if (!a || !b) return NULL;
    
    // For now, assume both operands have the same dtype
    DType result_dtype = a->dtype;
    DeviceType result_device = a->device;
    
    float a_val = get_value_as_float(a);
    float b_val = get_value_as_float(b);
    float result_val = a_val - b_val;
    
    Node *result = tensor_with_options(result_val, (TensorOptions){
        a->requires_grad || b->requires_grad, 0, result_dtype, result_device
    });
    
    if (result && result->requires_grad) {
        result->grad_fn = create_function(OP_SUB);
        if (result->grad_fn) {
            result->grad_fn->inputs = (Node **)cm_safe_malloc(2 * sizeof(Node *), __FILE__, __LINE__);
            if (result->grad_fn->inputs) {
                result->grad_fn->inputs[0] = a;
                result->grad_fn->inputs[1] = b;
                result->grad_fn->ninputs = 2;
            }
        }
    }
    
    return result;
}

Node *mul(Node *a, Node *b)
{
    if (!a || !b) return NULL;
    
    // For now, assume both operands have the same dtype
    DType result_dtype = a->dtype;
    DeviceType result_device = a->device;
    
    float a_val = get_value_as_float(a);
    float b_val = get_value_as_float(b);
    float result_val = a_val * b_val;
    
    Node *result = tensor_with_options(result_val, (TensorOptions){
        a->requires_grad || b->requires_grad, 0, result_dtype, result_device
    });
    
    if (result && result->requires_grad) {
        result->grad_fn = create_function(OP_MUL);
        if (result->grad_fn) {
            result->grad_fn->inputs = (Node **)cm_safe_malloc(2 * sizeof(Node *), __FILE__, __LINE__);
            if (result->grad_fn->inputs) {
                result->grad_fn->inputs[0] = a;
                result->grad_fn->inputs[1] = b;
                result->grad_fn->ninputs = 2;
            }
        }
    }
    
    return result;
}

Node *div_tensor(Node *a, Node *b)
{
    if (!a || !b) return NULL;
    
    // For now, assume both operands have the same dtype
    DType result_dtype = a->dtype;
    DeviceType result_device = a->device;
    
    float a_val = get_value_as_float(a);
    float b_val = get_value_as_float(b);
    float result_val = a_val / b_val;
    
    Node *result = tensor_with_options(result_val, (TensorOptions){
        a->requires_grad || b->requires_grad, 0, result_dtype, result_device
    });
    
    if (result && result->requires_grad) {
        result->grad_fn = create_function(OP_DIV);
        if (result->grad_fn) {
            result->grad_fn->inputs = (Node **)cm_safe_malloc(2 * sizeof(Node *), __FILE__, __LINE__);
            if (result->grad_fn->inputs) {
                result->grad_fn->inputs[0] = a;
                result->grad_fn->inputs[1] = b;
                result->grad_fn->ninputs = 2;
            }
        }
    }
    
    return result;
}

Node *pow_tensor(Node *a, Node *b)
{
    if (!a || !b) return NULL;
    
    // For now, assume both operands have the same dtype
    DType result_dtype = a->dtype;
    DeviceType result_device = a->device;
    
    float a_val = get_value_as_float(a);
    float b_val = get_value_as_float(b);
    float result_val = powf(a_val, b_val);
    
    Node *result = tensor_with_options(result_val, (TensorOptions){
        a->requires_grad || b->requires_grad, 0, result_dtype, result_device
    });
    
    if (result && result->requires_grad) {
        result->grad_fn = create_function(OP_POW);
        if (result->grad_fn) {
            result->grad_fn->inputs = (Node **)cm_safe_malloc(2 * sizeof(Node *), __FILE__, __LINE__);
            if (result->grad_fn->inputs) {
                result->grad_fn->inputs[0] = a;
                result->grad_fn->inputs[1] = b;
                result->grad_fn->ninputs = 2;
            }
        }
    }
    
    return result;
}

Node *neg(Node *a)
{
    if (!a) return NULL;
    
    // For now, assume a has the same dtype and device
    DType result_dtype = a->dtype;
    DeviceType result_device = a->device;
    
    float a_val = get_value_as_float(a);
    float result_val = -a_val;
    
    Node *result = tensor_with_options(result_val, (TensorOptions){
        a->requires_grad, 0, result_dtype, result_device
    });
    
    if (result && result->requires_grad) {
        result->grad_fn = create_function(OP_NEG);
        if (result->grad_fn) {
            result->grad_fn->inputs = (Node **)cm_safe_malloc(sizeof(Node *), __FILE__, __LINE__);
            if (result->grad_fn->inputs) {
                result->grad_fn->inputs[0] = a;
                result->grad_fn->ninputs = 1;
            }
        }
    }
    
    return result;
}

Node *abs_tensor(Node *a)
{
    if (!a) return NULL;
    
    // For now, assume a has the same dtype and device
    DType result_dtype = a->dtype;
    DeviceType result_device = a->device;
    
    float a_val = get_value_as_float(a);
    float result_val = fabsf(a_val);
    
    Node *result = tensor_with_options(result_val, (TensorOptions){
        a->requires_grad, 0, result_dtype, result_device
    });
    
    if (result && result->requires_grad) {
        result->grad_fn = create_function(OP_ABS);
        if (result->grad_fn) {
            result->grad_fn->inputs = (Node **)cm_safe_malloc(sizeof(Node *), __FILE__, __LINE__);
            if (result->grad_fn->inputs) {
                result->grad_fn->inputs[0] = a;
                result->grad_fn->ninputs = 1;
            }
        }
    }
    
    return result;
}

// Mathematical functions
Node *exp_tensor(Node *a)
{
    if (!a) return NULL;
    
    // For now, assume a has the same dtype and device
    DType result_dtype = a->dtype;
    DeviceType result_device = a->device;
    
    float a_val = get_value_as_float(a);
    float result_val = expf(a_val);
    
    Node *result = tensor_with_options(result_val, (TensorOptions){
        a->requires_grad, 0, result_dtype, result_device
    });
    
    if (result && result->requires_grad) {
        result->grad_fn = create_function(OP_EXP);
        if (result->grad_fn) {
            result->grad_fn->inputs = (Node **)cm_safe_malloc(sizeof(Node *), __FILE__, __LINE__);
            if (result->grad_fn->inputs) {
                result->grad_fn->inputs[0] = a;
                result->grad_fn->ninputs = 1;
            }
        }
    }
    
    return result;
}

Node *log_tensor(Node *a)
{
    if (!a) return NULL;
    
    // For now, assume a has the same dtype and device
    DType result_dtype = a->dtype;
    DeviceType result_device = a->device;
    
    float a_val = get_value_as_float(a);
    float result_val = logf(a_val);
    
    Node *result = tensor_with_options(result_val, (TensorOptions){
        a->requires_grad, 0, result_dtype, result_device
    });
    
    if (result && result->requires_grad) {
        result->grad_fn = create_function(OP_LOG);
        if (result->grad_fn) {
            result->grad_fn->inputs = (Node **)cm_safe_malloc(sizeof(Node *), __FILE__, __LINE__);
            if (result->grad_fn->inputs) {
                result->grad_fn->inputs[0] = a;
                result->grad_fn->ninputs = 1;
            }
        }
    }
    
    return result;
}

Node *sqrt_tensor(Node *a)
{
    if (!a) return NULL;
    
    // For now, assume a has the same dtype and device
    DType result_dtype = a->dtype;
    DeviceType result_device = a->device;
    
    float a_val = get_value_as_float(a);
    float result_val = sqrtf(a_val);
    
    Node *result = tensor_with_options(result_val, (TensorOptions){
        a->requires_grad, 0, result_dtype, result_device
    });
    
    if (result && result->requires_grad) {
        result->grad_fn = create_function(OP_SQRT);
        if (result->grad_fn) {
            result->grad_fn->inputs = (Node **)cm_safe_malloc(sizeof(Node *), __FILE__, __LINE__);
            if (result->grad_fn->inputs) {
                result->grad_fn->inputs[0] = a;
                result->grad_fn->ninputs = 1;
            }
        }
    }
    
    return result;
}

// Element-wise comparison and clamp functions
Node *max_elementwise(Node *a, Node *b)
{
    if (!a || !b) return NULL;
    
    // For now, assume both operands have the same dtype
    DType result_dtype = a->dtype;
    DeviceType result_device = a->device;
    
    float a_val = get_value_as_float(a);
    float b_val = get_value_as_float(b);
    float result_val = fmaxf(a_val, b_val);
    
    Node *result = tensor_with_options(result_val, (TensorOptions){
        a->requires_grad || b->requires_grad, 0, result_dtype, result_device
    });
    
    if (result && result->requires_grad) {
        result->grad_fn = create_function(OP_MAX);
        if (result->grad_fn) {
            result->grad_fn->inputs = (Node **)cm_safe_malloc(2 * sizeof(Node *), __FILE__, __LINE__);
            if (result->grad_fn->inputs) {
                result->grad_fn->inputs[0] = a;
                result->grad_fn->inputs[1] = b;
                result->grad_fn->ninputs = 2;
            }
        }
    }
    
    return result;
}

Node *min_elementwise(Node *a, Node *b)
{
    if (!a || !b) return NULL;
    
    // For now, assume both operands have the same dtype
    DType result_dtype = a->dtype;
    DeviceType result_device = a->device;
    
    float a_val = get_value_as_float(a);
    float b_val = get_value_as_float(b);
    float result_val = fminf(a_val, b_val);
    
    Node *result = tensor_with_options(result_val, (TensorOptions){
        a->requires_grad || b->requires_grad, 0, result_dtype, result_device
    });
    
    if (result && result->requires_grad) {
        result->grad_fn = create_function(OP_MIN);
        if (result->grad_fn) {
            result->grad_fn->inputs = (Node **)cm_safe_malloc(2 * sizeof(Node *), __FILE__, __LINE__);
            if (result->grad_fn->inputs) {
                result->grad_fn->inputs[0] = a;
                result->grad_fn->inputs[1] = b;
                result->grad_fn->ninputs = 2;
            }
        }
    }
    
    return result;
}

Node *clamp(Node *a, float min_val, float max_val)
{
    if (!a) return NULL;
    
    // For now, assume a has the same dtype and device
    DType result_dtype = a->dtype;
    DeviceType result_device = a->device;
    
    float a_val = get_value_as_float(a);
    float result_val = fmaxf(min_val, fminf(max_val, a_val));
    
    Node *result = tensor_with_options(result_val, (TensorOptions){
        a->requires_grad, 0, result_dtype, result_device
    });
    
    if (result && result->requires_grad) {
        result->grad_fn = create_function(OP_MAX); // Use MAX op for clamp
        if (result->grad_fn) {
            result->grad_fn->inputs = (Node **)cm_safe_malloc(sizeof(Node *), __FILE__, __LINE__);
            if (result->grad_fn->inputs) {
                result->grad_fn->inputs[0] = a;
                // Note: min_val and max_val are not Nodes, so they are not added to grad_fn->inputs
                // The backward pass for clamp (max_backward) needs to handle this.
                // For simplicity, we are only tracking 'a' as input here.
                // A more complete solution might involve saving min_val and max_val in grad_fn or handling them differently.
                result->grad_fn->ninputs = 1;
            }
        }
    }
    
    return result;
}

// Scalar operations
Node *add_scalar(Node *a, float scalar)
{
    if (!a) return NULL;
    Node *scalar_node = tensor(scalar, 0);
    return add(a, scalar_node);
}

Node *sub_scalar(Node *a, float scalar)
{
    if (!a) return NULL;
    Node *scalar_node = tensor(scalar, 0);
    return sub(a, scalar_node);
}

Node *mul_scalar(Node *a, float scalar)
{
    if (!a) return NULL;
    Node *scalar_node = tensor(scalar, 0);
    return mul(a, scalar_node);
}

Node *div_scalar(Node *a, float scalar)
{
    if (!a) return NULL;
    Node *scalar_node = tensor(scalar, 0);
    return div_tensor(a, scalar_node);
}

// Activation functions
Node *relu(Node *x)
{
    if (!x) return NULL;
    
    // For now, assume x has the same dtype and device
    DType result_dtype = x->dtype;
    DeviceType result_device = x->device;
    
    float x_val = get_value_as_float(x);
    float result_val = fmaxf(0.0f, x_val);
    
    Node *result = tensor_with_options(result_val, (TensorOptions){
        x->requires_grad, 0, result_dtype, result_device
    });
    
    if (result && result->requires_grad) {
        result->grad_fn = create_function(OP_RELU);
        if (result->grad_fn) {
            result->grad_fn->inputs = (Node **)cm_safe_malloc(sizeof(Node *), __FILE__, __LINE__);
            if (result->grad_fn->inputs) {
                result->grad_fn->inputs[0] = x;
                result->grad_fn->ninputs = 1;
            }
        }
    }
    
    return result;
}

Node *tanh_tensor(Node *x)
{
    if (!x) return NULL;
    
    // For now, assume x has the same dtype and device
    DType result_dtype = x->dtype;
    DeviceType result_device = x->device;
    
    float x_val = get_value_as_float(x);
    float result_val = tanhf(x_val);
    
    Node *result = tensor_with_options(result_val, (TensorOptions){
        x->requires_grad, 0, result_dtype, result_device
    });
    
    if (result && result->requires_grad) {
        result->grad_fn = create_function(OP_TANH);
        if (result->grad_fn) {
            result->grad_fn->inputs = (Node **)cm_safe_malloc(sizeof(Node *), __FILE__, __LINE__);
            if (result->grad_fn->inputs) {
                result->grad_fn->inputs[0] = x;
                result->grad_fn->ninputs = 1;
            }
        }
    }
    
    return result;
}

Node *sigmoid(Node *x)
{
    if (!x) return NULL;
    
    // For now, assume x has the same dtype and device
    DType result_dtype = x->dtype;
    DeviceType result_device = x->device;
    
    float x_val = get_value_as_float(x);
    float result_val = 1.0f / (1.0f + expf(-x_val));
    
    Node *result = tensor_with_options(result_val, (TensorOptions){
        x->requires_grad, 0, result_dtype, result_device
    });
    
    if (result && result->requires_grad) {
        result->grad_fn = create_function(OP_SIGMOID);
        if (result->grad_fn) {
            result->grad_fn->inputs = (Node **)cm_safe_malloc(sizeof(Node *), __FILE__, __LINE__);
            if (result->grad_fn->inputs) {
                result->grad_fn->inputs[0] = x;
                result->grad_fn->ninputs = 1;
            }
        }
    }
    
    return result;
}

Node *softmax(Node *x, int dim)
{
    if (!x) return NULL;
    
    // Simplified softmax for single element - ignore dim for now
    (void)dim; // Suppress unused parameter warning
    // For now, assume x has the same dtype and device
    DType result_dtype = x->dtype;
    DeviceType result_device = x->device;
    
    float x_val = get_value_as_float(x);
    float result_val = x_val > 0 ? 1.0f : 0.0f;
    
    Node *result = tensor_with_options(result_val, (TensorOptions){
        x->requires_grad, 0, result_dtype, result_device
    });
    
    if (result && result->requires_grad) {
        result->grad_fn = create_function(OP_SOFTMAX);
        if (result->grad_fn) {
            result->grad_fn->inputs = (Node **)cm_safe_malloc(sizeof(Node *), __FILE__, __LINE__);
            if (result->grad_fn->inputs) {
                result->grad_fn->inputs[0] = x;
                result->grad_fn->ninputs = 1;
            }
        }
    }
    
    return result;
}

Node *elu(Node *x, float alpha)
{
    if (!x) return NULL;
    
    // For now, assume x has the same dtype and device
    DType result_dtype = x->dtype;
    DeviceType result_device = x->device;
    
    float x_val = get_value_as_float(x);
    float result_val = x_val >= 0.0f ? x_val : alpha * (expf(x_val) - 1.0f);
    
    Node *result = tensor_with_options(result_val, (TensorOptions){
        x->requires_grad, 0, result_dtype, result_device
    });
    
    if (result && result->requires_grad) {
        result->grad_fn = create_function(OP_ELU);
        if (result->grad_fn) {
            result->grad_fn->inputs = (Node **)cm_safe_malloc(sizeof(Node *), __FILE__, __LINE__);
            if (result->grad_fn->inputs) {
                result->grad_fn->inputs[0] = x;
                result->grad_fn->ninputs = 1;
            }
        }
    }
    
    return result;
}

Node *gelu(Node *x) {
    // GELU function: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // Intermediate computations
    Node *three_node = tensor(3.0f, 0); // Create a Node for the scalar 3.0f
    Node *x_cubed = pow_tensor(x, three_node); 
    Node *term_in_pow = mul_scalar(x_cubed, 0.044715f);
    Node *sum_x_term = add(x, term_in_pow);

    // sqrt(2/pi) approx 0.7978845608
    Node *scaled_sum = mul_scalar(sum_x_term, 0.7978845608f);
    Node *tanh_val = tanh_tensor(scaled_sum);
    Node *one_plus_tanh = add_scalar(tanh_val, 1.0f);
    Node *product_x_one_plus_tanh = mul(x, one_plus_tanh);
    Node *result = mul_scalar(product_x_one_plus_tanh, 0.5f);

    if (x->requires_grad) {
        result->requires_grad = true;
        result->grad_fn = create_function(OP_GELU);
        if (result->grad_fn) {
            result->grad_fn->inputs = (Node **)cm_safe_malloc(sizeof(Node *), __FILE__, __LINE__);
            if (result->grad_fn->inputs) {
                result->grad_fn->inputs[0] = x;
                result->grad_fn->ninputs = 1;
            }
            result->grad_fn->backward = gelu_backward;
        }
    }

    // Temporarily removed cleanup of intermediate nodes to focus on segfault
    // if (x_cubed != result && x_cubed->grad_fn == NULL && !x_cubed->requires_grad) free_node_recursive(x_cubed); 
    // if (term_in_pow != result && term_in_pow->grad_fn == NULL && !term_in_pow->requires_grad) free_node_recursive(term_in_pow); 
    // if (sum_x_term != result && sum_x_term->grad_fn == NULL && !sum_x_term->requires_grad) free_node_recursive(sum_x_term); 
    // if (scaled_sum != result && scaled_sum->grad_fn == NULL && !scaled_sum->requires_grad) free_node_recursive(scaled_sum); 
    // if (one_plus_tanh != result && one_plus_tanh->grad_fn == NULL && !one_plus_tanh->requires_grad) free_node_recursive(one_plus_tanh); 
    // if (product_x_one_plus_tanh != result && product_x_one_plus_tanh->grad_fn == NULL && !product_x_one_plus_tanh->requires_grad) free_node_recursive(product_x_one_plus_tanh);
    // if (three_node) free_node_recursive(three_node); // Also cleanup the temporary node

    return result;
}

// Simple MSE loss implementation
Node *mse_loss(Node *input, Node *target, int reduction)
{
    if (!input || !target) return NULL;
    
    // For now, ignore reduction parameter
    (void)reduction; // Suppress unused parameter warning
    
    Node *diff = sub(input, target);
    Node *squared = mul(diff, diff);
    return squared;
}

void free_node(Node *node) {
    if (!node || node->is_freed) return;
    
    node->is_freed = 1;
    
    if (node->tensor) {
        if (node->tensor->storage) {
            node->tensor->storage->ref_count--;
            if (node->tensor->storage->ref_count <= 0) {
                cm_safe_free((void **)&node->tensor->storage->data);
                cm_safe_free((void **)&node->tensor->storage);
            }
        }
        cm_safe_free((void **)&node->tensor->sizes);
        cm_safe_free((void **)&node->tensor->strides);
        cm_safe_free((void **)&node->tensor);
    }
    
    if (node->grad_fn) {
        cm_safe_free((void **)&node->grad_fn->inputs);
        cm_safe_free((void **)&node->grad_fn->saved_tensors);
        cm_safe_free((void **)&node->grad_fn);
    }
    
    cm_safe_free((void **)&node->input_nodes);
    cm_safe_free((void **)&node);
}