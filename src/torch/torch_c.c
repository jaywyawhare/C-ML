/*
 * torch_c.c - PyTorch-like C API for C-ML (optimized hot paths)
 *
 * Optimizations (no inline assembly):
 *  - Cached default dtype/device avoids mutex on tensor creation
 *  - Precomputed TensorConfig embedded in TorchTensorOptions
 *  - Direct tensor_* / autograd calls (skip cml_* wrapper indirection)
 *  - Unified tensor creation helper
 *  - See torch_c_inline.h for header inlines
 */

#include "torch/torch_c.h"
#include "torch/torch_c_internal.h"
#include "torch/torch_eager.h"

#include "cml.h"
#include "tensor/tensor.h"
#include "tensor/realize.h"
#include "ops/uops.h"
#include "ops/ir/context.h"
#include "core/error_stack.h"
#include "backend/device.h"

#include <stdlib.h>
#include <stdatomic.h>

static _Atomic uint32_t g_torch_default_dtype_storage  = DTYPE_FLOAT32;
static _Atomic uint32_t g_torch_default_device_storage = DEVICE_CPU;

/**
 * Thread-safe read of the cached default dtype (avoids a mutex on tensor creation).
 * @return (DType)
 */
DType torch_default_dtype_cached(void) {
    return (DType)atomic_load_explicit(&g_torch_default_dtype_storage, memory_order_relaxed);
}

/**
 * Thread-safe read of the cached default device.
 * @return (DeviceType)
 */
DeviceType torch_default_device_cached(void) {
    return (DeviceType)atomic_load_explicit(&g_torch_default_device_storage, memory_order_relaxed);
}

/* ------------------------------------------------------------------ */
/* Lifecycle                                                           */
/* ------------------------------------------------------------------ */

/**
 * Initialize the runtime (torch.* entry point) and prime the cached dtype/device.
 * @return (int)
 */
int torch_init(void) {
    int rc = cml_init();
    if (rc == 0) {
        atomic_store_explicit(&g_torch_default_dtype_storage, (uint32_t)cml_get_default_dtype(),
                              memory_order_relaxed);
        atomic_store_explicit(&g_torch_default_device_storage, (uint32_t)cml_get_default_device(),
                              memory_order_relaxed);
    }
    return rc;
}

/**
 * Tear down the runtime; mirror of torch_init.
 * @return (int)
 */
int torch_cleanup(void) { return cml_cleanup(); }

/**
 * Report the C-ML library version (torch.__version__ analogue).
 * @param major (int*)
 * @param minor (int*)
 * @param patch (int*)
 * @param version_string (const char**)
 * @return (void)
 */
void torch_get_version(int* major, int* minor, int* patch, const char** version_string) {
    cml_get_version(major, minor, patch, version_string);
}

/* ------------------------------------------------------------------ */
/* TensorOptions                                                       */
/* ------------------------------------------------------------------ */

/**
 * Construct default TensorOptions from the cached dtype/device (torch.TensorOptions).
 * @return (TorchTensorOptions)
 */
TorchTensorOptions torch_options(void) {
    TorchTensorOptions opts = {0};
    opts.dtype              = torch_default_dtype_cached();
    opts.device             = torch_default_device_cached();
    opts.requires_grad      = false;
    opts.has_dtype          = false;
    opts.has_device         = false;
    torch_opts_sync_config(&opts);
    return opts;
}

/**
 * Builder: return a copy of opts with an explicit dtype (torch.dtype()).
 * @param opts (TorchTensorOptions)
 * @param dtype (DType)
 * @return (TorchTensorOptions)
 */
TorchTensorOptions torch_options_dtype(TorchTensorOptions opts, DType dtype) {
    opts.dtype     = dtype;
    opts.has_dtype = true;
    torch_opts_sync_config(&opts);
    return opts;
}

/**
 * Builder: return a copy of opts with an explicit device (torch.device()).
 * @param opts (TorchTensorOptions)
 * @param device (DeviceType)
 * @return (TorchTensorOptions)
 */
TorchTensorOptions torch_options_device(TorchTensorOptions opts, DeviceType device) {
    opts.device     = device;
    opts.has_device = true;
    torch_opts_sync_config(&opts);
    return opts;
}

/**
 * Builder: return a copy of opts with requires_grad set.
 * @param opts (TorchTensorOptions)
 * @param requires_grad (bool)
 * @return (TorchTensorOptions)
 */
TorchTensorOptions torch_options_requires_grad(TorchTensorOptions opts, bool requires_grad) {
    opts.requires_grad = requires_grad;
    return opts;
}

/**
 * Extract the precomputed internal TensorConfig from TensorOptions for core calls.
 * @param opts (const TorchTensorOptions*)
 * @return (TensorConfig)
 */
TensorConfig torch_options_to_config(const TorchTensorOptions* opts) {
    if (!opts)
        return torch_config_default();
    return opts->config;
}

/* ------------------------------------------------------------------ */
/* Device & dtype                                                      */
/* ------------------------------------------------------------------ */

/**
 * torch.cuda.is_available(): true if a CUDA device is present.
 * @return (bool)
 */
bool torch_cuda_is_available(void) { return device_cuda_available(); }

/**
 * torch.cuda.device_count(): number of available CUDA devices.
 * @return (int)
 */
int torch_cuda_device_count(void) { return device_cuda_get_count(); }

/**
 * torch.get_default_device(): the current default device.
 * @return (DeviceType)
 */
DeviceType torch_get_default_device(void) { return torch_default_device_cached(); }

/**
 * torch.set_default_device(): set the default device and refresh the cache.
 * @param device (DeviceType)
 * @return (void)
 */
void torch_set_default_device(DeviceType device) {
    atomic_store_explicit(&g_torch_default_device_storage, (uint32_t)device, memory_order_relaxed);
    cml_set_default_device(device);
}

/**
 * torch.get_default_dtype(): the current default dtype.
 * @return (DType)
 */
DType torch_get_default_dtype(void) { return torch_default_dtype_cached(); }

/**
 * torch.set_default_dtype(): set the default dtype and refresh the cache.
 * @param dtype (DType)
 * @return (void)
 */
void torch_set_default_dtype(DType dtype) {
    atomic_store_explicit(&g_torch_default_dtype_storage, (uint32_t)dtype, memory_order_relaxed);
    cml_set_default_dtype(dtype);
}

/**
 * torch.manual_seed(): seed the global RNG.
 * @param seed (uint64_t)
 * @return (void)
 */
void torch_manual_seed(uint64_t seed) { cml_manual_seed(seed); }

/* ------------------------------------------------------------------ */
/* Tensor lifecycle & accessors                                        */
/* ------------------------------------------------------------------ */

/**
 * Increment a tensor's reference count (Tensor.retain).
 * @param t (Tensor*)
 * @return (void)
 */
void torch_tensor_retain(Tensor* t) { torch_tensor_retain_fast(t); }

/**
 * Release a reference to a tensor (del / Tensor.free).
 * @param t (Tensor*)
 * @return (void)
 */
void torch_tensor_free(Tensor* t) { tensor_free(t); }

/**
 * Current reference count of a tensor (0 if NULL).
 * @param t (const Tensor*)
 * @return (int)
 */
int torch_tensor_ref_count(const Tensor* t) {
    if (!t)
        return 0;
    return t->ref_count;
}

/**
 * Tensor.dim(): number of dimensions.
 * @param t (const Tensor*)
 * @return (int)
 */
int torch_tensor_ndim(const Tensor* t) { return torch_tensor_ndim_fast(t); }

/**
 * Tensor.numel(): total element count.
 * @param t (const Tensor*)
 * @return (size_t)
 */
size_t torch_tensor_numel(const Tensor* t) { return torch_tensor_numel_fast(t); }

/**
 * Tensor.dtype.
 * @param t (const Tensor*)
 * @return (DType)
 */
DType torch_tensor_dtype(const Tensor* t) { return torch_tensor_dtype_fast(t); }

/**
 * Tensor.device.
 * @param t (const Tensor*)
 * @return (DeviceType)
 */
DeviceType torch_tensor_device(const Tensor* t) { return torch_tensor_device_fast(t); }

/**
 * Tensor.is_contiguous().
 * @param t (const Tensor*)
 * @return (bool)
 */
bool torch_tensor_is_contiguous(const Tensor* t) { return torch_tensor_is_contiguous_fast(t); }

/**
 * Tensor.requires_grad.
 * @param t (const Tensor*)
 * @return (bool)
 */
bool torch_tensor_requires_grad(const Tensor* t) { return torch_tensor_requires_grad_fast(t); }

/**
 * Tensor.requires_grad_(): set whether autograd tracks this tensor.
 * @param t (Tensor*)
 * @param requires_grad (bool)
 * @return (void)
 */
void torch_tensor_set_requires_grad(Tensor* t, bool requires_grad) {
    if (t)
        tensor_set_requires_grad(t, requires_grad);
}

/**
 * Tensor.size(): pointer to the shape array (valid until the tensor is freed).
 * @param t (const Tensor*)
 * @return (const int*)
 */
const int* torch_tensor_sizes(const Tensor* t) { return torch_tensor_sizes_fast(t); }

/**
 * Tensor.data_ptr(): realize any lazy IR and return the raw buffer; NULL on error.
 * @param t (Tensor*)
 * @return (void*)
 */
void* torch_tensor_data_ptr(Tensor* t) {
    if (!t) {
        error_stack_push(CM_INVALID_ARGUMENT, "torch_tensor_data_ptr: null tensor", __FILE__,
                         __LINE__, __func__);
        return NULL;
    }
    if (t->ir_node && tensor_realize(t) != 0) {
        error_stack_push(CM_OPERATION_FAILED, "torch_tensor_data_ptr: realize failed", __FILE__,
                         __LINE__, __func__);
        return NULL;
    }
    void* p = tensor_data_ptr(t);
    if (!p) {
        error_stack_push(CM_OPERATION_FAILED, "torch_tensor_data_ptr: data unavailable", __FILE__,
                         __LINE__, __func__);
    }
    return p;
}

/**
 * Typed data_ptr for float32 tensors; NULL if the dtype is not float32.
 * @param t (Tensor*)
 * @return (float*)
 */
float* torch_tensor_data_ptr_f32(Tensor* t) {
    if (!t || t->dtype != DTYPE_FLOAT32) {
        error_stack_push(CM_INVALID_ARGUMENT, "torch_tensor_data_ptr_f32: expected float32 tensor",
                         __FILE__, __LINE__, __func__);
        return NULL;
    }
    return (float*)torch_tensor_data_ptr(t);
}

/**
 * True when the tensor is a materialized leaf with no pending lazy IR node.
 * @param t (const Tensor*)
 * @return (bool)
 */
bool torch_tensor_is_materialized(const Tensor* t) {
    return t != NULL && t->ir_node == NULL && t->is_executed;
}

/**
 * True when the tensor still references a lazy IR graph node.
 * @param t (const Tensor*)
 * @return (bool)
 */
bool torch_tensor_has_lazy_ir(const Tensor* t) { return t != NULL && t->ir_node != NULL; }

/**
 * Tensor.item() for a scalar float32 tensor; 0.0f and an error on misuse.
 * @param t (Tensor*)
 * @return (float)
 */
float torch_tensor_item_float(Tensor* t) {
    if (!t || t->numel != 1 || t->dtype != DTYPE_FLOAT32) {
        error_stack_push(CM_INVALID_ARGUMENT, "torch_tensor_item_float: requires scalar float32",
                         __FILE__, __LINE__, __func__);
        return 0.0f;
    }
    return tensor_get_float(t, 0);
}

/**
 * Write the single element of a scalar float32 tensor (Tensor.fill_ for numel==1).
 * @param t (Tensor*)
 * @param value (float)
 * @return (void)
 */
void torch_tensor_set_item_float(Tensor* t, float value) {
    if (!t || t->numel != 1 || t->dtype != DTYPE_FLOAT32) {
        error_stack_push(CM_INVALID_ARGUMENT,
                         "torch_tensor_set_item_float: requires scalar float32", __FILE__, __LINE__,
                         __func__);
        return;
    }
    tensor_set_float(t, 0, value);
}

/**
 * torch.empty(): uninitialized tensor of the given shape.
 * @param shape (int*)
 * @param ndim (int)
 * @param opts (const TorchTensorOptions*)
 * @return (Tensor*)
 */
Tensor* torch_empty(int* shape, int ndim, const TorchTensorOptions* opts) {
    return torch_create_tensor(tensor_empty, shape, ndim, opts);
}

/**
 * torch.zeros(): zero-filled tensor of the given shape.
 * @param shape (int*)
 * @param ndim (int)
 * @param opts (const TorchTensorOptions*)
 * @return (Tensor*)
 */
Tensor* torch_zeros(int* shape, int ndim, const TorchTensorOptions* opts) {
    return torch_create_tensor(tensor_zeros, shape, ndim, opts);
}

/**
 * torch.ones(): one-filled tensor of the given shape.
 * @param shape (int*)
 * @param ndim (int)
 * @param opts (const TorchTensorOptions*)
 * @return (Tensor*)
 */
Tensor* torch_ones(int* shape, int ndim, const TorchTensorOptions* opts) {
    return torch_create_tensor(tensor_ones, shape, ndim, opts);
}

/**
 * torch.full(): tensor of the given shape filled with `value`.
 * @param shape (int*)
 * @param ndim (int)
 * @param opts (const TorchTensorOptions*)
 * @param value (float)
 * @return (Tensor*)
 */
Tensor* torch_full(int* shape, int ndim, const TorchTensorOptions* opts, float value) {
    TensorConfig scratch;
    const TensorConfig* cfg = torch_resolve_config(opts, &scratch);
    Tensor* t               = tensor_full(shape, ndim, cfg, value);
    if (t && opts && opts->requires_grad)
        t->requires_grad = true;
    return t;
}

/**
 * torch.rand(): tensor sampled from the uniform [0, 1) distribution.
 * @param shape (int*)
 * @param ndim (int)
 * @param opts (const TorchTensorOptions*)
 * @return (Tensor*)
 */
Tensor* torch_rand(int* shape, int ndim, const TorchTensorOptions* opts) {
    return torch_create_tensor(tensor_rand, shape, ndim, opts);
}

/**
 * torch.randn(): tensor sampled from the standard normal distribution.
 * @param shape (int*)
 * @param ndim (int)
 * @param opts (const TorchTensorOptions*)
 * @return (Tensor*)
 */
Tensor* torch_randn(int* shape, int ndim, const TorchTensorOptions* opts) {
    return torch_create_tensor(tensor_randn, shape, ndim, opts);
}

/**
 * torch.eye(): n-by-n identity matrix.
 * @param n (int)
 * @param opts (const TorchTensorOptions*)
 * @return (Tensor*)
 */
Tensor* torch_eye(int n, const TorchTensorOptions* opts) {
    TensorConfig scratch;
    const TensorConfig* cfg = torch_resolve_config(opts, &scratch);
    Tensor* t               = tensor_eye(n, cfg);
    if (t && opts && opts->requires_grad)
        t->requires_grad = true;
    return t;
}

/**
 * torch.arange(): 1-D tensor of values from start to end stepping by step.
 * @param start (float)
 * @param end (float)
 * @param step (float)
 * @param opts (const TorchTensorOptions*)
 * @return (Tensor*)
 */
Tensor* torch_arange(float start, float end, float step, const TorchTensorOptions* opts) {
    TensorConfig scratch;
    const TensorConfig* cfg = torch_resolve_config(opts, &scratch);
    Tensor* t               = tensor_arange(start, end, step, cfg);
    if (t && opts && opts->requires_grad)
        t->requires_grad = true;
    return t;
}

/**
 * torch.linspace(): `steps` evenly spaced values from start to end inclusive.
 * @param start (float)
 * @param end (float)
 * @param steps (int)
 * @param opts (const TorchTensorOptions*)
 * @return (Tensor*)
 */
Tensor* torch_linspace(float start, float end, int steps, const TorchTensorOptions* opts) {
    TensorConfig scratch;
    const TensorConfig* cfg = torch_resolve_config(opts, &scratch);
    Tensor* t               = tensor_linspace(start, end, steps, cfg);
    if (t && opts && opts->requires_grad)
        t->requires_grad = true;
    return t;
}

/**
 * torch.from_blob(): wrap caller-owned storage as a tensor without copying.
 * @param data (void*)
 * @param shape (int*)
 * @param ndim (int)
 * @param opts (const TorchTensorOptions*)
 * @return (Tensor*)
 */
Tensor* torch_from_blob(void* data, int* shape, int ndim, const TorchTensorOptions* opts) {
    if (!data || !shape || ndim <= 0) {
        error_stack_push(CM_INVALID_ARGUMENT, "torch_from_blob: data/shape required", __FILE__,
                         __LINE__, __func__);
        return NULL;
    }
    TensorConfig scratch;
    const TensorConfig* cfg = torch_resolve_config(opts, &scratch);
    Tensor* t               = tensor_from_data(data, shape, ndim, cfg);
    if (t && opts && opts->requires_grad)
        t->requires_grad = true;
    return t;
}

/**
 * torch.zeros_like(): zeros with the same shape/options as t.
 * @param t (Tensor*)
 * @return (Tensor*)
 */
Tensor* torch_zeros_like(Tensor* t) { return t ? tensor_zeros_like(t) : NULL; }

/**
 * torch.ones_like(): ones with the same shape/options as t.
 * @param t (Tensor*)
 * @return (Tensor*)
 */
Tensor* torch_ones_like(Tensor* t) { return t ? tensor_ones_like(t) : NULL; }

/**
 * torch.randn_like(): standard-normal samples with the same shape/options as t.
 * @param t (Tensor*)
 * @return (Tensor*)
 */
Tensor* torch_randn_like(Tensor* t) { return t ? tensor_randn_like(t) : NULL; }

/* ------------------------------------------------------------------ */
/* Tensor operations (direct autograd forward ops)                     */
/* ------------------------------------------------------------------ */

/**
 * torch.add(): elementwise sum, trying the eager fast path before the lazy IR op.
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* torch_add(Tensor* a, Tensor* b) {
    Tensor* e = torch_eager_binary(UOP_ADD, a, b);
    return e ? e : tensor_add(a, b);
}
/**
 * torch.sub(): elementwise difference, eager fast path with lazy fallback.
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* torch_sub(Tensor* a, Tensor* b) {
    Tensor* e = torch_eager_binary(UOP_SUB, a, b);
    return e ? e : tensor_sub(a, b);
}
/**
 * torch.mul(): elementwise product, eager fast path with lazy fallback.
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* torch_mul(Tensor* a, Tensor* b) {
    Tensor* e = torch_eager_binary(UOP_MUL, a, b);
    return e ? e : tensor_mul(a, b);
}
/**
 * torch.div(): elementwise quotient, eager fast path with lazy fallback.
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* torch_div(Tensor* a, Tensor* b) {
    Tensor* e = torch_eager_binary(UOP_DIV, a, b);
    return e ? e : tensor_div(a, b);
}
/**
 * torch.matmul(): matrix product, eager BLAS fast path with lazy fallback.
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* torch_matmul(Tensor* a, Tensor* b) {
    Tensor* e = torch_eager_binary(UOP_MATMUL, a, b);
    return e ? e : tensor_matmul(a, b);
}
/**
 * torch.pow(): elementwise exponentiation.
 * @param a (Tensor*)
 * @param b (Tensor*)
 * @return (Tensor*)
 */
Tensor* torch_pow(Tensor* a, Tensor* b) { return tensor_pow(a, b); }

/**
 * torch.sum(): reduce-sum over a dim (or all dims if dim < 0).
 * @param a (Tensor*)
 * @param dim (int)
 * @param keepdim (bool)
 * @return (Tensor*)
 */
Tensor* torch_sum(Tensor* a, int dim, bool keepdim) { return tensor_sum(a, dim, keepdim); }
/**
 * torch.mean(): reduce-mean over a dim.
 * @param a (Tensor*)
 * @param dim (int)
 * @param keepdim (bool)
 * @return (Tensor*)
 */
Tensor* torch_mean(Tensor* a, int dim, bool keepdim) { return tensor_mean(a, dim, keepdim); }
/**
 * torch.max(): reduce-max over a dim.
 * @param a (Tensor*)
 * @param dim (int)
 * @param keepdim (bool)
 * @return (Tensor*)
 */
Tensor* torch_max(Tensor* a, int dim, bool keepdim) { return tensor_max(a, dim, keepdim); }
/**
 * torch.min(): reduce-min over a dim.
 * @param a (Tensor*)
 * @param dim (int)
 * @param keepdim (bool)
 * @return (Tensor*)
 */
Tensor* torch_min(Tensor* a, int dim, bool keepdim) { return tensor_min(a, dim, keepdim); }

/**
 * torch.relu(): rectified linear unit, eager fast path with lazy fallback.
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* torch_relu(Tensor* a) {
    Tensor* e = torch_eager_unary(UOP_RELU, a);
    return e ? e : tensor_relu(a);
}
/**
 * torch.sigmoid(): logistic activation, eager fast path with lazy fallback.
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* torch_sigmoid(Tensor* a) {
    Tensor* e = torch_eager_unary(UOP_SIGMOID, a);
    return e ? e : tensor_sigmoid(a);
}
/**
 * torch.tanh(): hyperbolic tangent, eager fast path with lazy fallback.
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* torch_tanh(Tensor* a) {
    Tensor* e = torch_eager_unary(UOP_TANH, a);
    return e ? e : tensor_tanh(a);
}
/**
 * torch.softmax(): softmax along a dim.
 * @param a (Tensor*)
 * @param dim (int)
 * @return (Tensor*)
 */
Tensor* torch_softmax(Tensor* a, int dim) { return tensor_softmax(a, dim); }
/**
 * torch.gelu(): Gaussian error linear unit activation.
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* torch_gelu(Tensor* a) { return uop_gelu(a); }

/**
 * Tensor.reshape(): view with a new shape (same element count).
 * @param a (Tensor*)
 * @param new_shape (int*)
 * @param new_ndim (int)
 * @return (Tensor*)
 */
Tensor* torch_reshape(Tensor* a, int* new_shape, int new_ndim) {
    return tensor_reshape(a, new_shape, new_ndim);
}
/**
 * Tensor.transpose(): swap two dimensions.
 * @param a (Tensor*)
 * @param dim0 (int)
 * @param dim1 (int)
 * @return (Tensor*)
 */
Tensor* torch_transpose(Tensor* a, int dim0, int dim1) { return tensor_transpose(a, dim0, dim1); }
/**
 * Tensor.squeeze(): drop a size-1 dimension.
 * @param a (Tensor*)
 * @param dim (int)
 * @return (Tensor*)
 */
Tensor* torch_squeeze(Tensor* a, int dim) { return tensor_squeeze(a, dim); }
/**
 * Tensor.unsqueeze(): insert a size-1 dimension.
 * @param a (Tensor*)
 * @param dim (int)
 * @return (Tensor*)
 */
Tensor* torch_unsqueeze(Tensor* a, int dim) { return tensor_unsqueeze(a, dim); }
/**
 * torch.cat(): concatenate tensors along an existing dimension.
 * @param tensors (Tensor**)
 * @param num_tensors (int)
 * @param dim (int)
 * @return (Tensor*)
 */
Tensor* torch_cat(Tensor** tensors, int num_tensors, int dim) {
    return tensor_concat(tensors, num_tensors, dim);
}
/**
 * torch.stack(): stack tensors along a new dimension.
 * @param tensors (Tensor**)
 * @param num_tensors (int)
 * @param dim (int)
 * @return (Tensor*)
 */
Tensor* torch_stack(Tensor** tensors, int num_tensors, int dim) {
    return tensor_stack(tensors, num_tensors, dim);
}
/**
 * Tensor.clone(): deep copy of a tensor.
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* torch_clone(Tensor* a) { return tensor_clone(a); }
/**
 * Tensor.detach(): view sharing storage but excluded from autograd.
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* torch_detach(Tensor* a) { return tensor_detach(a); }
/**
 * Tensor.contiguous(): a contiguous-memory copy (or the input if already contiguous).
 * @param a (Tensor*)
 * @return (Tensor*)
 */
Tensor* torch_contiguous(Tensor* a) { return tensor_contiguous(a); }

/* ------------------------------------------------------------------ */
/* Autograd (direct engine calls)                                      */
/* ------------------------------------------------------------------ */

/**
 * Tensor.backward(): run the backward pass, forwarded to the autograd engine.
 * @param tensor (Tensor*)
 * @param gradient (Tensor*)
 * @param retain_graph (bool)
 * @param create_graph (bool)
 * @return (void)
 */
void torch_backward(Tensor* tensor, Tensor* gradient, bool retain_graph, bool create_graph) {
    tensor_backward(tensor, gradient, retain_graph, create_graph);
}

/**
 * Tensor.grad.zero_(): clear a tensor's accumulated gradient.
 * @param tensor (Tensor*)
 * @return (void)
 */
void torch_zero_grad(Tensor* tensor) { tensor_zero_grad(tensor); }

/**
 * torch.no_grad(): disable autograd tracking for subsequent ops.
 * @return (void)
 */
void torch_no_grad(void) { autograd_no_grad_enter(); }

/**
 * torch.enable_grad(): re-enable autograd tracking.
 * @return (void)
 */
void torch_enable_grad(void) { autograd_set_grad_mode(true); }

/**
 * torch.is_grad_enabled(): whether autograd is currently tracking.
 * @return (bool)
 */
bool torch_is_grad_enabled(void) { return autograd_is_grad_enabled(); }

/**
 * Tensor.grad: the accumulated gradient of a tensor.
 * @param t (Tensor*)
 * @return (Tensor*)
 */
Tensor* torch_get_grad(Tensor* t) { return tensor_get_grad(t); }

/* ------------------------------------------------------------------ */
/* nn.Module API                                                       */
/* ------------------------------------------------------------------ */

/**
 * Module.forward()/__call__(): run a module's forward pass.
 * @param module (Module*)
 * @param input (Tensor*)
 * @return (Tensor*)
 */
Tensor* torch_module_forward(Module* module, Tensor* input) {
    return module_forward(module, input);
}

/**
 * Module.train(): put a module in training mode.
 * @param module (Module*)
 * @return (void)
 */
void torch_module_train(Module* module) { module_set_training(module, true); }

/**
 * Module.eval(): put a module in evaluation mode.
 * @param module (Module*)
 * @return (void)
 */
void torch_module_eval(Module* module) { module_set_training(module, false); }

/**
 * Module.training: whether the module is in training mode.
 * @param module (Module*)
 * @return (bool)
 */
bool torch_module_is_training(Module* module) { return module_is_training(module); }

/**
 * Module.zero_grad(): clear gradients on all of a module's parameters.
 * @param module (Module*)
 * @return (void)
 */
void torch_module_zero_grad(Module* module) { module_zero_grad(module); }

/**
 * nn.Linear(): construct a linear layer with the cached default dtype/device.
 * @param in_features (int)
 * @param out_features (int)
 * @param bias (bool)
 * @return (Linear*)
 */
Linear* torch_nn_linear(int in_features, int out_features, bool bias) {
    return cml_nn_linear(in_features, out_features, torch_default_dtype_cached(),
                         torch_default_device_cached(), bias);
}

/**
 * nn.ReLU(): construct a ReLU activation module.
 * @return (ReLU*)
 */
ReLU* torch_nn_relu(void) { return cml_nn_relu(false); }

/**
 * nn.Sequential(): construct an empty sequential container.
 * @return (Sequential*)
 */
Sequential* torch_nn_sequential(void) { return cml_nn_sequential(); }

/**
 * Sequential.append(): add a layer to a sequential container.
 * @param seq (Sequential*)
 * @param layer (Module*)
 * @return (void)
 */
void torch_nn_sequential_add(Sequential* seq, Module* layer) { cml_nn_sequential_add(seq, layer); }

/**
 * Run a sequential container's forward pass over the input.
 * @param seq (Sequential*)
 * @param input (Tensor*)
 * @return (Tensor*)
 */
Tensor* torch_nn_sequential_forward(Sequential* seq, Tensor* input) {
    return cml_nn_sequential_forward(seq, input);
}

/* ------------------------------------------------------------------ */
/* State dict                                                          */
/* ------------------------------------------------------------------ */

/**
 * Module.state_dict(): collect a module's named parameters/buffers under a prefix.
 * @param module (Module*)
 * @param prefix (const char*)
 * @return (StateDict*)
 */
StateDict* torch_module_state_dict(Module* module, const char* prefix) {
    return nn_get_state_dict(module, prefix);
}

/**
 * Module.load_state_dict(): copy parameters from a state dict into a module.
 * @param module (Module*)
 * @param sd (const StateDict*)
 * @param strict (bool)
 * @return (int)
 */
int torch_module_load_state_dict(Module* module, const StateDict* sd, bool strict) {
    return nn_load_state_dict(module, sd, strict);
}

/**
 * Look up a tensor by key in a state dict.
 * @param sd (const StateDict*)
 * @param key (const char*)
 * @return (Tensor*)
 */
Tensor* torch_state_dict_get(const StateDict* sd, const char* key) {
    return nn_state_dict_get(sd, key);
}

/**
 * Free a state dict produced by torch_module_state_dict.
 * @param sd (StateDict*)
 * @return (void)
 */
void torch_state_dict_free(StateDict* sd) { nn_state_dict_free(sd); }

/* ------------------------------------------------------------------ */
/* Optimizers                                                          */
/* ------------------------------------------------------------------ */

/**
 * torch.optim.Adam(): build an Adam optimizer over a model's parameters.
 * @param model (Module*)
 * @param lr (float)
 * @param weight_decay (float)
 * @param beta1 (float)
 * @param beta2 (float)
 * @param eps (float)
 * @return (Optimizer*)
 */
Optimizer* torch_optim_adam(Module* model, float lr, float weight_decay, float beta1, float beta2,
                            float eps) {
    return cml_optim_adam_for_model(model, lr, weight_decay, beta1, beta2, eps);
}

/**
 * torch.optim.SGD(): build an SGD optimizer over a model's parameters.
 * @param model (Module*)
 * @param lr (float)
 * @param momentum (float)
 * @param weight_decay (float)
 * @return (Optimizer*)
 */
Optimizer* torch_optim_sgd(Module* model, float lr, float momentum, float weight_decay) {
    return cml_optim_sgd_for_model(model, lr, momentum, weight_decay);
}

/**
 * Optimizer.zero_grad(): clear gradients tracked by the optimizer.
 * @param optimizer (Optimizer*)
 * @return (void)
 */
void torch_optim_zero_grad(Optimizer* optimizer) { cml_optim_zero_grad(optimizer); }

/**
 * Optimizer.step(): apply one parameter update.
 * @param optimizer (Optimizer*)
 * @return (void)
 */
void torch_optim_step(Optimizer* optimizer) { cml_optim_step(optimizer); }

/**
 * Free an optimizer.
 * @param optimizer (Optimizer*)
 * @return (void)
 */
void torch_optim_free(Optimizer* optimizer) { optimizer_free(optimizer); }

/* ------------------------------------------------------------------ */
/* Loss functions                                                      */
/* ------------------------------------------------------------------ */

/**
 * nn.functional.mse_loss(): mean squared error between input and target.
 * @param input (Tensor*)
 * @param target (Tensor*)
 * @return (Tensor*)
 */
Tensor* torch_nn_mse_loss(Tensor* input, Tensor* target) { return cml_nn_mse_loss(input, target); }

/**
 * nn.functional.cross_entropy(): cross-entropy loss between logits and target.
 * @param input (Tensor*)
 * @param target (Tensor*)
 * @return (Tensor*)
 */
Tensor* torch_nn_cross_entropy_loss(Tensor* input, Tensor* target) {
    return cml_nn_cross_entropy_loss(input, target);
}

/* ------------------------------------------------------------------ */
/* Runtime module                                                      */
/* ------------------------------------------------------------------ */

/**
 * Allocate a zeroed runtime module wrapper tagged with the given kind.
 * @param kind (TorchRuntimeKind)
 * @return (TorchRuntimeModule*)
 */
static TorchRuntimeModule* torch_runtime_alloc(TorchRuntimeKind kind) {
    TorchRuntimeModule* rt = (TorchRuntimeModule*)calloc(1, sizeof(TorchRuntimeModule));
    if (rt)
        rt->kind = kind;
    return rt;
}

/**
 * Wrap an existing eager Module as a runtime module (does not take ownership).
 * @param module (Module*)
 * @return (TorchRuntimeModule*)
 */
TorchRuntimeModule* torch_runtime_from_module(Module* module) {
    if (!module)
        return NULL;
    TorchRuntimeModule* rt = torch_runtime_alloc(TORCH_RUNTIME_EAGER);
    if (!rt)
        return NULL;
    rt->eager_module = module;
    return rt;
}

/**
 * Load an AOT-compiled shared library into a runtime module.
 * @param path (const char*)
 * @return (TorchRuntimeModule*)
 */
TorchRuntimeModule* torch_runtime_load_aot(const char* path) {
    if (!path)
        return NULL;
    CMLAOTModel* aot = cml_aot_load(path);
    if (!aot)
        return NULL;
    TorchRuntimeModule* rt = torch_runtime_alloc(TORCH_RUNTIME_AOT);
    if (!rt) {
        cml_aot_free(aot);
        return NULL;
    }
    rt->aot_model = aot;
    return rt;
}

/**
 * Load a .cpte portable program and provision its arena into a runtime module.
 * @param path (const char*)
 * @return (TorchRuntimeModule*)
 */
TorchRuntimeModule* torch_runtime_load_pte(const char* path) {
    if (!path)
        return NULL;
    CMLPTEModel* pte = torch_pte_load(path);
    if (!pte)
        return NULL;

    TorchRuntimeModule* rt = torch_runtime_alloc(TORCH_RUNTIME_PTE);
    if (!rt) {
        torch_pte_free(pte);
        return NULL;
    }
    rt->pte_model = pte;

    size_t arena = torch_pte_get_required_arena_size(pte);
    if (arena > 0) {
        rt->memory      = torch_memory_create(arena);
        rt->owns_memory = rt->memory != NULL;
        pte->memory     = rt->memory;
    }
    return rt;
}

/**
 * Export a module to a .cpte program (ahead-of-time preparation).
 * @param module (Module*)
 * @param sample_input (Tensor*)
 * @param path (const char*)
 * @param opts (const TorchPTEExportOptions*)
 * @return (int)
 */
int torch_runtime_export_pte(Module* module, Tensor* sample_input, const char* path,
                             const TorchPTEExportOptions* opts) {
    return torch_pte_export_module(module, sample_input, path, opts);
}

/**
 * Attach a caller-owned memory arena to a runtime, releasing any owned one.
 * @param runtime (TorchRuntimeModule*)
 * @param memory (TorchMemoryManager*)
 * @return (void)
 */
void torch_runtime_set_memory(TorchRuntimeModule* runtime, TorchMemoryManager* memory) {
    if (!runtime)
        return;
    if (runtime->memory == memory)
        return;
    if (runtime->owns_memory && runtime->memory)
        torch_memory_free(runtime->memory);
    runtime->memory      = memory;
    runtime->owns_memory = false;
    if (runtime->pte_model)
        runtime->pte_model->memory = memory;
}

/** Run a runtime module's forward pass, dispatching by kind (eager/AOT/PTE). */
__attribute__((hot)) Tensor* torch_runtime_forward(TorchRuntimeModule* runtime, Tensor* input) {
    if (!runtime || !input)
        return NULL;

    switch (runtime->kind) {
    case TORCH_RUNTIME_EAGER:
        return module_forward(runtime->eager_module, input);

    case TORCH_RUNTIME_AOT: {
        if (!runtime->aot_model)
            return NULL;
        Tensor* inputs[] = {input};
        Tensor* outputs[1];
        if (cml_aot_execute(runtime->aot_model, inputs, 1, outputs, 1) != 0)
            return NULL;
        return outputs[0];
    }

    case TORCH_RUNTIME_PTE: {
        if (!runtime->pte_model)
            return NULL;
        Tensor* inputs[]   = {input};
        Tensor* outputs[1] = {NULL};
        if (torch_pte_execute(runtime->pte_model, inputs, 1, outputs, 1) != 0)
            return NULL;
        return outputs[0];
    }

    default:
        return NULL;
    }
}

/**
 * Free a runtime module and any resources (AOT/PTE model, arena, owned module) it owns.
 * @param runtime (TorchRuntimeModule*)
 * @return (void)
 */
void torch_runtime_free(TorchRuntimeModule* runtime) {
    if (!runtime)
        return;
    if (runtime->kind == TORCH_RUNTIME_AOT && runtime->aot_model)
        cml_aot_free(runtime->aot_model);
    if (runtime->kind == TORCH_RUNTIME_PTE && runtime->pte_model)
        torch_pte_free(runtime->pte_model);
    if (runtime->owns_memory && runtime->memory)
        torch_memory_free(runtime->memory);
    if (runtime->owns_eager && runtime->eager_module)
        module_free(runtime->eager_module);
    free(runtime);
}

/**
 * The backing kind (eager/AOT/PTE) of a runtime module.
 * @param runtime (const TorchRuntimeModule*)
 * @return (TorchRuntimeKind)
 */
TorchRuntimeKind torch_runtime_get_kind(const TorchRuntimeModule* runtime) {
    return runtime ? runtime->kind : TORCH_RUNTIME_EAGER;
}

/**
 * True if a runtime module has an attached memory arena.
 * @param runtime (const TorchRuntimeModule*)
 * @return (bool)
 */
bool torch_runtime_has_memory(const TorchRuntimeModule* runtime) {
    return runtime != NULL && runtime->memory != NULL;
}

/**
 * Arena size required by a PTE runtime module (0 for other kinds).
 * @param runtime (const TorchRuntimeModule*)
 * @return (size_t)
 */
size_t torch_runtime_pte_arena_size(const TorchRuntimeModule* runtime) {
    if (!runtime || runtime->kind != TORCH_RUNTIME_PTE || !runtime->pte_model)
        return 0;
    return torch_pte_get_required_arena_size(runtime->pte_model);
}

/* ------------------------------------------------------------------ */
/* IR helpers                                                          */
/* ------------------------------------------------------------------ */

/**
 * Reset the global IR context, freeing accumulated lazy graph nodes.
 * @return (void)
 */
void torch_reset_ir(void) { cml_ir_reset_global_context(); }

/**
 * Soft IR reset: clear the graph only, keeping the surrounding context.
 * @return (void)
 */
void torch_reset_ir_soft(void) { cml_ir_reset_graph_only(); }

/* ------------------------------------------------------------------ */
/* Error handling                                                      */
/* ------------------------------------------------------------------ */

/**
 * Message of the most recent error on the error stack.
 * @return (const char*)
 */
const char* torch_get_last_error(void) { return error_stack_get_last_message(); }

/**
 * Code of the most recent error on the error stack.
 * @return (int)
 */
int torch_get_last_error_code(void) { return error_stack_get_last_code(); }

/**
 * Whether any error is pending on the error stack.
 * @return (bool)
 */
bool torch_has_error(void) { return error_stack_has_errors(); }

/**
 * Clear all pending errors from the error stack.
 * @return (void)
 */
void torch_clear_error(void) { error_stack_clear(); }
