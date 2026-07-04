/*
 * torch_c.c — PyTorch-like C API for C-ML (optimized hot paths)
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

DType torch_default_dtype_cached(void) {
    return (DType)atomic_load_explicit(&g_torch_default_dtype_storage, memory_order_relaxed);
}

DeviceType torch_default_device_cached(void) {
    return (DeviceType)atomic_load_explicit(&g_torch_default_device_storage, memory_order_relaxed);
}

/* ------------------------------------------------------------------ */
/* Lifecycle                                                           */
/* ------------------------------------------------------------------ */

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

int torch_cleanup(void) { return cml_cleanup(); }

void torch_get_version(int* major, int* minor, int* patch, const char** version_string) {
    cml_get_version(major, minor, patch, version_string);
}

/* ------------------------------------------------------------------ */
/* TensorOptions                                                       */
/* ------------------------------------------------------------------ */

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

TorchTensorOptions torch_options_dtype(TorchTensorOptions opts, DType dtype) {
    opts.dtype     = dtype;
    opts.has_dtype = true;
    torch_opts_sync_config(&opts);
    return opts;
}

TorchTensorOptions torch_options_device(TorchTensorOptions opts, DeviceType device) {
    opts.device     = device;
    opts.has_device = true;
    torch_opts_sync_config(&opts);
    return opts;
}

TorchTensorOptions torch_options_requires_grad(TorchTensorOptions opts, bool requires_grad) {
    opts.requires_grad = requires_grad;
    return opts;
}

TensorConfig torch_options_to_config(const TorchTensorOptions* opts) {
    if (!opts)
        return torch_config_default();
    return opts->config;
}

/* ------------------------------------------------------------------ */
/* Device & dtype                                                      */
/* ------------------------------------------------------------------ */

bool torch_cuda_is_available(void) { return device_cuda_available(); }

int torch_cuda_device_count(void) { return device_cuda_get_count(); }

DeviceType torch_get_default_device(void) { return torch_default_device_cached(); }

void torch_set_default_device(DeviceType device) {
    atomic_store_explicit(&g_torch_default_device_storage, (uint32_t)device, memory_order_relaxed);
    cml_set_default_device(device);
}

DType torch_get_default_dtype(void) { return torch_default_dtype_cached(); }

void torch_set_default_dtype(DType dtype) {
    atomic_store_explicit(&g_torch_default_dtype_storage, (uint32_t)dtype, memory_order_relaxed);
    cml_set_default_dtype(dtype);
}

void torch_manual_seed(uint64_t seed) { cml_manual_seed(seed); }

/* ------------------------------------------------------------------ */
/* Tensor lifecycle & accessors                                        */
/* ------------------------------------------------------------------ */

void torch_tensor_retain(Tensor* t) { torch_tensor_retain_fast(t); }

void torch_tensor_free(Tensor* t) { tensor_free(t); }

int torch_tensor_ref_count(const Tensor* t) {
    if (!t)
        return 0;
    return t->ref_count;
}

int torch_tensor_ndim(const Tensor* t) { return torch_tensor_ndim_fast(t); }

size_t torch_tensor_numel(const Tensor* t) { return torch_tensor_numel_fast(t); }

DType torch_tensor_dtype(const Tensor* t) { return torch_tensor_dtype_fast(t); }

DeviceType torch_tensor_device(const Tensor* t) { return torch_tensor_device_fast(t); }

bool torch_tensor_is_contiguous(const Tensor* t) { return torch_tensor_is_contiguous_fast(t); }

bool torch_tensor_requires_grad(const Tensor* t) { return torch_tensor_requires_grad_fast(t); }

void torch_tensor_set_requires_grad(Tensor* t, bool requires_grad) {
    if (t)
        tensor_set_requires_grad(t, requires_grad);
}

const int* torch_tensor_sizes(const Tensor* t) { return torch_tensor_sizes_fast(t); }

void* torch_tensor_data_ptr(Tensor* t) {
    if (!t) {
        error_stack_push(CM_INVALID_ARGUMENT, "torch_tensor_data_ptr: null tensor", __FILE__, __LINE__,
                         __func__);
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

float* torch_tensor_data_ptr_f32(Tensor* t) {
    if (!t || t->dtype != DTYPE_FLOAT32) {
        error_stack_push(CM_INVALID_ARGUMENT, "torch_tensor_data_ptr_f32: expected float32 tensor",
                         __FILE__, __LINE__, __func__);
        return NULL;
    }
    return (float*)torch_tensor_data_ptr(t);
}

bool torch_tensor_is_materialized(const Tensor* t) {
    return t != NULL && t->ir_node == NULL && t->is_executed;
}

bool torch_tensor_has_lazy_ir(const Tensor* t) { return t != NULL && t->ir_node != NULL; }

float torch_tensor_item_float(Tensor* t) {
    if (!t || t->numel != 1 || t->dtype != DTYPE_FLOAT32) {
        error_stack_push(CM_INVALID_ARGUMENT, "torch_tensor_item_float: requires scalar float32",
                         __FILE__, __LINE__, __func__);
        return 0.0f;
    }
    return tensor_get_float(t, 0);
}

void torch_tensor_set_item_float(Tensor* t, float value) {
    if (!t || t->numel != 1 || t->dtype != DTYPE_FLOAT32) {
        error_stack_push(CM_INVALID_ARGUMENT, "torch_tensor_set_item_float: requires scalar float32",
                         __FILE__, __LINE__, __func__);
        return;
    }
    tensor_set_float(t, 0, value);
}

Tensor* torch_empty(int* shape, int ndim, const TorchTensorOptions* opts) {
    return torch_create_tensor(tensor_empty, shape, ndim, opts);
}

Tensor* torch_zeros(int* shape, int ndim, const TorchTensorOptions* opts) {
    return torch_create_tensor(tensor_zeros, shape, ndim, opts);
}

Tensor* torch_ones(int* shape, int ndim, const TorchTensorOptions* opts) {
    return torch_create_tensor(tensor_ones, shape, ndim, opts);
}

Tensor* torch_full(int* shape, int ndim, const TorchTensorOptions* opts, float value) {
    TensorConfig scratch;
    const TensorConfig* cfg = torch_resolve_config(opts, &scratch);
    Tensor* t               = tensor_full(shape, ndim, cfg, value);
    if (t && opts && opts->requires_grad)
        t->requires_grad = true;
    return t;
}

Tensor* torch_rand(int* shape, int ndim, const TorchTensorOptions* opts) {
    return torch_create_tensor(tensor_rand, shape, ndim, opts);
}

Tensor* torch_randn(int* shape, int ndim, const TorchTensorOptions* opts) {
    return torch_create_tensor(tensor_randn, shape, ndim, opts);
}

Tensor* torch_eye(int n, const TorchTensorOptions* opts) {
    TensorConfig scratch;
    const TensorConfig* cfg = torch_resolve_config(opts, &scratch);
    Tensor* t               = tensor_eye(n, cfg);
    if (t && opts && opts->requires_grad)
        t->requires_grad = true;
    return t;
}

Tensor* torch_arange(float start, float end, float step, const TorchTensorOptions* opts) {
    TensorConfig scratch;
    const TensorConfig* cfg = torch_resolve_config(opts, &scratch);
    Tensor* t               = tensor_arange(start, end, step, cfg);
    if (t && opts && opts->requires_grad)
        t->requires_grad = true;
    return t;
}

Tensor* torch_linspace(float start, float end, int steps, const TorchTensorOptions* opts) {
    TensorConfig scratch;
    const TensorConfig* cfg = torch_resolve_config(opts, &scratch);
    Tensor* t               = tensor_linspace(start, end, steps, cfg);
    if (t && opts && opts->requires_grad)
        t->requires_grad = true;
    return t;
}

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

Tensor* torch_zeros_like(Tensor* t) { return t ? tensor_zeros_like(t) : NULL; }

Tensor* torch_ones_like(Tensor* t) { return t ? tensor_ones_like(t) : NULL; }

Tensor* torch_randn_like(Tensor* t) { return t ? tensor_randn_like(t) : NULL; }

/* ------------------------------------------------------------------ */
/* Tensor operations (direct autograd forward ops)                     */
/* ------------------------------------------------------------------ */

Tensor* torch_add(Tensor* a, Tensor* b) {
    Tensor* e = torch_eager_binary(UOP_ADD, a, b);
    return e ? e : tensor_add(a, b);
}
Tensor* torch_sub(Tensor* a, Tensor* b) {
    Tensor* e = torch_eager_binary(UOP_SUB, a, b);
    return e ? e : tensor_sub(a, b);
}
Tensor* torch_mul(Tensor* a, Tensor* b) {
    Tensor* e = torch_eager_binary(UOP_MUL, a, b);
    return e ? e : tensor_mul(a, b);
}
Tensor* torch_div(Tensor* a, Tensor* b) {
    Tensor* e = torch_eager_binary(UOP_DIV, a, b);
    return e ? e : tensor_div(a, b);
}
Tensor* torch_matmul(Tensor* a, Tensor* b) {
    Tensor* e = torch_eager_binary(UOP_MATMUL, a, b);
    return e ? e : tensor_matmul(a, b);
}
Tensor* torch_pow(Tensor* a, Tensor* b) { return tensor_pow(a, b); }

Tensor* torch_sum(Tensor* a, int dim, bool keepdim) { return tensor_sum(a, dim, keepdim); }
Tensor* torch_mean(Tensor* a, int dim, bool keepdim) { return tensor_mean(a, dim, keepdim); }
Tensor* torch_max(Tensor* a, int dim, bool keepdim) { return tensor_max(a, dim, keepdim); }
Tensor* torch_min(Tensor* a, int dim, bool keepdim) { return tensor_min(a, dim, keepdim); }

Tensor* torch_relu(Tensor* a) {
    Tensor* e = torch_eager_unary(UOP_RELU, a);
    return e ? e : tensor_relu(a);
}
Tensor* torch_sigmoid(Tensor* a) {
    Tensor* e = torch_eager_unary(UOP_SIGMOID, a);
    return e ? e : tensor_sigmoid(a);
}
Tensor* torch_tanh(Tensor* a) {
    Tensor* e = torch_eager_unary(UOP_TANH, a);
    return e ? e : tensor_tanh(a);
}
Tensor* torch_softmax(Tensor* a, int dim) { return tensor_softmax(a, dim); }
Tensor* torch_gelu(Tensor* a) { return uop_gelu(a); }

Tensor* torch_reshape(Tensor* a, int* new_shape, int new_ndim) {
    return tensor_reshape(a, new_shape, new_ndim);
}
Tensor* torch_transpose(Tensor* a, int dim0, int dim1) { return tensor_transpose(a, dim0, dim1); }
Tensor* torch_squeeze(Tensor* a, int dim) { return tensor_squeeze(a, dim); }
Tensor* torch_unsqueeze(Tensor* a, int dim) { return tensor_unsqueeze(a, dim); }
Tensor* torch_cat(Tensor** tensors, int num_tensors, int dim) {
    return tensor_concat(tensors, num_tensors, dim);
}
Tensor* torch_stack(Tensor** tensors, int num_tensors, int dim) {
    return tensor_stack(tensors, num_tensors, dim);
}
Tensor* torch_clone(Tensor* a) { return tensor_clone(a); }
Tensor* torch_detach(Tensor* a) { return tensor_detach(a); }
Tensor* torch_contiguous(Tensor* a) { return tensor_contiguous(a); }

/* ------------------------------------------------------------------ */
/* Autograd (direct engine calls)                                      */
/* ------------------------------------------------------------------ */

void torch_backward(Tensor* tensor, Tensor* gradient, bool retain_graph, bool create_graph) {
    tensor_backward(tensor, gradient, retain_graph, create_graph);
}

void torch_zero_grad(Tensor* tensor) { tensor_zero_grad(tensor); }

void torch_no_grad(void) { autograd_no_grad_enter(); }

void torch_enable_grad(void) { autograd_set_grad_mode(true); }

bool torch_is_grad_enabled(void) { return autograd_is_grad_enabled(); }

Tensor* torch_get_grad(Tensor* t) { return tensor_get_grad(t); }

/* ------------------------------------------------------------------ */
/* nn.Module API                                                       */
/* ------------------------------------------------------------------ */

Tensor* torch_module_forward(Module* module, Tensor* input) {
    return module_forward(module, input);
}

void torch_module_train(Module* module) { module_set_training(module, true); }

void torch_module_eval(Module* module) { module_set_training(module, false); }

bool torch_module_is_training(Module* module) { return module_is_training(module); }

void torch_module_zero_grad(Module* module) { module_zero_grad(module); }

Linear* torch_nn_linear(int in_features, int out_features, bool bias) {
    return cml_nn_linear(in_features, out_features, torch_default_dtype_cached(),
                         torch_default_device_cached(), bias);
}

ReLU* torch_nn_relu(void) { return cml_nn_relu(false); }

Sequential* torch_nn_sequential(void) { return cml_nn_sequential(); }

void torch_nn_sequential_add(Sequential* seq, Module* layer) {
    cml_nn_sequential_add(seq, layer);
}

Tensor* torch_nn_sequential_forward(Sequential* seq, Tensor* input) {
    return cml_nn_sequential_forward(seq, input);
}

/* ------------------------------------------------------------------ */
/* State dict                                                          */
/* ------------------------------------------------------------------ */

StateDict* torch_module_state_dict(Module* module, const char* prefix) {
    return nn_get_state_dict(module, prefix);
}

int torch_module_load_state_dict(Module* module, const StateDict* sd, bool strict) {
    return nn_load_state_dict(module, sd, strict);
}

Tensor* torch_state_dict_get(const StateDict* sd, const char* key) {
    return nn_state_dict_get(sd, key);
}

void torch_state_dict_free(StateDict* sd) { nn_state_dict_free(sd); }

/* ------------------------------------------------------------------ */
/* Optimizers                                                          */
/* ------------------------------------------------------------------ */

Optimizer* torch_optim_adam(Module* model, float lr, float weight_decay, float beta1, float beta2,
                            float eps) {
    return cml_optim_adam_for_model(model, lr, weight_decay, beta1, beta2, eps);
}

Optimizer* torch_optim_sgd(Module* model, float lr, float momentum, float weight_decay) {
    return cml_optim_sgd_for_model(model, lr, momentum, weight_decay);
}

void torch_optim_zero_grad(Optimizer* optimizer) { cml_optim_zero_grad(optimizer); }

void torch_optim_step(Optimizer* optimizer) { cml_optim_step(optimizer); }

void torch_optim_free(Optimizer* optimizer) { optimizer_free(optimizer); }

/* ------------------------------------------------------------------ */
/* Loss functions                                                      */
/* ------------------------------------------------------------------ */

Tensor* torch_nn_mse_loss(Tensor* input, Tensor* target) {
    return cml_nn_mse_loss(input, target);
}

Tensor* torch_nn_cross_entropy_loss(Tensor* input, Tensor* target) {
    return cml_nn_cross_entropy_loss(input, target);
}

/* ------------------------------------------------------------------ */
/* Runtime module                                                      */
/* ------------------------------------------------------------------ */

static TorchRuntimeModule* torch_runtime_alloc(TorchRuntimeKind kind) {
    TorchRuntimeModule* rt = (TorchRuntimeModule*)calloc(1, sizeof(TorchRuntimeModule));
    if (rt)
        rt->kind = kind;
    return rt;
}

TorchRuntimeModule* torch_runtime_from_module(Module* module) {
    if (!module)
        return NULL;
    TorchRuntimeModule* rt = torch_runtime_alloc(TORCH_RUNTIME_EAGER);
    if (!rt)
        return NULL;
    rt->eager_module = module;
    return rt;
}

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

int torch_runtime_export_pte(Module* module, Tensor* sample_input, const char* path,
                             const TorchPTEExportOptions* opts) {
    return torch_pte_export_module(module, sample_input, path, opts);
}

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

__attribute__((hot)) Tensor* torch_runtime_forward(TorchRuntimeModule* runtime, Tensor* input) {
    if (!runtime || !input)
        return NULL;

    switch (runtime->kind) {
    case TORCH_RUNTIME_EAGER:
        return module_forward(runtime->eager_module, input);

    case TORCH_RUNTIME_AOT: {
        if (!runtime->aot_model)
            return NULL;
        Tensor* inputs[]  = {input};
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

TorchRuntimeKind torch_runtime_get_kind(const TorchRuntimeModule* runtime) {
    return runtime ? runtime->kind : TORCH_RUNTIME_EAGER;
}

bool torch_runtime_has_memory(const TorchRuntimeModule* runtime) {
    return runtime != NULL && runtime->memory != NULL;
}

size_t torch_runtime_pte_arena_size(const TorchRuntimeModule* runtime) {
    if (!runtime || runtime->kind != TORCH_RUNTIME_PTE || !runtime->pte_model)
        return 0;
    return torch_pte_get_required_arena_size(runtime->pte_model);
}

/* ------------------------------------------------------------------ */
/* IR helpers                                                          */
/* ------------------------------------------------------------------ */

void torch_reset_ir(void) { cml_ir_reset_global_context(); }

void torch_reset_ir_soft(void) { cml_ir_reset_graph_only(); }

/* ------------------------------------------------------------------ */
/* Error handling                                                      */
/* ------------------------------------------------------------------ */

const char* torch_get_last_error(void) { return error_stack_get_last_message(); }

int torch_get_last_error_code(void) { return error_stack_get_last_code(); }

bool torch_has_error(void) { return error_stack_has_errors(); }

void torch_clear_error(void) { error_stack_clear(); }
