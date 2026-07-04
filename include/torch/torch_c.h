/*
 * torch_c.h — PyTorch-like C API for C-ML
 *
 * Inspired by ExecuTorch's runtime façade (Section 5.2): a thin, PyTorch-native
 * surface over the C-ML core runtime.  Provides eager-mode tensor ops, Module
 * forward, state_dict, device/options, and AOT runtime loading — all without
 * requiring Python or C++.
 *
 * Usage:
 *   #include "torch/torch_c.h"
 *   torch_init();
 *   TorchTensorOptions opts = torch_options();
 *   opts = torch_options_dtype(opts, DTYPE_FLOAT32);
 *   int shape[] = {2, 3};
 *   Tensor* t = torch_zeros(shape, 2, &opts);
 *   ...
 *   torch_tensor_free(t);
 *   torch_cleanup();
 */

#ifndef CML_TORCH_C_H
#define CML_TORCH_C_H

#include "core/export.h"
#include "tensor/tensor.h"
#include "nn.h"
#include "nn/layers.h"
#include "optim.h"
#include "nn/state.h"
#include "ops/ir/aot.h"
#include "torch/pte.h"
#include "torch/memory.h"
#include "torch/delegate.h"
#include "torch/selective_build.h"
#include "torch/torch_eager.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Lifecycle                                                           */
/* ------------------------------------------------------------------ */

CML_API int  torch_init(void);
CML_API int  torch_cleanup(void);
CML_API void torch_get_version(int* major, int* minor, int* patch, const char** version_string);

/* ------------------------------------------------------------------ */
/* TensorOptions (PyTorch TensorOptions analogue)                      */
/* ------------------------------------------------------------------ */

typedef struct TorchTensorOptions {
    DType      dtype;
    DeviceType device;
    bool       requires_grad;
    bool       has_dtype;
    bool       has_device;
    TensorConfig config; /* Precomputed; always valid after builder calls. */
} TorchTensorOptions;

CML_API TorchTensorOptions torch_options(void);
CML_API TorchTensorOptions torch_options_dtype(TorchTensorOptions opts, DType dtype);
CML_API TorchTensorOptions torch_options_device(TorchTensorOptions opts, DeviceType device);
CML_API TorchTensorOptions torch_options_requires_grad(TorchTensorOptions opts, bool requires_grad);

/* Convert to internal TensorConfig for C-ML calls. */
CML_API TensorConfig torch_options_to_config(const TorchTensorOptions* opts);

/* ------------------------------------------------------------------ */
/* Device & dtype defaults                                             */
/* ------------------------------------------------------------------ */

CML_API bool       torch_cuda_is_available(void);
CML_API int        torch_cuda_device_count(void);
CML_API DeviceType torch_get_default_device(void);
CML_API void       torch_set_default_device(DeviceType device);
CML_API DType      torch_get_default_dtype(void);
CML_API void       torch_set_default_dtype(DType dtype);
CML_API void       torch_manual_seed(uint64_t seed);

/* ------------------------------------------------------------------ */
/* Tensor lifecycle & accessors (TensorPtr-style zero-copy handles)      */
/* ------------------------------------------------------------------ */

CML_API Tensor* torch_empty(int* shape, int ndim, const TorchTensorOptions* opts);
CML_API Tensor* torch_zeros(int* shape, int ndim, const TorchTensorOptions* opts);
CML_API Tensor* torch_ones(int* shape, int ndim, const TorchTensorOptions* opts);
CML_API Tensor* torch_full(int* shape, int ndim, const TorchTensorOptions* opts, float value);
CML_API Tensor* torch_rand(int* shape, int ndim, const TorchTensorOptions* opts);
CML_API Tensor* torch_randn(int* shape, int ndim, const TorchTensorOptions* opts);
CML_API Tensor* torch_eye(int n, const TorchTensorOptions* opts);
CML_API Tensor* torch_arange(float start, float end, float step, const TorchTensorOptions* opts);
CML_API Tensor* torch_linspace(float start, float end, int steps, const TorchTensorOptions* opts);

/* Wrap caller-owned storage (not copied). `data` must be non-NULL; returns NULL on
 * invalid arguments. The tensor does not take ownership of `data`. */
CML_API Tensor* torch_from_blob(void* data, int* shape, int ndim, const TorchTensorOptions* opts);

CML_API Tensor* torch_zeros_like(Tensor* t);
CML_API Tensor* torch_ones_like(Tensor* t);
CML_API Tensor* torch_randn_like(Tensor* t);

CML_API void    torch_tensor_retain(Tensor* t);
CML_API void    torch_tensor_free(Tensor* t);
CML_API int     torch_tensor_ref_count(const Tensor* t);

CML_API int     torch_tensor_ndim(const Tensor* t);
CML_API size_t  torch_tensor_numel(const Tensor* t);
CML_API DType   torch_tensor_dtype(const Tensor* t);
CML_API DeviceType torch_tensor_device(const Tensor* t);
CML_API bool    torch_tensor_is_contiguous(const Tensor* t);
CML_API bool    torch_tensor_requires_grad(const Tensor* t);
CML_API void    torch_tensor_set_requires_grad(Tensor* t, bool requires_grad);

/* Returns shape pointer (valid until tensor is freed; do not free). */
CML_API const int* torch_tensor_sizes(const Tensor* t);

/* Materializes lazy tensor and returns raw data pointer.
 * Returns NULL on failure (OOM, device error, realize error); check
 * torch_has_error() / torch_get_last_error() for details. */
CML_API void*   torch_tensor_data_ptr(Tensor* t);

/* Like torch_tensor_data_ptr but returns NULL unless dtype is DTYPE_FLOAT32. */
CML_API float*  torch_tensor_data_ptr_f32(Tensor* t);

/* True when the tensor is a materialized leaf (no pending IR node). */
CML_API bool    torch_tensor_is_materialized(const Tensor* t);

/* True when the tensor still references a lazy IR graph node. */
CML_API bool    torch_tensor_has_lazy_ir(const Tensor* t);

/* Scalar accessors: require numel()==1 and DTYPE_FLOAT32. Returns 0.0f / no-op on
 * invalid tensors; check torch_has_error() after failures. */
CML_API float   torch_tensor_item_float(Tensor* t);
CML_API void    torch_tensor_set_item_float(Tensor* t, float value);

/* ------------------------------------------------------------------ */
/* Tensor operations                                                   */
/* ------------------------------------------------------------------ */

CML_API Tensor* torch_add(Tensor* a, Tensor* b);
CML_API Tensor* torch_sub(Tensor* a, Tensor* b);
CML_API Tensor* torch_mul(Tensor* a, Tensor* b);
CML_API Tensor* torch_div(Tensor* a, Tensor* b);
CML_API Tensor* torch_matmul(Tensor* a, Tensor* b);
CML_API Tensor* torch_pow(Tensor* a, Tensor* b);

CML_API Tensor* torch_sum(Tensor* a, int dim, bool keepdim);
CML_API Tensor* torch_mean(Tensor* a, int dim, bool keepdim);
CML_API Tensor* torch_max(Tensor* a, int dim, bool keepdim);
CML_API Tensor* torch_min(Tensor* a, int dim, bool keepdim);

CML_API Tensor* torch_relu(Tensor* a);
CML_API Tensor* torch_sigmoid(Tensor* a);
CML_API Tensor* torch_tanh(Tensor* a);
CML_API Tensor* torch_softmax(Tensor* a, int dim);
CML_API Tensor* torch_gelu(Tensor* a);

CML_API Tensor* torch_reshape(Tensor* a, int* new_shape, int new_ndim);
CML_API Tensor* torch_transpose(Tensor* a, int dim0, int dim1);
CML_API Tensor* torch_squeeze(Tensor* a, int dim);
CML_API Tensor* torch_unsqueeze(Tensor* a, int dim);
CML_API Tensor* torch_cat(Tensor** tensors, int num_tensors, int dim);
CML_API Tensor* torch_stack(Tensor** tensors, int num_tensors, int dim);
CML_API Tensor* torch_clone(Tensor* a);
CML_API Tensor* torch_detach(Tensor* a);
CML_API Tensor* torch_contiguous(Tensor* a);

/* ------------------------------------------------------------------ */
/* Autograd                                                            */
/* ------------------------------------------------------------------ */

/* Backward pass. retain_graph and create_graph are accepted for API compatibility
 * but are not yet implemented (graph is always freed after backward). */
CML_API void torch_backward(Tensor* tensor, Tensor* gradient, bool retain_graph, bool create_graph);
CML_API void torch_zero_grad(Tensor* tensor);
CML_API void torch_no_grad(void);
CML_API void torch_enable_grad(void);
CML_API bool torch_is_grad_enabled(void);

CML_API Tensor* torch_get_grad(Tensor* t);

/* ------------------------------------------------------------------ */
/* nn.Module API (eager-mode, ExecuTorch Module analogue)              */
/* ------------------------------------------------------------------ */

CML_API Tensor* torch_module_forward(Module* module, Tensor* input);
CML_API void    torch_module_train(Module* module);
CML_API void    torch_module_eval(Module* module);
CML_API bool    torch_module_is_training(Module* module);
CML_API void    torch_module_zero_grad(Module* module);

CML_API Linear*       torch_nn_linear(int in_features, int out_features, bool bias);
CML_API ReLU*         torch_nn_relu(void);
CML_API Sequential*   torch_nn_sequential(void);
CML_API void          torch_nn_sequential_add(Sequential* seq, Module* layer);
CML_API Tensor*       torch_nn_sequential_forward(Sequential* seq, Tensor* input);

/* ------------------------------------------------------------------ */
/* State dict (torch.nn.Module.state_dict analogue)                    */
/* ------------------------------------------------------------------ */

CML_API StateDict* torch_module_state_dict(Module* module, const char* prefix);
CML_API int        torch_module_load_state_dict(Module* module, const StateDict* sd, bool strict);
CML_API Tensor*    torch_state_dict_get(const StateDict* sd, const char* key);
CML_API void       torch_state_dict_free(StateDict* sd);

/* ------------------------------------------------------------------ */
/* Optimizers                                                          */
/* ------------------------------------------------------------------ */

CML_API Optimizer* torch_optim_adam(Module* model, float lr, float weight_decay,
                                    float beta1, float beta2, float eps);
CML_API Optimizer* torch_optim_sgd(Module* model, float lr, float momentum, float weight_decay);
CML_API void       torch_optim_zero_grad(Optimizer* optimizer);
CML_API void       torch_optim_step(Optimizer* optimizer);
CML_API void       torch_optim_free(Optimizer* optimizer);

/* ------------------------------------------------------------------ */
/* Loss functions                                                      */
/* ------------------------------------------------------------------ */

CML_API Tensor* torch_nn_mse_loss(Tensor* input, Tensor* target);
CML_API Tensor* torch_nn_cross_entropy_loss(Tensor* input, Tensor* target);

/* ------------------------------------------------------------------ */
/* Runtime module (AOT / eager, ExecuTorch loaded-model analogue)      */
/* ------------------------------------------------------------------ */

typedef enum {
    TORCH_RUNTIME_EAGER = 0,
    TORCH_RUNTIME_AOT   = 1,
    TORCH_RUNTIME_PTE   = 2,
} TorchRuntimeKind;

typedef struct TorchRuntimeModule {
    TorchRuntimeKind kind;
    Module*          eager_module;
    CMLAOTModel*     aot_model;
    CMLPTEModel*     pte_model;
    TorchMemoryManager* memory;
    bool             owns_eager;
    bool             owns_memory;
} TorchRuntimeModule;

/* Prefer these accessors over reading struct fields directly. */
CML_API TorchRuntimeKind torch_runtime_get_kind(const TorchRuntimeModule* runtime);
CML_API bool             torch_runtime_has_memory(const TorchRuntimeModule* runtime);
CML_API size_t           torch_runtime_pte_arena_size(const TorchRuntimeModule* runtime);

/* Wrap an existing eager Module (does not take ownership). */
CML_API TorchRuntimeModule* torch_runtime_from_module(Module* module);

/* Load an AOT-compiled shared library (ExecuTorch PTE analogue). */
CML_API TorchRuntimeModule* torch_runtime_load_aot(const char* path);

/* Load a .cpte portable program (ExecuTorch PTE format). */
CML_API TorchRuntimeModule* torch_runtime_load_pte(const char* path);

/* Export a module to .cpte (ahead-of-time preparation). */
CML_API int torch_runtime_export_pte(Module* module, Tensor* sample_input, const char* path,
                                     const TorchPTEExportOptions* opts);

CML_API Tensor* torch_runtime_forward(TorchRuntimeModule* runtime, Tensor* input);
CML_API void    torch_runtime_free(TorchRuntimeModule* runtime);
CML_API void    torch_runtime_set_memory(TorchRuntimeModule* runtime, TorchMemoryManager* memory);

/* ------------------------------------------------------------------ */
/* IR / inference helpers                                              */
/* ------------------------------------------------------------------ */

CML_API void torch_reset_ir(void);
CML_API void torch_reset_ir_soft(void);

/* ------------------------------------------------------------------ */
/* Error handling                                                      */
/* ------------------------------------------------------------------ */

CML_API const char* torch_get_last_error(void);
CML_API int         torch_get_last_error_code(void);
CML_API bool        torch_has_error(void);
CML_API void        torch_clear_error(void);

/* Optional zero-overhead inlines for in-process hot loops. */
#include "torch/torch_c_inline.h"

#ifdef __cplusplus
}
#endif

#endif /* CML_TORCH_C_H */
