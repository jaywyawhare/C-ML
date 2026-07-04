/*
 * torch_c_internal.h — Internal helpers for torch_c hot paths (not public API).
 */

#ifndef CML_TORCH_C_INTERNAL_H
#define CML_TORCH_C_INTERNAL_H

#include "torch/torch_c.h"
#include "autograd/autograd.h"
#include "autograd/forward_ops.h"
#include "tensor/tensor_views.h"
#include "tensor/tensor_manipulation.h"
#include "ops/uops.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Cached defaults (synced in torch_set_default_*). Use accessors for thread-safe reads. */
DType      torch_default_dtype_cached(void);
DeviceType torch_default_device_cached(void);

static inline void torch_opts_sync_config(TorchTensorOptions* opts) {
    opts->config.dtype      = opts->has_dtype ? opts->dtype : torch_default_dtype_cached();
    opts->config.device     = opts->has_device ? opts->device : torch_default_device_cached();
    opts->config.has_dtype  = true;
    opts->config.has_device = true;
}

static inline TensorConfig torch_config_default(void) {
    TensorConfig cfg = {
        .dtype       = torch_default_dtype_cached(),
        .device      = torch_default_device_cached(),
        .has_dtype   = true,
        .has_device  = true,
    };
    return cfg;
}

static inline const TensorConfig* torch_resolve_config(const TorchTensorOptions* opts,
                                                       TensorConfig* scratch) {
    if (opts)
        return &opts->config;
    *scratch = torch_config_default();
    return scratch;
}

typedef Tensor* (*TorchCreateFn)(int* shape, int ndim, const TensorConfig* config);

static inline Tensor* torch_create_tensor(TorchCreateFn fn, int* shape, int ndim,
                                          const TorchTensorOptions* opts) {
    TensorConfig scratch;
    const TensorConfig* cfg = torch_resolve_config(opts, &scratch);
    Tensor* t               = fn(shape, ndim, cfg);
    if (t && opts && opts->requires_grad)
        t->requires_grad = true;
    return t;
}

#ifdef __cplusplus
}
#endif

#endif /* CML_TORCH_C_INTERNAL_H */
