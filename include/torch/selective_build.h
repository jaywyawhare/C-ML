/*
 * selective_build.h — Link only the kernels required by a deployed model
 *
 * ExecuTorch selective-build analogue: at compile or load time, restrict the
 * runtime to a subset of UOp kernels to reduce binary size on embedded targets.
 */

#ifndef CML_TORCH_SELECTIVE_BUILD_H
#define CML_TORCH_SELECTIVE_BUILD_H

#include "core/export.h"
#include "ops/uops.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CML_TORCH_MAX_SELECTIVE_DTYPES 16

typedef struct TorchSelectiveBuildConfig {
    bool     all_enabled;
    bool     enabled[UOP_COUNT];
    bool     dtype_enabled[CML_TORCH_MAX_SELECTIVE_DTYPES]; /* indexed by DType enum value */
} TorchSelectiveBuildConfig;

CML_API TorchSelectiveBuildConfig torch_selective_build_all(void);
CML_API TorchSelectiveBuildConfig torch_selective_build_none(void);

CML_API void torch_selective_build_enable_op(TorchSelectiveBuildConfig* cfg, UOpType op);
CML_API void torch_selective_build_enable_ops(TorchSelectiveBuildConfig* cfg,
                                              const UOpType* ops, int count);

CML_API void torch_selective_build_apply(const TorchSelectiveBuildConfig* cfg);
CML_API void torch_selective_build_reset(void);

CML_API bool torch_selective_build_is_op_enabled(UOpType op);
CML_API bool torch_selective_build_is_dtype_enabled(DType dtype);

/* Parse comma-separated UOp names, e.g. "add,mul,matmul,relu".
 * Names are case-insensitive; surrounding whitespace is ignored.
 * Unknown tokens are skipped. Valid names match torch_selective_build_save
 * output: add, sub, mul, div, matmul, relu, sigmoid, tanh, sum, mean,
 * quick_gelu, reshape, transpose, linear.
 * Returns 0 on success, -1 if spec or out is NULL. */
CML_API int torch_selective_build_from_string(const char* spec,
                                              TorchSelectiveBuildConfig* out);

/* Derive required ops from a loaded .cpte program */
CML_API int torch_selective_build_from_pte(const char* pte_path,
                                           TorchSelectiveBuildConfig* out);

CML_API int torch_selective_build_save(const TorchSelectiveBuildConfig* cfg,
                                       const char* path);
CML_API int torch_selective_build_load(const char* path,
                                       TorchSelectiveBuildConfig* out);

#ifdef __cplusplus
}
#endif

#endif /* CML_TORCH_SELECTIVE_BUILD_H */
