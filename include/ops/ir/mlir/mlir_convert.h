#ifndef CML_OPS_IR_MLIR_CONVERT_H
#define CML_OPS_IR_MLIR_CONVERT_H

#include "ops/ir/ir.h"
#include "ops/ir/mlir/mlir_context.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef CML_HAS_MLIR

#include <mlir-c/IR.h>

/**
 * @brief Convert C-ML IR to MLIR module
 * @param ctx MLIR context (module will be stored in ctx->module)
 * @param ir C-ML IR to convert
 * @return true on success, false on failure
 *
 * The resulting module is accessible via ctx->module after success.
 */
bool cml_ir_to_mlir(CMLMLIRContext* ctx, CMLIR_t ir);

/**
 * @brief Convert C-ML IR Backward graph to MLIR module
 * @param ctx MLIR context (module will be stored in ctx->module)
 * @param ir C-ML IR to convert
 * @return true on success, false on failure
 */
bool cml_ir_backward_to_mlir(CMLMLIRContext* ctx, CMLIR_t ir);

#endif // CML_HAS_MLIR

#ifdef __cplusplus
}
#endif

#endif // CML_OPS_IR_MLIR_CONVERT_H
