#ifndef CML_OPS_IR_MLIR_UOPS_BUILDER_H
#define CML_OPS_IR_MLIR_UOPS_BUILDER_H

#include "ops/ir/mlir/mlir_backend.h"
#include "ops/ir/ir.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Build an MLIR module from the C-ML IR graph using the MLIR context.
 *
 * This is the primary uops→MLIR builder. It walks the IRNode chain in @p ir
 * and populates @p ctx->module with a single entry function that represents
 * the graph. On success, the module is accessible via ctx->module.
 *
 * @param ctx Initialized MLIR context.
 * @param ir  C-ML IR graph built from uops.
 * @return true on success, false on failure.
 */
bool cml_mlir_build_from_ir(CMLMLIRContext* ctx, CMLIR_t ir);

#ifdef __cplusplus
}
#endif

#endif // CML_OPS_IR_MLIR_UOPS_BUILDER_H
