#ifndef CML_OPS_IR_MLIR_FUSION_H
#define CML_OPS_IR_MLIR_FUSION_H

#include "ops/ir/ir.h"
#include "ops/ir/mlir/mlir_context.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Configuration for MLIR fusion passes
 */
typedef struct {
    bool enable_linalg_fusion;    // Fuse linalg operations
    bool enable_vectorization;    // Auto-vectorize (SIMD)
    bool enable_buffer_fusion;    // Fuse buffer allocations
    bool enable_loop_fusion;      // Fuse loops
    bool enable_constant_folding; // Fold constants
    bool enable_cse;              // Common subexpression elimination
    int vectorization_size;       // Vector width (0 = auto)
    int tile_size;                // Tile size for loop tiling (0 = no tiling)
} MLIRFusionConfig;

/**
 * @brief Create default fusion configuration
 * @return Allocated config (must be freed)
 */
MLIRFusionConfig* cml_mlir_fusion_config_create(void);

/**
 * @brief Destroy fusion configuration
 */
void cml_mlir_fusion_config_destroy(MLIRFusionConfig* cfg);

/**
 * @brief Apply MLIR optimization passes with custom config
 * @param mlir_module Opaque MLIR module pointer
 * @param mlir_context Opaque MLIR context pointer
 * @param config Fusion configuration (NULL = use defaults)
 * @return 0 on success, -1 on failure
 */
int cml_mlir_apply_fusion_passes(const void* mlir_module, const void* mlir_context,
                                 const MLIRFusionConfig* config);

/**
 * @brief Apply MLIR optimization passes to module
 * @param mlir_module Opaque MLIR module pointer
 * @param mlir_context Opaque MLIR context pointer
 * @return 0 on success, -1 on failure
 *
 * Applies comprehensive optimization including:
 * - Common subexpression elimination
 * - Constant folding
 * - Operation fusion
 * - Vectorization
 * - Buffer optimization
 */
int cml_mlir_optimize(const void* mlir_module, const void* mlir_context);

/**
 * @brief Apply full two-level fusion pipeline
 * @param ir C-ML IR (will be optimized in-place)
 * @param ctx MLIR context
 * @return Optimized MLIR module
 *
 * Two-level pipeline:
 * 1. C-ML IR fusion (9 patterns)
 * 2. MLIR polyhedral optimization
 */
const void* cml_apply_full_fusion_pipeline(CMLIR_t ir, CMLMLIRContext* ctx);

/**
 * @brief Dump MLIR module as text (for debugging)
 * @param mlir_module Opaque MLIR module pointer
 * @return Allocated string with MLIR IR text (caller must free)
 */
char* cml_mlir_dump_module(void* mlir_module);

#ifdef __cplusplus
}
#endif

#endif // CML_OPS_IR_MLIR_FUSION_H
