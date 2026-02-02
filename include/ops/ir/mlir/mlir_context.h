#ifndef CML_OPS_IR_MLIR_CONTEXT_H
#define CML_OPS_IR_MLIR_CONTEXT_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Execution modes for C-ML operations
 */
typedef enum {
    CML_EXEC_INTERPRETED, ///< Use current hand-rolled codegen (fast compile, slower run)
    CML_EXEC_JIT,         ///< Use MLIR JIT compilation (slow compile, fast run)
    CML_EXEC_AOT          ///< Use pre-compiled kernels (no compile, fastest run)
} CMLExecutionMode;

/**
 * @brief MLIR backend status
 */
typedef enum {
    CML_MLIR_NOT_AVAILABLE, ///< MLIR not compiled in
    CML_MLIR_AVAILABLE,     ///< MLIR available but not initialized
    CML_MLIR_INITIALIZED,   ///< MLIR initialized and ready
    CML_MLIR_ERROR          ///< MLIR encountered an error
} CMLMLIRStatus;

// Forward declarations
typedef struct CMLMLIRContext CMLMLIRContext;

/**
 * @brief Check if MLIR support is available (compiled in)
 * @return true if MLIR support is available, false otherwise
 */
bool cml_mlir_is_available(void);

/**
 * @brief Get current MLIR status
 * @return Current MLIR status
 */
CMLMLIRStatus cml_mlir_get_status(void);

/**
 * @brief Initialize MLIR subsystem
 * @return MLIR context on success, NULL on failure
 *
 * This function initializes the MLIR context, registers dialects,
 * and prepares the JIT compilation engine.
 */
CMLMLIRContext* cml_mlir_init(void);

/**
 * @brief Destroy MLIR context and free resources
 * @param ctx MLIR context to destroy
 */
void cml_mlir_destroy(CMLMLIRContext* ctx);

/**
 * @brief Set global execution mode
 * @param mode Execution mode to use
 *
 * This affects all subsequent tensor operations.
 * Default is CML_EXEC_INTERPRETED.
 */
void cml_set_execution_mode(CMLExecutionMode mode);

/**
 * @brief Get current execution mode
 * @return Current execution mode
 */
CMLExecutionMode cml_get_execution_mode(void);

/**
 * @brief Enable or disable JIT compilation
 * @param enable true to enable JIT, false to use interpreted mode
 *
 * Convenience function equivalent to:
 * cml_set_execution_mode(enable ? CML_EXEC_JIT : CML_EXEC_INTERPRETED)
 */
void cml_enable_jit(bool enable);

#ifdef __cplusplus
}
#endif

#endif // CML_OPS_IR_MLIR_CONTEXT_H
