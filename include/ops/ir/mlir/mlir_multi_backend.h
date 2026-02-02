#ifndef CML_OPS_IR_MLIR_MULTI_BACKEND_H
#define CML_OPS_IR_MLIR_MULTI_BACKEND_H

#include "ops/ir/mlir/mlir_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Supported target backends
 */
typedef enum {
    MLIR_TARGET_CPU,    ///< Native CPU (LLVM IR -> Machine Code)
    MLIR_TARGET_CUDA,   ///< NVIDIA GPU (NVVM -> PTX)
    MLIR_TARGET_ROCM,   ///< AMD GPU (ROCDL -> HSACO)
    MLIR_TARGET_VULKAN, ///< Cross-platform GPU (SPIR-V)
    MLIR_TARGET_METAL,  ///< Apple GPU (Metal Shading Language)
    MLIR_TARGET_WEBGPU  ///< Web (WGSL)
} MLIRTargetBackend;

/**
 * @brief Set the target backend for the context
 * @param ctx MLIR context
 * @param target Target backend
 * @return 0 on success, -1 on failure
 */
int cml_mlir_set_target(CMLMLIRContext* ctx, MLIRTargetBackend target);

/**
 * @brief Get the current target backend
 * @param ctx MLIR context
 * @return Current target backend
 */
MLIRTargetBackend cml_mlir_get_target(CMLMLIRContext* ctx);

/**
 * @brief Compile module for specific target
 * @param ctx MLIR context
 * @param module MLIR module
 * @param target Target backend
 * @return Opaque handle to compiled binary (e.g., PTX string, SPIR-V blob), or NULL
 */
void* cml_mlir_compile_for_target(CMLMLIRContext* ctx, void* module, MLIRTargetBackend target);

/**
 * @brief Detect available compute devices
 *
 * Scans the system for available compute devices (CUDA, ROCm, Vulkan, Metal, etc.)
 * and logs the results. This should be called once at initialization.
 */
void cml_mlir_detect_devices(void);

/**
 * @brief Select the best available device automatically
 * @return Best available backend (priority: CUDA > Metal > Vulkan > ROCm > CPU)
 */
MLIRTargetBackend cml_mlir_select_best_device(void);

#ifdef __cplusplus
}
#endif

#endif // CML_OPS_IR_MLIR_MULTI_BACKEND_H
