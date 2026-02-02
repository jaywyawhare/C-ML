#ifndef CML_OPS_IR_MLIR_EXECUTE_H
#define CML_OPS_IR_MLIR_EXECUTE_H

#include "ops/ir/ir.h"
#include "ops/ir/mlir/mlir_context.h"
#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct CMLJITEngine CMLJITEngine;
typedef void (*CMLJITKernelFunction)(void** args);

/**
 * @brief Create JIT compilation engine
 * @return JIT engine on success, NULL on failure
 */
CMLJITEngine* cml_jit_engine_create(void);

/**
 * @brief Destroy JIT engine
 * @param engine JIT engine to destroy
 */
void cml_jit_engine_destroy(CMLJITEngine* engine);

/**
 * @brief Compile IR to executable kernel (with caching)
 * @param engine JIT engine
 * @param ir C-ML IR to compile
 * @return Compiled kernel function, NULL on failure
 *
 * This function checks cache first. If IR hash is found,
 * returns cached kernel. Otherwise, compiles and caches.
 */
CMLJITKernelFunction cml_jit_compile_kernel(CMLJITEngine* engine, CMLIR_t ir);

#ifdef CML_HAS_MLIR
/**
 * @brief JIT compile MLIR module to executable function
 * @param ctx MLIR context
 * @param mlir_module Opaque MLIR module pointer (from cml_ir_to_mlir)
 * @return Function pointer that can be executed, NULL on failure
 */
CMLJITKernelFunction cml_mlir_jit_compile(CMLMLIRContext* ctx, void* mlir_module);

/**
 * @brief Compile MLIR module to object file (AOT)
 * @param mlir_module Opaque MLIR module pointer
 * @param output_path Path to write object file
 * @return 0 on success, negative on error
 */
int cml_mlir_compile_to_object(const void* mlir_module, const char* output_path);
#endif // CML_HAS_MLIR

/**
 * @brief Execute compiled kernel
 * @param fn Compiled kernel function
 * @param inputs Array of input tensor data pointers
 * @param num_inputs Number of input tensors
 * @param output Output tensor data pointer
 */
void cml_jit_execute(CMLJITKernelFunction fn, float** inputs, int num_inputs, float* output);

/**
 * @brief Execute MLIR module with tensor inputs/outputs
 * @param engine MLIR execution engine
 * @param inputs Array of input tensors
 * @param num_inputs Number of input tensors
 * @param outputs Array of output tensors
 * @param num_outputs Number of output tensors
 * @return 0 on success, -1 on failure
 */
int cml_mlir_execute(void* engine, Tensor** inputs, int num_inputs, Tensor** outputs,
                     int num_outputs);

/**
 * @brief Clear JIT kernel cache
 */
void cml_jit_cache_clear(void);

/**
 * @brief Set maximum JIT cache size in bytes
 * @param max_bytes Maximum cache size (0 = unlimited)
 */
void cml_jit_cache_set_size(size_t max_bytes);

/**
 * @brief Get current JIT cache statistics
 * @param hits Output: Number of cache hits
 * @param misses Output: Number of cache misses
 * @param size_bytes Output: Current cache size in bytes
 */
void cml_jit_cache_stats(size_t* hits, size_t* misses, size_t* size_bytes);

#ifdef __cplusplus
}
#endif

#endif // CML_OPS_IR_MLIR_EXECUTE_H
