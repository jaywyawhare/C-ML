#ifndef CML_OPS_IR_MLIR_CPP_BRIDGE_H
#define CML_OPS_IR_MLIR_CPP_BRIDGE_H

#include "core/logging.h"

#ifdef CML_HAS_MLIR
#include <mlir-c/IR.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Register every upstream MLIR dialect via the C++ utility helpers.
 */
int cml_mlir_register_all_dialects_cpp(MlirDialectRegistry registry);

/**
 * @brief Register all upstream MLIR extensions (conversion + transform) that
 *        are only available through the C++ API.
 *
 * This ensures conversion interfaces such as Func → LLVM are attached so that
 * LLVM translation works when using the C API elsewhere in the project.
 *
 * @param registry Dialect registry to extend.
 * @return 0 on success, -1 on failure.
 */
int cml_mlir_register_all_extensions(MlirDialectRegistry registry);

/**
 * @brief Register every upstream MLIR pass via C++.
 *
 * Should be called once during startup so passes referenced by MLIR pipelines
 * are populated.
 */
int cml_mlir_register_all_passes_cpp(void);

/**
 * @brief Register the Func → LLVM translation interface explicitly.
 *
 * This is exposed separately so callers can register just the Func extension
 * if desired.
 *
 * @param registry Dialect registry to extend.
 * @return 0 on success, -1 on failure.
 */
int cml_mlir_register_func_to_llvm_interface(MlirDialectRegistry registry);

/**
 * @brief Register all LLVM IR translations via the C++ API (registry level).
 *
 * @param registry Dialect registry to extend with translation interfaces.
 * @return 0 on success, -1 otherwise.
 */
int cml_mlir_register_all_llvm_translations(MlirDialectRegistry registry);

/**
 * @brief Register the GPU-only LLVM IR translations via C++ helpers.
 */
int cml_mlir_register_all_gpu_llvm_translations(MlirDialectRegistry registry);

/**
 * @brief Register all "from LLVM IR" translations (LLVM → MLIR) via C++ helpers.
 */
int cml_mlir_register_all_from_llvm_translations(MlirDialectRegistry registry);

/**
 * @brief Run the canonical tensor→LLVM lowering pipeline implemented in C++.
 *
 * @param module MLIR module to lower in-place.
 * @return 0 on success, -1 on failure.
 */
int cml_mlir_lower_module_to_llvm(MlirModule module);

/**
 * @brief Create an array of identity affine maps for linalg operations.
 *
 * @param ctx MLIR context.
 * @param num_maps Number of identity maps to create.
 * @param rank Rank of the maps.
 * @return MlirAttribute containing the array of affine maps.
 */
MlirAttribute cml_mlir_create_indexing_maps_attr(MlirContext ctx, int num_maps, int rank);

/**
 * @brief Create affine maps for transpose operation (2D).
 *
 * @param ctx MLIR context.
 * @return MlirAttribute containing the array of affine maps for transpose.
 */
MlirAttribute cml_mlir_create_transpose_maps_attr(MlirContext ctx);
/**
 * @brief Create affine maps for broadcast operation (last dim).
 *
 * @param ctx MLIR context.
 * @param rank Rank of the operation.
 * @return MlirAttribute containing the array of affine maps.
 */
MlirAttribute cml_mlir_create_broadcast_maps_attr(MlirContext ctx, int rank);
/**
 * @brief Create affine maps for scalar broadcast operation.
 *
 * @param ctx MLIR context.
 * @param rank Rank of the tensor operand.
 * @return MlirAttribute containing the array of affine maps.
 */
MlirAttribute cml_mlir_create_scalar_broadcast_maps_attr(MlirContext ctx, int rank);
MlirAttribute cml_mlir_create_reduction_maps_attr(MlirContext ctx, int rank, int* reduced_dims,
                                                  int num_reduced_dims, int out_rank);

/**
 * @brief GPU backend type for lowering
 */
typedef enum CMLGPUBackend {
    CML_GPU_BACKEND_CUDA = 0, // NVIDIA CUDA (NVVM)
    CML_GPU_BACKEND_ROCM,     // AMD ROCm (ROCDL)
    CML_GPU_BACKEND_SPIRV,    // Vulkan/OpenCL (SPIR-V)
    CML_GPU_BACKEND_METAL     // Apple Metal (via SPIR-V Cross)
} CMLGPUBackend;

/**
 * @brief Register GPU-related dialects (gpu, nvvm, rocdl, spirv)
 *
 * @param registry Dialect registry to extend
 * @return 0 on success, -1 on failure
 */
int cml_mlir_register_gpu_dialects(MlirDialectRegistry registry);

/**
 * @brief Lower module to GPU dialect (parallel loops → GPU kernels)
 *
 * @param module MLIR module to lower in-place
 * @return 0 on success, -1 on failure
 */
int cml_mlir_lower_to_gpu(MlirModule module);

/**
 * @brief Lower GPU dialect to target-specific dialect
 *
 * @param module MLIR module to lower in-place
 * @param backend Target GPU backend (CUDA, ROCm, SPIRV)
 * @return 0 on success, -1 on failure
 */
int cml_mlir_lower_gpu_to_target(MlirModule module, CMLGPUBackend backend);

/**
 * @brief Run the full GPU lowering pipeline for a specific backend
 *
 * Combines: Linalg → SCF → GPU → Target → LLVM
 *
 * @param module MLIR module to lower in-place
 * @param backend Target GPU backend
 * @return 0 on success, -1 on failure
 */
int cml_mlir_lower_module_to_gpu(MlirModule module, CMLGPUBackend backend);

/**
 * @brief Serialize SPIR-V module to binary
 *
 * @param module MLIR module containing SPIR-V dialect
 * @param output Output buffer (caller must free)
 * @param output_size Size of output buffer
 * @return 0 on success, -1 on failure
 */
int cml_mlir_serialize_spirv(MlirModule module, void** output, size_t* output_size);

#ifdef __cplusplus
}
#endif

#else

typedef void* MlirDialectRegistry;
typedef void* MlirModule;
typedef void* MlirContext;
typedef void* MlirAttribute;

static inline int cml_mlir_register_all_dialects_cpp(MlirDialectRegistry registry) {
    (void)registry;
    LOG_ERROR("MLIR support not available");
    return -1;
}

static inline int cml_mlir_register_all_extensions(MlirDialectRegistry registry) {
    (void)registry;
    LOG_ERROR("MLIR support not available");
    return -1;
}

static inline int cml_mlir_register_all_passes_cpp(void) {
    LOG_ERROR("MLIR support not available");
    return -1;
}

static inline int cml_mlir_register_func_to_llvm_interface(MlirDialectRegistry registry) {
    (void)registry;
    LOG_ERROR("MLIR support not available");
    return -1;
}

static inline int cml_mlir_register_all_llvm_translations(MlirDialectRegistry registry) {
    (void)registry;
    LOG_ERROR("MLIR support not available");
    return -1;
}

static inline int cml_mlir_register_all_gpu_llvm_translations(MlirDialectRegistry registry) {
    (void)registry;
    LOG_ERROR("MLIR support not available");
    return -1;
}

static inline int cml_mlir_register_all_from_llvm_translations(MlirDialectRegistry registry) {
    (void)registry;
    LOG_ERROR("MLIR support not available");
    return -1;
}

static inline int cml_mlir_lower_module_to_llvm(MlirModule module) {
    (void)module;
    LOG_ERROR("MLIR support not available");
    return -1;
}
static inline MlirAttribute cml_mlir_create_indexing_maps_attr(MlirContext ctx, int num_maps,
                                                               int rank) {
    (void)ctx;
    (void)num_maps;
    LOG_ERROR("MLIR support not available");
    MlirAttribute null_attr = {nullptr};
    return null_attr;
}

static inline MlirAttribute cml_mlir_create_transpose_maps_attr(MlirContext ctx) {
    (void)ctx;
    LOG_ERROR("MLIR support not available");
    MlirAttribute null_attr = {nullptr};
    return null_attr;
}

static inline MlirAttribute cml_mlir_create_broadcast_maps_attr(MlirContext ctx, int rank) {
    (void)ctx;
    (void)rank;
    LOG_ERROR("MLIR support not available");
    MlirAttribute null_attr = {nullptr};
    return null_attr;
}

static inline MlirAttribute cml_mlir_create_scalar_broadcast_maps_attr(MlirContext ctx, int rank) {
    (void)ctx;
    (void)rank;
    LOG_ERROR("MLIR support not available");
    MlirAttribute null_attr = {nullptr};
    return null_attr;
}

// GPU backend stubs when MLIR not available
typedef enum CMLGPUBackend {
    CML_GPU_BACKEND_CUDA = 0,
    CML_GPU_BACKEND_ROCM,
    CML_GPU_BACKEND_SPIRV,
    CML_GPU_BACKEND_METAL
} CMLGPUBackend;

static inline int cml_mlir_register_gpu_dialects(MlirDialectRegistry registry) {
    (void)registry;
    LOG_ERROR("MLIR support not available");
    return -1;
}

static inline int cml_mlir_lower_to_gpu(MlirModule module) {
    (void)module;
    LOG_ERROR("MLIR support not available");
    return -1;
}

static inline int cml_mlir_lower_gpu_to_target(MlirModule module, CMLGPUBackend backend) {
    (void)module;
    (void)backend;
    LOG_ERROR("MLIR support not available");
    return -1;
}

static inline int cml_mlir_lower_module_to_gpu(MlirModule module, CMLGPUBackend backend) {
    (void)module;
    (void)backend;
    LOG_ERROR("MLIR support not available");
    return -1;
}

static inline int cml_mlir_serialize_spirv(MlirModule module, void** output, size_t* output_size) {
    (void)module;
    (void)output;
    (void)output_size;
    LOG_ERROR("MLIR support not available");
    return -1;
}

#endif // CML_HAS_MLIR

#endif // CML_OPS_IR_MLIR_CPP_BRIDGE_H
