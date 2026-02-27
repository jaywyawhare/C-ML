/**
 * @file gpu_codegen.h
 * @brief GPU kernel code generation via LLVM NVPTX/AMDGPU targets
 *
 * Builds LLVM IR for GPU kernels (replacing CPU loops with thread indexing),
 * emits PTX (CUDA) or HSACO (ROCm) via LLVM target machines, then uses
 * the existing CUDA/ROCm runtime backends for compilation and execution.
 */

#ifndef CML_GPU_CODEGEN_H
#define CML_GPU_CODEGEN_H

#include "ops/ir/ir.h"
#include "ops/ir/gpu/cuda_backend.h"
#include "ops/ir/gpu/rocm_backend.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief GPU target selection
 */
typedef enum GPUTarget {
    GPU_TARGET_CUDA,  // NVIDIA via NVPTX
    GPU_TARGET_ROCM   // AMD via AMDGPU
} GPUTarget;

/**
 * @brief GPU code generation context
 */
typedef struct CMLGPUCodegen {
    GPUTarget target;
    bool initialized;
    int kernel_count;

    char target_cpu[32];      // "sm_75" or "gfx900"
    char target_triple[64];   // "nvptx64-nvidia-cuda" or "amdgcn-amd-amdhsa"
    int default_block_size;   // 256

    // Backend pointers (one will be non-NULL)
    CMLCUDABackend* cuda;
    CMLROCmBackend* rocm;
} CMLGPUCodegen;

/**
 * @brief Create a GPU codegen context
 * @param target GPU target (CUDA or ROCm)
 * @param backend Pointer to the initialized backend (CMLCUDABackend* or CMLROCmBackend*)
 * @return New codegen context, or NULL on failure
 */
CMLGPUCodegen* cml_gpu_codegen_create(GPUTarget target, void* backend);

/**
 * @brief Destroy GPU codegen context
 * @param cg Codegen context to destroy
 */
void cml_gpu_codegen_destroy(CMLGPUCodegen* cg);

/**
 * @brief Execute an entire IR graph on GPU
 *
 * Walks the IR node list, builds GPU kernels, emits PTX/HSACO,
 * uploads data, launches kernels, downloads results.
 * Falls back to CPU for unsupported ops.
 *
 * @param cg GPU codegen context
 * @param ir IR graph to execute
 * @return 0 on success, -1 on failure
 */
int cml_gpu_execute(CMLGPUCodegen* cg, CMLGraph_t ir);

/**
 * @brief Execute IR graph up to a specific target node on GPU
 * @param cg GPU codegen context
 * @param ir IR graph
 * @param target Execute up to and including this node
 * @return 0 on success, -1 on failure
 */
int cml_gpu_execute_up_to(CMLGPUCodegen* cg, CMLGraph_t ir, struct IRNode* target);

#ifdef __cplusplus
}
#endif

#endif // CML_GPU_CODEGEN_H
