/**
 * @file nir_compiler.h
 * @brief NIR/Mesa multi-vendor GPU compilation
 *
 * Compiles IR to NIR (Mesa's intermediate representation) for multi-vendor
 * GPU support through Mesa's driver stack (AMD, Intel, Qualcomm, etc.).
 *
 * NIR is Mesa's common compiler IR used by all Gallium3D and Vulkan drivers.
 * This module dynamically loads libmesa_nir.so at runtime so no build-time
 * dependency on Mesa internals is required.  When Mesa is not available every
 * public function gracefully returns an error / false.
 *
 * Compilation pipeline:
 *   CML UOps -> NIR ALU/intrinsic ops -> NIR optimisation passes -> SPIR-V
 *   The SPIR-V output can then be fed into the Vulkan backend for execution.
 */

#ifndef CML_NIR_COMPILER_H
#define CML_NIR_COMPILER_H

#include "ops/ir/ir.h"
#include "ops/uops.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    NIR_TARGET_RADEONSI = 0, /* AMD RDNA/GCN via radeonsi */
    NIR_TARGET_IRIS,          /* Intel Xe via iris */
    NIR_TARGET_TURNIP,        /* Qualcomm Adreno via turnip */
    NIR_TARGET_PANFROST,      /* ARM Mali via panfrost */
    NIR_TARGET_FREEDRENO,     /* Qualcomm via freedreno */
    NIR_TARGET_NVK,           /* NVIDIA via NVK (Nouveau Vulkan) */
    NIR_TARGET_RADV,          /* AMD via RADV (Vulkan) */
    NIR_TARGET_LLVMPIPE,      /* Software rasteriser (llvmpipe) */
    NIR_TARGET_COUNT,
} CMLNIRTarget;

typedef struct CMLNIRCompiler {
    bool initialized;
    CMLNIRTarget target;
    void* mesa_lib;           /* dlopen("libmesa_nir.so") handle */
    void* nir_shader;         /* nir_shader* handle */
    void* compiler_options;
    int version;

    /* SPIR-V output (NIR -> SPIR-V for Vulkan pipeline) */
    uint32_t* spirv_output;
    size_t spirv_size;

    /* NIR builder function pointers */
    void* (*nir_builder_init_simple_shader)(void* mem_ctx, void* options,
                                            int stage, const char* name);
    void* (*nir_fadd)(void* builder, void* a, void* b);
    void* (*nir_fmul)(void* builder, void* a, void* b);
    void* (*nir_fexp2)(void* builder, void* a);
    void* (*nir_load_ssbo)(void* builder, int components, int bit_size,
                           void* index, void* offset);
    void  (*nir_store_ssbo)(void* builder, void* value, void* index,
                            void* offset);
    void* (*nir_load_global_invocation_id)(void* builder);
    void* (*nir_shader_to_spirv)(void* shader, size_t* size);
} CMLNIRCompiler;

/** Check if NIR compilation is available (requires Mesa) */
bool cml_nir_available(void);

/** Create NIR compiler for target */
CMLNIRCompiler* cml_nir_compiler_create(CMLNIRTarget target);

/** Free NIR compiler */
void cml_nir_compiler_free(CMLNIRCompiler* compiler);

/** Compile IR graph to NIR */
int cml_nir_compile(CMLNIRCompiler* compiler, CMLGraph_t ir);

/** Get compiled binary size (SPIR-V output) */
size_t cml_nir_binary_size(const CMLNIRCompiler* compiler);

/** Get compiled binary data (SPIR-V output) */
const void* cml_nir_binary_data(const CMLNIRCompiler* compiler);

/** Get target name */
const char* cml_nir_target_name(CMLNIRTarget target);

/** Map UOp to NIR operation */
int cml_nir_emit_uop(CMLNIRCompiler* compiler, UOpType op, int num_inputs);

#ifdef __cplusplus
}
#endif

#endif /* CML_NIR_COMPILER_H */
