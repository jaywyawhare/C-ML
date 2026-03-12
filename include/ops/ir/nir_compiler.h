/**
 * @file nir_compiler.h
 * @brief NIR/Mesa multi-vendor GPU compilation
 *
 * Compiles IR to NIR (Mesa's intermediate representation) for multi-vendor
 * GPU support through Mesa's driver stack (AMD, Intel, Qualcomm, etc.).
 */

#ifndef CML_NIR_COMPILER_H
#define CML_NIR_COMPILER_H

#include "ops/ir/ir.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    NIR_TARGET_RADEONSI = 0, /* AMD RDNA/GCN via radeonsi */
    NIR_TARGET_IRIS,          /* Intel Xe via iris */
    NIR_TARGET_TURNIP,        /* Qualcomm Adreno via turnip */
    NIR_TARGET_PANFROST,      /* ARM Mali via panfrost */
    NIR_TARGET_FREEDRENO,     /* Qualcomm via freedreno */
    NIR_TARGET_COUNT,
} CMLNIRTarget;

typedef struct CMLNIRCompiler {
    bool initialized;
    CMLNIRTarget target;
    void* nir_shader;         /* nir_shader* handle */
    void* compiler_options;
    int version;              /* NIR version */
} CMLNIRCompiler;

/** Check if NIR compilation is available (requires Mesa) */
bool cml_nir_available(void);

/** Create NIR compiler for target */
CMLNIRCompiler* cml_nir_compiler_create(CMLNIRTarget target);

/** Free NIR compiler */
void cml_nir_compiler_free(CMLNIRCompiler* compiler);

/** Compile IR graph to NIR */
int cml_nir_compile(CMLNIRCompiler* compiler, CMLGraph_t ir);

/** Get compiled binary size */
size_t cml_nir_binary_size(const CMLNIRCompiler* compiler);

/** Get compiled binary data */
const void* cml_nir_binary_data(const CMLNIRCompiler* compiler);

/** Get target name */
const char* cml_nir_target_name(CMLNIRTarget target);

#ifdef __cplusplus
}
#endif

#endif /* CML_NIR_COMPILER_H */
