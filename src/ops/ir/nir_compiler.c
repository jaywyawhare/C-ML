/**
 * @file nir_compiler.c
 * @brief NIR/Mesa multi-vendor GPU compilation (stub)
 */

#include "ops/ir/nir_compiler.h"
#include <stdlib.h>
#include <string.h>

static const char* nir_target_names[] = {
    "radeonsi", "iris", "turnip", "panfrost", "freedreno",
};

bool cml_nir_available(void) {
    /* Would check for Mesa NIR library */
    return false;
}

CMLNIRCompiler* cml_nir_compiler_create(CMLNIRTarget target) {
    CMLNIRCompiler* c = (CMLNIRCompiler*)calloc(1, sizeof(CMLNIRCompiler));
    if (c) c->target = target;
    return c;
}

void cml_nir_compiler_free(CMLNIRCompiler* compiler) {
    free(compiler);
}

int cml_nir_compile(CMLNIRCompiler* compiler, CMLGraph_t ir) {
    (void)ir;
    if (!compiler || !cml_nir_available()) return -1;
    return -1;
}

size_t cml_nir_binary_size(const CMLNIRCompiler* compiler) {
    (void)compiler;
    return 0;
}

const void* cml_nir_binary_data(const CMLNIRCompiler* compiler) {
    (void)compiler;
    return NULL;
}

const char* cml_nir_target_name(CMLNIRTarget target) {
    if (target >= NIR_TARGET_COUNT) return "unknown";
    return nir_target_names[target];
}
