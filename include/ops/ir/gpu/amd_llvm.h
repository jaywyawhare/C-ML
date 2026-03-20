#pragma once
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CMLAMDLLVMCompiler {
    void* llvm_ctx;
    char target_triple[64];
    char target_cpu[32];
    char features[128];
    int opt_level;
} CMLAMDLLVMCompiler;

CMLAMDLLVMCompiler* cml_amd_llvm_create(const char* gpu_arch);
void cml_amd_llvm_free(CMLAMDLLVMCompiler* comp);

int cml_amd_llvm_compile_ir(CMLAMDLLVMCompiler* comp,
                            const char* ir_source,
                            void** code_object, size_t* code_size);

int cml_amd_llvm_compile_source(CMLAMDLLVMCompiler* comp,
                                const char* source,
                                void** code_object, size_t* code_size);

bool cml_amd_llvm_available(void);

#ifdef __cplusplus
}
#endif
