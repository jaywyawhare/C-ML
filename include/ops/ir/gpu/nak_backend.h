#pragma once
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CMLNAKBackend {
    void* nak_lib;
    int gpu_arch;
    bool initialized;
} CMLNAKBackend;

bool cml_nak_available(void);
CMLNAKBackend* cml_nak_create(int gpu_arch);
void cml_nak_free(CMLNAKBackend* nak);

int cml_nak_compile(CMLNAKBackend* nak, const void* nir_shader,
                    void** binary, size_t* binary_size);

int cml_nak_compile_spirv(CMLNAKBackend* nak, const void* spirv,
                          size_t spirv_size, void** binary, size_t* binary_size);

#ifdef __cplusplus
}
#endif
