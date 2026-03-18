#ifndef CML_CORE_BACKEND_H
#define CML_CORE_BACKEND_H

#include "tensor/tensor.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    BACKEND_SCALAR, // Default scalar implementation
    BACKEND_SSE,    // SSE/SSE2
    BACKEND_AVX,    // AVX/AVX2
    BACKEND_BLAS,   // BLAS library (OpenBLAS, MKL, etc.)
    BACKEND_CUDA,   // CUDA/GPU
    BACKEND_METAL,  // Metal (Apple GPU)
    BACKEND_ROCM,   // ROCm (AMD GPU)
    BACKEND_OPENCL  // OpenCL (cross-vendor GPU)
} BackendType;

typedef struct {
    void (*matmul)(const void* a, const void* b, void* out, int m, int n, int k, DType dtype);
    void (*matmul_add)(const void* a, const void* b, const void* bias, void* out, int m, int n,
                       int k, DType dtype);

    void (*add)(const void* a, const void* b, void* out, size_t n, DType dtype);
    void (*mul)(const void* a, const void* b, void* out, size_t n, DType dtype);
    void (*relu)(const void* x, void* out, size_t n, DType dtype);
    void (*sigmoid)(const void* x, void* out, size_t n, DType dtype);

    void (*sum)(const void* x, void* out, size_t n, DType dtype);
    void (*mean)(const void* x, void* out, size_t n, DType dtype);
} BackendOps;

typedef struct {
    BackendType type;
    BackendOps ops;
    void* context; // Backend-specific context
} Backend;

Backend* backend_get_current(void);
int backend_set(BackendType type);
int backend_init(BackendType type);
void backend_cleanup(void);
bool backend_is_available(BackendType type);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_BACKEND_H
