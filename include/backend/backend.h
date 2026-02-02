/**
 * @file backend.h
 * @brief SIMD/BLAS backend interface for pluggable kernels
 *
 * Provides a backend interface that allows plugging in different
 * implementations (scalar, SSE/AVX, BLAS, GPU).
 */

#ifndef CML_CORE_BACKEND_H
#define CML_CORE_BACKEND_H

#include "tensor/tensor.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Backend type
 */
typedef enum {
    BACKEND_SCALAR, // Default scalar implementation
    BACKEND_SSE,    // SSE/SSE2
    BACKEND_AVX,    // AVX/AVX2
    BACKEND_BLAS,   // BLAS library (OpenBLAS, MKL, etc.)
    BACKEND_CUDA,   // CUDA/GPU
    BACKEND_METAL,  // Metal (Apple GPU)
    BACKEND_ROCM    // ROCm (AMD GPU)
} BackendType;

/**
 * @brief Backend function pointers for operations
 */
typedef struct {
    // Matrix operations
    void (*matmul)(const void* a, const void* b, void* out, int m, int n, int k, DType dtype);
    void (*matmul_add)(const void* a, const void* b, const void* bias, void* out, int m, int n,
                       int k, DType dtype);

    // Elementwise operations
    void (*add)(const void* a, const void* b, void* out, size_t n, DType dtype);
    void (*mul)(const void* a, const void* b, void* out, size_t n, DType dtype);
    void (*relu)(const void* x, void* out, size_t n, DType dtype);
    void (*sigmoid)(const void* x, void* out, size_t n, DType dtype);

    // Reduction operations
    void (*sum)(const void* x, void* out, size_t n, DType dtype);
    void (*mean)(const void* x, void* out, size_t n, DType dtype);
} BackendOps;

/**
 * @brief Backend context
 */
typedef struct {
    BackendType type;
    BackendOps ops;
    void* context; // Backend-specific context
} Backend;

/**
 * @brief Get current backend
 */
Backend* backend_get_current(void);

/**
 * @brief Set backend
 */
int backend_set(BackendType type);

/**
 * @brief Initialize backend
 */
int backend_init(BackendType type);

/**
 * @brief Cleanup backend
 */
void backend_cleanup(void);

/**
 * @brief Check if backend is available
 */
bool backend_is_available(BackendType type);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_BACKEND_H
