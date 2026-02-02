/**
 * @file blas.h
 * @brief BLAS library integration for optimized linear algebra operations
 *
 * This module provides dynamic loading of BLAS libraries (OpenBLAS, MKL,
 * Accelerate, etc.) and exposes optimized matrix operations.
 */

#ifndef CML_BACKEND_BLAS_H
#define CML_BACKEND_BLAS_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// CBLAS row/column major order
typedef enum CMLBlasOrder { CML_BLAS_ROW_MAJOR = 101, CML_BLAS_COL_MAJOR = 102 } CMLBlasOrder;

// CBLAS transpose options
typedef enum CMLBlasTranspose {
    CML_BLAS_NO_TRANS   = 111,
    CML_BLAS_TRANS      = 112,
    CML_BLAS_CONJ_TRANS = 113
} CMLBlasTranspose;

/**
 * @brief BLAS context for managing library state
 */
typedef struct CMLBlasContext {
    void* lib_handle;   // Dynamic library handle
    char lib_name[256]; // Name of loaded library
    bool initialized;

    // CBLAS function pointers (single precision)
    void (*cblas_sgemm)(int Order, int TransA, int TransB, int M, int N, int K, float alpha,
                        const float* A, int lda, const float* B, int ldb, float beta, float* C,
                        int ldc);

    void (*cblas_sgemv)(int Order, int TransA, int M, int N, float alpha, const float* A, int lda,
                        const float* X, int incX, float beta, float* Y, int incY);

    void (*cblas_saxpy)(int N, float alpha, const float* X, int incX, float* Y, int incY);
    void (*cblas_sscal)(int N, float alpha, float* X, int incX);
    float (*cblas_sdot)(int N, const float* X, int incX, const float* Y, int incY);
    float (*cblas_snrm2)(int N, const float* X, int incX);

    // CBLAS function pointers (double precision)
    void (*cblas_dgemm)(int Order, int TransA, int TransB, int M, int N, int K, double alpha,
                        const double* A, int lda, const double* B, int ldb, double beta, double* C,
                        int ldc);

    // Fortran-style BLAS (if CBLAS not available)
    void (*sgemm_)(const char* TransA, const char* TransB, const int* M, const int* N, const int* K,
                   const float* alpha, const float* A, const int* lda, const float* B,
                   const int* ldb, const float* beta, float* C, const int* ldc);
} CMLBlasContext;

// ============================================================================
// Initialization
// ============================================================================

/**
 * @brief Check if BLAS is available on the system
 * @return true if any BLAS library is found
 */
bool cml_blas_available(void);

/**
 * @brief Initialize BLAS context (loads library dynamically)
 * @return BLAS context or NULL if not available
 */
CMLBlasContext* cml_blas_init(void);

/**
 * @brief Free BLAS context
 * @param ctx BLAS context to free
 */
void cml_blas_free(CMLBlasContext* ctx);

/**
 * @brief Get global BLAS context (lazy initialized singleton)
 * @return Global BLAS context or NULL if not available
 */
CMLBlasContext* cml_blas_get_context(void);

// ============================================================================
// Matrix Operations (High-Level API)
// ============================================================================

/**
 * @brief Matrix multiplication: C = alpha * A @ B + beta * C
 *
 * @param ctx BLAS context (NULL uses global)
 * @param A Input matrix A [M x K]
 * @param B Input matrix B [K x N]
 * @param C Output matrix C [M x N]
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param alpha Scalar multiplier for A @ B
 * @param beta Scalar multiplier for C (0 to overwrite)
 * @return 0 on success, -1 on failure
 */
int cml_blas_sgemm(CMLBlasContext* ctx, const float* A, const float* B, float* C, int M, int N,
                   int K, float alpha, float beta);

/**
 * @brief Matrix-vector multiplication: y = alpha * A @ x + beta * y
 *
 * @param ctx BLAS context (NULL uses global)
 * @param A Input matrix A [M x N]
 * @param x Input vector x [N]
 * @param y Output vector y [M]
 * @param M Number of rows in A
 * @param N Number of columns in A
 * @param alpha Scalar multiplier
 * @param beta Scalar multiplier for y
 * @return 0 on success, -1 on failure
 */
int cml_blas_sgemv(CMLBlasContext* ctx, const float* A, const float* x, float* y, int M, int N,
                   float alpha, float beta);

/**
 * @brief Vector addition: y = alpha * x + y
 *
 * @param ctx BLAS context (NULL uses global)
 * @param x Input vector
 * @param y Output vector (modified in place)
 * @param n Vector length
 * @param alpha Scalar multiplier
 * @return 0 on success, -1 on failure
 */
int cml_blas_saxpy(CMLBlasContext* ctx, const float* x, float* y, int n, float alpha);

/**
 * @brief Vector scaling: x = alpha * x
 *
 * @param ctx BLAS context (NULL uses global)
 * @param x Vector to scale (modified in place)
 * @param n Vector length
 * @param alpha Scale factor
 * @return 0 on success, -1 on failure
 */
int cml_blas_sscal(CMLBlasContext* ctx, float* x, int n, float alpha);

/**
 * @brief Dot product: result = x . y
 *
 * @param ctx BLAS context (NULL uses global)
 * @param x Input vector x
 * @param y Input vector y
 * @param n Vector length
 * @return Dot product result
 */
float cml_blas_sdot(CMLBlasContext* ctx, const float* x, const float* y, int n);

/**
 * @brief Vector 2-norm: result = ||x||_2
 *
 * @param ctx BLAS context (NULL uses global)
 * @param x Input vector
 * @param n Vector length
 * @return 2-norm of vector
 */
float cml_blas_snrm2(CMLBlasContext* ctx, const float* x, int n);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Get name of loaded BLAS library
 * @param ctx BLAS context
 * @return Library name string
 */
const char* cml_blas_get_library_name(CMLBlasContext* ctx);

/**
 * @brief Print BLAS status
 * @param ctx BLAS context
 */
void cml_blas_print_status(CMLBlasContext* ctx);

#ifdef __cplusplus
}
#endif

#endif // CML_BACKEND_BLAS_H
