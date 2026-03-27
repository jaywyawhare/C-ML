#ifndef CML_BACKEND_BLAS_H
#define CML_BACKEND_BLAS_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum CMLBlasOrder { CML_BLAS_ROW_MAJOR = 101 } CMLBlasOrder;
typedef enum CMLBlasTranspose { CML_BLAS_NO_TRANS = 111, CML_BLAS_TRANS = 112 } CMLBlasTranspose;

typedef struct CMLBlasContext {
    void* lib_handle;   // Dynamic library handle
    char lib_name[256]; // Name of loaded library
    bool initialized;

    void (*cblas_sgemm)(int Order, int TransA, int TransB, int M, int N, int K, float alpha,
                        const float* A, int lda, const float* B, int ldb, float beta, float* C,
                        int ldc);

    void (*cblas_sgemv)(int Order, int TransA, int M, int N, float alpha, const float* A, int lda,
                        const float* X, int incX, float beta, float* Y, int incY);

    void (*cblas_saxpy)(int N, float alpha, const float* X, int incX, float* Y, int incY);
    void (*cblas_sscal)(int N, float alpha, float* X, int incX);
    float (*cblas_sdot)(int N, const float* X, int incX, const float* Y, int incY);
    float (*cblas_snrm2)(int N, const float* X, int incX);

    void (*cblas_dgemm)(int Order, int TransA, int TransB, int M, int N, int K, double alpha,
                        const double* A, int lda, const double* B, int ldb, double beta, double* C,
                        int ldc);

    void (*sgemm_)(const char* TransA, const char* TransB, const int* M, const int* N, const int* K,
                   const float* alpha, const float* A, const int* lda, const float* B,
                   const int* ldb, const float* beta, float* C, const int* ldc);

    bool is_mkl;
    bool is_openblas;
    bool is_blis;
} CMLBlasContext;

bool cml_blas_available(void);
CMLBlasContext* cml_blas_init(void);
void cml_blas_free(CMLBlasContext* ctx);

/* Lazy initialized singleton */
CMLBlasContext* cml_blas_get_context(void);

/* C = alpha * A @ B + beta * C */
int cml_blas_sgemm(CMLBlasContext* ctx, const float* A, const float* B, float* C, int M, int N,
                   int K, float alpha, float beta);

/* C = alpha * op(A) @ op(B) + beta * C, where op(X) = X or X^T */
int cml_blas_sgemm_ex(CMLBlasContext* ctx, const float* A, const float* B, float* C, int M, int N,
                      int K, float alpha, float beta, bool transA, bool transB);

/* y = alpha * A @ x + beta * y */
int cml_blas_sgemv(CMLBlasContext* ctx, const float* A, const float* x, float* y, int M, int N,
                   float alpha, float beta);

/* y = alpha * x + y */
int cml_blas_saxpy(CMLBlasContext* ctx, const float* x, float* y, int n, float alpha);

/* x = alpha * x */
int cml_blas_sscal(CMLBlasContext* ctx, float* x, int n, float alpha);

/* result = x . y */
float cml_blas_sdot(CMLBlasContext* ctx, const float* x, const float* y, int n);

/* result = ||x||_2 */
float cml_blas_snrm2(CMLBlasContext* ctx, const float* x, int n);

const char* cml_blas_get_library_name(CMLBlasContext* ctx);

#ifdef __cplusplus
}
#endif

#endif // CML_BACKEND_BLAS_H
