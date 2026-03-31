#include "ops/ir/gpu/amx.h"

#include <string.h>
#include <stdint.h>

#if defined(__APPLE__) && defined(__aarch64__)

bool cml_amx_available(void) {
    return false;
}

int cml_amx_matmul_f32(const float* A, const float* B, float* C,
                       int M, int N, int K, int lda, int ldb, int ldc) {
    (void)A; (void)B; (void)C;
    (void)M; (void)N; (void)K;
    (void)lda; (void)ldb; (void)ldc;
    return -1;
}

int cml_amx_matmul_f16(const void* A, const void* B, float* C,
                       int M, int N, int K, int lda, int ldb, int ldc) {
    (void)A; (void)B; (void)C;
    (void)M; (void)N; (void)K;
    (void)lda; (void)ldb; (void)ldc;
    return -1;
}

#else

bool cml_amx_available(void) {
    return false;
}

int cml_amx_matmul_f32(const float* A, const float* B, float* C,
                       int M, int N, int K, int lda, int ldb, int ldc) {
    (void)A; (void)B; (void)C;
    (void)M; (void)N; (void)K;
    (void)lda; (void)ldb; (void)ldc;
    return -1;
}

int cml_amx_matmul_f16(const void* A, const void* B, float* C,
                       int M, int N, int K, int lda, int ldb, int ldc) {
    (void)A; (void)B; (void)C;
    (void)M; (void)N; (void)K;
    (void)lda; (void)ldb; (void)ldc;
    return -1;
}

#endif
