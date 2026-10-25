#include "ops/ir/gpu/amx.h"

#include <string.h>
#include <stdint.h>

/* No AMX path is wired up on any target yet: both Apple AMX and the x86
 * AMX-TILE tiles fall back to the generic matmul. */
bool cml_amx_available(void) { return false; }

/**
 * Stub f32 AMX matmul: always fails (-1) so callers use the generic path.
 * @param A (const float*)
 * @param B (const float*)
 * @param C (float*)
 * @param M (int)
 * @param N (int)
 * @param K (int)
 * @param lda (int)
 * @param ldb (int)
 * @param ldc (int)
 * @return (int)
 */
int cml_amx_matmul_f32(const float* A, const float* B, float* C, int M, int N, int K, int lda,
                       int ldb, int ldc) {
    (void)A;
    (void)B;
    (void)C;
    (void)M;
    (void)N;
    (void)K;
    (void)lda;
    (void)ldb;
    (void)ldc;
    return -1;
}

/**
 * Stub f16 AMX matmul: always fails (-1) so callers use the generic path.
 * @param A (const void*)
 * @param B (const void*)
 * @param C (float*)
 * @param M (int)
 * @param N (int)
 * @param K (int)
 * @param lda (int)
 * @param ldb (int)
 * @param ldc (int)
 * @return (int)
 */
int cml_amx_matmul_f16(const void* A, const void* B, float* C, int M, int N, int K, int lda,
                       int ldb, int ldc) {
    (void)A;
    (void)B;
    (void)C;
    (void)M;
    (void)N;
    (void)K;
    (void)lda;
    (void)ldb;
    (void)ldc;
    return -1;
}
