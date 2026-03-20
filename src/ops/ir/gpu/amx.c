#include "ops/ir/gpu/amx.h"

#include <string.h>
#include <stdint.h>

#if defined(__APPLE__) && defined(__aarch64__)

#define AMX_SET()   __asm__ volatile(".word (0x00201000 + (17 << 5))" ::: "memory")
#define AMX_CLR()   __asm__ volatile(".word (0x00201000 + (17 << 5) + 1)" ::: "memory")

#define AMX_LDX(ptr, reg) \
    __asm__ volatile(".word (0x00205000 + (%0 & 0x1F) + ((" #reg " & 0x7) << 10))" \
        :: "r"((uint64_t)(ptr)) : "memory")

#define AMX_LDY(ptr, reg) \
    __asm__ volatile(".word (0x00205100 + (%0 & 0x1F) + ((" #reg " & 0x7) << 10))" \
        :: "r"((uint64_t)(ptr)) : "memory")

#define AMX_STZ(ptr, reg) \
    __asm__ volatile(".word (0x00205200 + (%0 & 0x1F) + ((" #reg " & 0x7) << 10))" \
        :: "r"((uint64_t)(ptr)) : "memory")

#define AMX_FMA32(operand) \
    __asm__ volatile(".word (0x00205C00 + (%0 & 0x1F))" \
        :: "r"((uint64_t)(operand)) : "memory")

#define AMX_TILE_DIM 32

bool cml_amx_available(void) {
    return true;
}

int cml_amx_matmul_f32(const float* A, const float* B, float* C,
                       int M, int N, int K, int lda, int ldb, int ldc) {
    if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0)
        return -1;

    float tile_a[AMX_TILE_DIM * AMX_TILE_DIM] __attribute__((aligned(128)));
    float tile_b[AMX_TILE_DIM * AMX_TILE_DIM] __attribute__((aligned(128)));
    float tile_c[AMX_TILE_DIM * AMX_TILE_DIM] __attribute__((aligned(128)));

    AMX_SET();

    for (int m0 = 0; m0 < M; m0 += AMX_TILE_DIM) {
        int tile_m = (m0 + AMX_TILE_DIM <= M) ? AMX_TILE_DIM : M - m0;
        for (int n0 = 0; n0 < N; n0 += AMX_TILE_DIM) {
            int tile_n = (n0 + AMX_TILE_DIM <= N) ? AMX_TILE_DIM : N - n0;

            memset(tile_c, 0, sizeof(tile_c));

            for (int k0 = 0; k0 < K; k0 += AMX_TILE_DIM) {
                int tile_k = (k0 + AMX_TILE_DIM <= K) ? AMX_TILE_DIM : K - k0;

                memset(tile_a, 0, sizeof(tile_a));
                for (int i = 0; i < tile_m; i++)
                    for (int j = 0; j < tile_k; j++)
                        tile_a[i * AMX_TILE_DIM + j] = A[(m0 + i) * lda + k0 + j];

                memset(tile_b, 0, sizeof(tile_b));
                for (int i = 0; i < tile_k; i++)
                    for (int j = 0; j < tile_n; j++)
                        tile_b[i * AMX_TILE_DIM + j] = B[(k0 + i) * ldb + n0 + j];

                for (int i = 0; i < AMX_TILE_DIM; i++) {
                    AMX_LDX(&tile_a[i * AMX_TILE_DIM], 0);
                    for (int j = 0; j < AMX_TILE_DIM; j++) {
                        AMX_LDY(&tile_b[j * AMX_TILE_DIM], 0);
                        uint64_t operand = ((uint64_t)i << 10) | ((uint64_t)j << 20);
                        AMX_FMA32(operand);
                    }
                }
            }

            for (int i = 0; i < AMX_TILE_DIM; i++)
                AMX_STZ(&tile_c[i * AMX_TILE_DIM], 0);

            for (int i = 0; i < tile_m; i++)
                for (int j = 0; j < tile_n; j++)
                    C[(m0 + i) * ldc + n0 + j] += tile_c[i * AMX_TILE_DIM + j];
        }
    }

    AMX_CLR();
    return 0;
}

int cml_amx_matmul_f16(const void* A, const void* B, float* C,
                       int M, int N, int K, int lda, int ldb, int ldc) {
    (void)A; (void)B; (void)C;
    (void)M; (void)N; (void)K;
    (void)lda; (void)ldb; (void)ldc;
    return -1;
}

#else /* !Apple Silicon */

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
