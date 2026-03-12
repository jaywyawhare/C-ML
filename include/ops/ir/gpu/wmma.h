/**
 * @file wmma.h
 * @brief Tensor Core / WMMA support for fp16 matmul
 *
 * Generate WMMA CUDA kernels via NVRTC. Detect SM >= 70.
 */

#ifndef CML_OPS_IR_GPU_WMMA_H
#define CML_OPS_IR_GPU_WMMA_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Fragment size variants ── */

typedef enum {
    WMMA_M16N16K16 = 0,
    WMMA_M32N8K16,
    WMMA_M8N32K16,
    WMMA_FRAGMENT_COUNT
} WMMAFragmentSize;

/* ── WMMA configuration ── */

typedef struct {
    WMMAFragmentSize fragment;
    int M, N, K;           /* fragment dimensions */
    int warp_m, warp_n;    /* number of warps in M and N */
    int block_m, block_n, block_k; /* tile sizes */
} WMMAConfig;

/* ── API ── */

/**
 * @brief Check if WMMA is available (SM >= 70 NVIDIA GPU)
 */
bool cml_wmma_available(void);

/**
 * @brief Select optimal WMMA configuration for given matmul dimensions
 *
 * @param M    Rows of A / rows of C
 * @param N    Cols of B / cols of C
 * @param K    Cols of A / rows of B
 * @param config Output configuration
 * @return 0 on success, -1 if WMMA not applicable
 */
int cml_wmma_select_config(int M, int N, int K, WMMAConfig* config);

/**
 * @brief Generate CUDA C source for a WMMA matmul kernel
 *
 * The returned string contains a complete CUDA kernel using
 * nvcuda::wmma intrinsics, compilable with NVRTC.
 *
 * @param config WMMA configuration
 * @param M      Rows of result
 * @param N      Cols of result
 * @param K      Inner dimension
 * @return Heap-allocated CUDA source string (caller frees), or NULL
 */
char* cml_wmma_generate_kernel(const WMMAConfig* config, int M, int N, int K);

/**
 * @brief Perform WMMA matmul: C = A * B
 *
 * A is [M, K] fp16, B is [K, N] fp16, C is [M, N] fp32.
 * Falls back to regular matmul if WMMA not available.
 *
 * @param A      Input matrix A (fp16 on device)
 * @param B      Input matrix B (fp16 on device)
 * @param C      Output matrix C (fp32 on device)
 * @param M      Rows
 * @param N      Cols
 * @param K      Inner dim
 * @return 0 on success
 */
int cml_wmma_matmul(const void* A, const void* B, void* C, int M, int N, int K);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPS_IR_GPU_WMMA_H */
