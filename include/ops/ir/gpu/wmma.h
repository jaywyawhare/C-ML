/*
 * Tensor Core / WMMA support for fp16 matmul.
 * Generate WMMA CUDA kernels via NVRTC. Requires SM >= 70.
 */

#ifndef CML_OPS_IR_GPU_WMMA_H
#define CML_OPS_IR_GPU_WMMA_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    WMMA_M16N16K16 = 0,
    WMMA_M32N8K16,
    WMMA_M8N32K16,
    WMMA_FRAGMENT_COUNT
} WMMAFragmentSize;

typedef struct {
    WMMAFragmentSize fragment;
    int M, N, K;           /* fragment dimensions */
    int warp_m, warp_n;    /* number of warps in M and N */
    int block_m, block_n, block_k; /* tile sizes */
} WMMAConfig;

bool cml_wmma_available(void);
int cml_wmma_select_config(int M, int N, int K, WMMAConfig* config);

/* Returns a complete CUDA kernel using nvcuda::wmma intrinsics,
   compilable with NVRTC. Caller frees. */
char* cml_wmma_generate_kernel(const WMMAConfig* config, int M, int N, int K);

/* C = A * B. A is [M,K] fp16, B is [K,N] fp16, C is [M,N] fp32.
   Falls back to regular matmul if WMMA not available. */
int cml_wmma_matmul(const void* A, const void* B, void* C, int M, int N, int K);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPS_IR_GPU_WMMA_H */
