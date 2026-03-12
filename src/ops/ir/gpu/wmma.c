/**
 * @file wmma.c
 * @brief WMMA (Warp Matrix Multiply Accumulate) Tensor Core support
 *
 * Generates CUDA C kernels using nvcuda::wmma intrinsics for fp16 matrix
 * multiplication on NVIDIA Volta+ GPUs (SM >= 70).  The generated source
 * is compiled at runtime via NVRTC through the existing CUDA backend.
 *
 * Guarded by CML_HAS_CUDA -- stubs are provided otherwise.
 */

#include "ops/ir/gpu/wmma.h"
#include "core/logging.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================
 * CML_HAS_CUDA -- full implementation
 * ======================================================================== */
#ifdef CML_HAS_CUDA

#include "ops/ir/gpu/cuda_backend.h"
#include "ops/ir/dispatch.h"

/* ── Availability ── */

static CMLCUDABackend* wmma_get_cuda_backend(void) {
    CMLDispatchContext* ctx = cml_dispatch_get_global();
    if (!ctx) return NULL;
    return (CMLCUDABackend*)ctx->backend_contexts[CML_BACKEND_CUDA];
}

bool cml_wmma_available(void) {
    CMLCUDABackend* backend = wmma_get_cuda_backend();
    if (!backend || !backend->initialized) {
        return false;
    }
    /* WMMA requires SM >= 7.0 (Volta) */
    int major = backend->compute_capability_major;
    int minor = backend->compute_capability_minor;
    return (major > 7) || (major == 7 && minor >= 0);
}

/* ── Configuration selection ── */

int cml_wmma_select_config(int M, int N, int K, WMMAConfig* config) {
    if (!config) return -1;
    if (!cml_wmma_available()) return -1;

    /* Zero out the config */
    memset(config, 0, sizeof(WMMAConfig));

    /*
     * WMMA fragment sizes (for fp16 inputs, fp32 accumulator):
     *   M16N16K16   -- default, well-balanced
     *   M32N8K16    -- tall-thin output tiles
     *   M8N32K16    -- wide-short output tiles
     *
     * Heuristic: prefer M16N16K16 unless the problem shape strongly
     * favours one of the rectangular variants.
     */
    if (M >= 16 && N >= 16) {
        /* Default: 16x16x16 */
        config->fragment = WMMA_M16N16K16;
        config->M = 16;
        config->N = 16;
        config->K = 16;
    } else if (M >= 32 && N >= 8) {
        config->fragment = WMMA_M32N8K16;
        config->M = 32;
        config->N = 8;
        config->K = 16;
    } else if (M >= 8 && N >= 32) {
        config->fragment = WMMA_M8N32K16;
        config->M = 8;
        config->N = 32;
        config->K = 16;
    } else {
        /* Problem too small for WMMA -- fall back */
        LOG_DEBUG("WMMA: Problem dimensions %dx%dx%d too small", M, N, K);
        return -1;
    }

    /* Compute warp-level tiling.
     * We assign one warp per output tile of size (config->M x config->N).
     * block_m/n are the number of tiles along M/N covered by one thread
     * block; block_k is the K-dimension tile size.
     */
    config->warp_m = 1;
    config->warp_n = 1;
    config->block_m = config->M * config->warp_m;
    config->block_n = config->N * config->warp_n;
    config->block_k = config->K;

    LOG_DEBUG("WMMA config: fragment=%dx%dx%d  block=%dx%dx%d",
              config->M, config->N, config->K,
              config->block_m, config->block_n, config->block_k);

    (void)K; /* K only used for logging / future heuristics */
    return 0;
}

/* ── Kernel generation ── */

/*
 * Maximum generated source size.  The WMMA kernel template is around
 * 2-3 KB; we allocate generously.
 */
#define WMMA_SRC_MAX 8192

static void src_appendf(char** buf, size_t* cap, size_t* len,
                         const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int needed = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (needed < 0) return;

    while (*len + (size_t)needed + 1 > *cap) {
        *cap *= 2;
        char* tmp = (char*)realloc(*buf, *cap);
        if (!tmp) {
            LOG_ERROR("wmma: realloc failed");
            return;
        }
        *buf = tmp;
    }

    va_start(ap, fmt);
    vsnprintf(*buf + *len, *cap - *len, fmt, ap);
    va_end(ap);
    *len += (size_t)needed;
}

char* cml_wmma_generate_kernel(const WMMAConfig* config, int M, int N, int K) {
    if (!config) return NULL;
    if (M <= 0 || N <= 0 || K <= 0) return NULL;

    size_t cap = WMMA_SRC_MAX;
    size_t len = 0;
    char* src = (char*)malloc(cap);
    if (!src) return NULL;
    src[0] = '\0';

    int frag_m = config->M;
    int frag_n = config->N;
    int frag_k = config->K;

    /* ── Includes and namespace ── */
    src_appendf(&src, &cap, &len,
        "#include <mma.h>\n"
        "#include <cuda_fp16.h>\n"
        "\n"
        "using namespace nvcuda;\n"
        "\n");

    /* ── Kernel function ── */
    src_appendf(&src, &cap, &len,
        "extern \"C\" __global__\n"
        "void wmma_matmul(const half* __restrict__ A,\n"
        "                  const half* __restrict__ B,\n"
        "                  float*      __restrict__ C,\n"
        "                  int M_total, int N_total, int K_total) {\n"
        "\n");

    /* ── Fragment declarations ── */
    src_appendf(&src, &cap, &len,
        "    // Fragment dimensions: %dx%dx%d\n"
        "    wmma::fragment<wmma::matrix_a, %d, %d, %d, half, wmma::row_major> frag_a;\n"
        "    wmma::fragment<wmma::matrix_b, %d, %d, %d, half, wmma::row_major> frag_b;\n"
        "    wmma::fragment<wmma::accumulator, %d, %d, %d, float> frag_c;\n"
        "\n",
        frag_m, frag_n, frag_k,
        frag_m, frag_n, frag_k,
        frag_m, frag_n, frag_k,
        frag_m, frag_n, frag_k);

    /* ── Warp-level tile coordinates ── */
    src_appendf(&src, &cap, &len,
        "    // Each warp computes one %dx%d output tile\n"
        "    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;\n"
        "    int numWarpsN = (N_total + %d - 1) / %d;\n"
        "    int warpRow = (warpId / numWarpsN) * %d;\n"
        "    int warpCol = (warpId %% numWarpsN) * %d;\n"
        "\n"
        "    // Bounds check: skip warps outside the output matrix\n"
        "    if (warpRow >= M_total || warpCol >= N_total) return;\n"
        "\n",
        frag_m, frag_n,
        frag_n, frag_n,
        frag_m,
        frag_n);

    /* ── Zero the accumulator ── */
    src_appendf(&src, &cap, &len,
        "    // Zero the accumulator fragment\n"
        "    wmma::fill_fragment(frag_c, 0.0f);\n"
        "\n");

    /* ── Tiled K loop ── */
    src_appendf(&src, &cap, &len,
        "    // Tiled loop over K dimension\n"
        "    for (int k = 0; k < K_total; k += %d) {\n"
        "        // Load A tile: A[warpRow .. warpRow+%d, k .. k+%d]\n"
        "        const half* a_ptr = A + warpRow * K_total + k;\n"
        "        wmma::load_matrix_sync(frag_a, a_ptr, K_total);\n"
        "\n"
        "        // Load B tile: B[k .. k+%d, warpCol .. warpCol+%d]\n"
        "        const half* b_ptr = B + k * N_total + warpCol;\n"
        "        wmma::load_matrix_sync(frag_b, b_ptr, N_total);\n"
        "\n"
        "        // Multiply-accumulate: C += A * B\n"
        "        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);\n"
        "    }\n"
        "\n",
        frag_k,
        frag_m, frag_k,
        frag_k, frag_n);

    /* ── Store result ── */
    src_appendf(&src, &cap, &len,
        "    // Store the result tile to C[warpRow, warpCol]\n"
        "    float* c_ptr = C + warpRow * N_total + warpCol;\n"
        "    wmma::store_matrix_sync(c_ptr, frag_c, N_total, wmma::mem_row_major);\n"
        "}\n");

    LOG_DEBUG("WMMA: Generated kernel for %dx%dx%d (fragment %dx%dx%d), %zu bytes",
              M, N, K, frag_m, frag_n, frag_k, len);

    return src;
}

/* ── Matrix multiply (compile + launch) ── */

int cml_wmma_matmul(const void* A, const void* B, void* C,
                    int M, int N, int K) {
    if (!A || !B || !C) return -1;
    if (M <= 0 || N <= 0 || K <= 0) return -1;

    if (!cml_wmma_available()) {
        LOG_WARNING("WMMA not available, cannot perform Tensor Core matmul");
        return -1;
    }

    CMLCUDABackend* cuda = wmma_get_cuda_backend();
    if (!cuda || !cuda->initialized) {
        LOG_ERROR("CUDA backend not initialized");
        return -1;
    }

    /* Select configuration */
    WMMAConfig config;
    if (cml_wmma_select_config(M, N, K, &config) != 0) {
        LOG_WARNING("WMMA config selection failed for %dx%dx%d", M, N, K);
        return -1;
    }

    /* Generate the CUDA kernel source */
    char* kernel_src = cml_wmma_generate_kernel(&config, M, N, K);
    if (!kernel_src) {
        LOG_ERROR("WMMA kernel generation failed");
        return -1;
    }

    /* Compile via NVRTC through the existing CUDA backend */
    CMLCUDAKernel* kernel = cml_cuda_compile_source(cuda, kernel_src, "wmma_matmul");
    free(kernel_src);

    if (!kernel) {
        LOG_ERROR("WMMA kernel compilation failed");
        return -1;
    }

    /* Calculate launch configuration.
     * One warp (32 threads) per output tile.
     * Total warps needed = ceil(M/frag_m) * ceil(N/frag_n)
     */
    int tiles_m = (M + config.M - 1) / config.M;
    int tiles_n = (N + config.N - 1) / config.N;
    int total_warps = tiles_m * tiles_n;
    int total_threads = total_warps * 32;

    int block_size = 256; /* must be multiple of 32 */
    int grid_size = (total_threads + block_size - 1) / block_size;

    cml_cuda_kernel_set_launch_config(kernel, grid_size, 1, 1, block_size, 1, 1);

    /* Set up kernel arguments */
    int M_arg = M, N_arg = N, K_arg = K;
    void* args[] = {
        (void*)&A, (void*)&B, (void*)&C,
        (void*)&M_arg, (void*)&N_arg, (void*)&K_arg
    };

    int result = cml_cuda_launch_kernel(cuda, kernel, args, 6);
    if (result == 0)
        result = cml_cuda_synchronize(cuda);

    cml_cuda_kernel_free(cuda, kernel);

    if (result != 0)
        LOG_ERROR("WMMA matmul launch/sync failed");

    return result;
}

/* ========================================================================
 * Stubs -- when CML_HAS_CUDA is NOT defined
 * ======================================================================== */
#else /* !CML_HAS_CUDA */

bool cml_wmma_available(void) {
    return false;
}

int cml_wmma_select_config(int M, int N, int K, WMMAConfig* config) {
    (void)M; (void)N; (void)K; (void)config;
    return -1;
}

char* cml_wmma_generate_kernel(const WMMAConfig* config, int M, int N, int K) {
    (void)config; (void)M; (void)N; (void)K;
    return NULL;
}

int cml_wmma_matmul(const void* A, const void* B, void* C,
                    int M, int N, int K) {
    (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;
    return -1;
}

#endif /* CML_HAS_CUDA */
