/**
 * @file blas.c
 * @brief BLAS library integration implementation
 */

#include "backend/blas.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef __linux__
#include <dlfcn.h>
#define LIB_LOAD(path) dlopen(path, RTLD_LAZY | RTLD_LOCAL)
#define LIB_SYM(handle, name) dlsym(handle, name)
#define LIB_CLOSE(handle) dlclose(handle)
#elif defined(__APPLE__)
#include <dlfcn.h>
#define LIB_LOAD(path) dlopen(path, RTLD_LAZY | RTLD_LOCAL)
#define LIB_SYM(handle, name) dlsym(handle, name)
#define LIB_CLOSE(handle) dlclose(handle)
#elif defined(_WIN32)
#include <windows.h>
#define LIB_LOAD(path) LoadLibraryA(path)
#define LIB_SYM(handle, name) GetProcAddress((HMODULE)handle, name)
#define LIB_CLOSE(handle) FreeLibrary((HMODULE)handle)
#else
#define LIB_LOAD(path) NULL
#define LIB_SYM(handle, name) NULL
#define LIB_CLOSE(handle) ((void)0)
#endif

static CMLBlasContext* g_blas_ctx = NULL;

// List of BLAS libraries to try loading
// Order: MKL (fastest) -> OpenBLAS -> ATLAS -> Reference
// Set CML_BLAS_LIB environment variable to override (e.g., "libcblas.so.3")
static const char* blas_library_paths[] = {
#ifdef __linux__
    "libmkl_rt.so",
    "libopenblas.so.0",
    "libopenblas.so",
    "libatlas.so",
    "libcblas.so.3",
    "libcblas.so",
    "libblas.so.3",
    "libblas.so",
#elif defined(__APPLE__)
    "/System/Library/Frameworks/Accelerate.framework/Accelerate",
    "libopenblas.dylib",
    "/opt/homebrew/opt/openblas/lib/libopenblas.dylib",
    "/usr/local/opt/openblas/lib/libopenblas.dylib",
    "libblas.dylib",
#elif defined(_WIN32)
    "openblas.dll", "libopenblas.dll", "mkl_rt.dll", "blas.dll",
#endif
    NULL};

bool cml_blas_available(void) {
    // Try to load any BLAS library
    for (int i = 0; blas_library_paths[i] != NULL; i++) {
        void* lib = LIB_LOAD(blas_library_paths[i]);
        if (lib) {
            // Check for CBLAS or Fortran BLAS symbols
            void* sgemm_cblas = LIB_SYM(lib, "cblas_sgemm");
            void* sgemm_f     = LIB_SYM(lib, "sgemm_");
            LIB_CLOSE(lib);

            if (sgemm_cblas || sgemm_f) {
                return true;
            }
        }
    }
    return false;
}

static void load_cblas_functions(CMLBlasContext* ctx) {
    if (!ctx || !ctx->lib_handle)
        return;

    // Try CBLAS-style functions first
    ctx->cblas_sgemm = LIB_SYM(ctx->lib_handle, "cblas_sgemm");
    ctx->cblas_sgemv = LIB_SYM(ctx->lib_handle, "cblas_sgemv");
    ctx->cblas_saxpy = LIB_SYM(ctx->lib_handle, "cblas_saxpy");
    ctx->cblas_sscal = LIB_SYM(ctx->lib_handle, "cblas_sscal");
    ctx->cblas_sdot  = LIB_SYM(ctx->lib_handle, "cblas_sdot");
    ctx->cblas_snrm2 = LIB_SYM(ctx->lib_handle, "cblas_snrm2");
    ctx->cblas_dgemm = LIB_SYM(ctx->lib_handle, "cblas_dgemm");

    // Try Fortran-style if CBLAS not available
    if (!ctx->cblas_sgemm) {
        ctx->sgemm_ = LIB_SYM(ctx->lib_handle, "sgemm_");
    }
}

CMLBlasContext* cml_blas_init(void) {
    CMLBlasContext* ctx = calloc(1, sizeof(CMLBlasContext));
    if (!ctx) {
        LOG_ERROR("Failed to allocate BLAS context");
        return NULL;
    }

    // Check for user-specified BLAS library override
    const char* env_blas = getenv("CML_BLAS_LIB");
    if (env_blas && env_blas[0] != '\0') {
        ctx->lib_handle = LIB_LOAD(env_blas);
        if (ctx->lib_handle) {
            load_cblas_functions(ctx);
            if (ctx->cblas_sgemm || ctx->sgemm_) {
                strncpy(ctx->lib_name, env_blas, sizeof(ctx->lib_name) - 1);
                ctx->initialized = true;
                LOG_INFO("Loaded BLAS library (from CML_BLAS_LIB): %s", ctx->lib_name);
                return ctx;
            }
            LIB_CLOSE(ctx->lib_handle);
            ctx->lib_handle = NULL;
        }
        LOG_WARNING("Failed to load BLAS from CML_BLAS_LIB=%s, trying defaults", env_blas);
    }

    // Try to load BLAS libraries in order
    for (int i = 0; blas_library_paths[i] != NULL; i++) {
        ctx->lib_handle = LIB_LOAD(blas_library_paths[i]);
        if (ctx->lib_handle) {
            load_cblas_functions(ctx);

            // Verify we have at least sgemm
            if (ctx->cblas_sgemm || ctx->sgemm_) {
                strncpy(ctx->lib_name, blas_library_paths[i], sizeof(ctx->lib_name) - 1);
                ctx->initialized = true;
                LOG_INFO("Loaded BLAS library: %s", ctx->lib_name);
                return ctx;
            }

            LIB_CLOSE(ctx->lib_handle);
            ctx->lib_handle = NULL;
        }
    }

    LOG_WARNING("No BLAS library found");
    free(ctx);
    return NULL;
}

void cml_blas_free(CMLBlasContext* ctx) {
    if (!ctx)
        return;

    if (ctx->lib_handle) {
        LIB_CLOSE(ctx->lib_handle);
    }

    if (ctx == g_blas_ctx) {
        g_blas_ctx = NULL;
    }

    free(ctx);
}

CMLBlasContext* cml_blas_get_context(void) {
    if (!g_blas_ctx) {
        g_blas_ctx = cml_blas_init();
    }
    return g_blas_ctx;
}

int cml_blas_sgemm(CMLBlasContext* ctx, const float* A, const float* B, float* C, int M, int N,
                   int K, float alpha, float beta) {
    if (!ctx)
        ctx = cml_blas_get_context();
    if (!ctx || !ctx->initialized) {
        LOG_ERROR("BLAS not available");
        return -1;
    }

    if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0) {
        LOG_ERROR("Invalid arguments to cml_blas_sgemm");
        return -1;
    }

    if (ctx->cblas_sgemm) {
        // CBLAS interface (row-major)
        // C = alpha * A @ B + beta * C
        // A is M x K, B is K x N, C is M x N
        ctx->cblas_sgemm(CML_BLAS_ROW_MAJOR, CML_BLAS_NO_TRANS, CML_BLAS_NO_TRANS, M, N, K, alpha,
                         A, K,      // A is M x K, leading dimension is K
                         B, N,      // B is K x N, leading dimension is N
                         beta, C, N // C is M x N, leading dimension is N
        );
        return 0;
    } else if (ctx->sgemm_) {
        // Fortran BLAS interface (column-major)
        // Need to transpose: C^T = B^T @ A^T
        // This effectively computes row-major result
        char transA = 'N';
        char transB = 'N';
        ctx->sgemm_(&transA, &transB, &N, &M, &K, &alpha, B, &N, A, &K, &beta, C, &N);
        return 0;
    }

    LOG_ERROR("No sgemm function available");
    return -1;
}

int cml_blas_sgemv(CMLBlasContext* ctx, const float* A, const float* x, float* y, int M, int N,
                   float alpha, float beta) {
    if (!ctx)
        ctx = cml_blas_get_context();
    if (!ctx || !ctx->initialized)
        return -1;

    if (!A || !x || !y || M <= 0 || N <= 0)
        return -1;

    if (ctx->cblas_sgemv) {
        ctx->cblas_sgemv(CML_BLAS_ROW_MAJOR, CML_BLAS_NO_TRANS, M, N, alpha, A, N, x, 1, beta, y,
                         1);
        return 0;
    }

    // Fallback: manual implementation
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        y[i] = alpha * sum + beta * y[i];
    }
    return 0;
}

int cml_blas_saxpy(CMLBlasContext* ctx, const float* x, float* y, int n, float alpha) {
    if (!ctx)
        ctx = cml_blas_get_context();
    if (!ctx || !ctx->initialized)
        return -1;

    if (!x || !y || n <= 0)
        return -1;

    if (ctx->cblas_saxpy) {
        ctx->cblas_saxpy(n, alpha, x, 1, y, 1);
        return 0;
    }

    // Fallback
    for (int i = 0; i < n; i++) {
        y[i] += alpha * x[i];
    }
    return 0;
}

int cml_blas_sscal(CMLBlasContext* ctx, float* x, int n, float alpha) {
    if (!ctx)
        ctx = cml_blas_get_context();
    if (!ctx || !ctx->initialized)
        return -1;

    if (!x || n <= 0)
        return -1;

    if (ctx->cblas_sscal) {
        ctx->cblas_sscal(n, alpha, x, 1);
        return 0;
    }

    // Fallback
    for (int i = 0; i < n; i++) {
        x[i] *= alpha;
    }
    return 0;
}

float cml_blas_sdot(CMLBlasContext* ctx, const float* x, const float* y, int n) {
    if (!ctx)
        ctx = cml_blas_get_context();
    if (!ctx || !ctx->initialized || !x || !y || n <= 0) {
        // Fallback
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += x[i] * y[i];
        }
        return sum;
    }

    if (ctx->cblas_sdot) {
        return ctx->cblas_sdot(n, x, 1, y, 1);
    }

    // Fallback
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}

float cml_blas_snrm2(CMLBlasContext* ctx, const float* x, int n) {
    if (!ctx)
        ctx = cml_blas_get_context();
    if (!ctx || !ctx->initialized || !x || n <= 0) {
        // Fallback
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += x[i] * x[i];
        }
        return sqrtf(sum);
    }

    if (ctx->cblas_snrm2) {
        return ctx->cblas_snrm2(n, x, 1);
    }

    // Fallback
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += x[i] * x[i];
    }
    return sqrtf(sum);
}

const char* cml_blas_get_library_name(CMLBlasContext* ctx) {
    if (!ctx || !ctx->initialized) {
        return "None";
    }
    return ctx->lib_name;
}
