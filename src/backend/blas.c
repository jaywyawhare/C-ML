#include "backend/blas.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdatomic.h>
#include <pthread.h>
#ifdef __linux__
#include <unistd.h>
#endif

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

static pthread_mutex_t g_blas_lock = PTHREAD_MUTEX_INITIALIZER;

static inline void blas_lock(void) {
    pthread_mutex_lock(&g_blas_lock);
}
static inline void blas_unlock(void) {
    pthread_mutex_unlock(&g_blas_lock);
}

static CMLBlasContext* g_blas_ctx = NULL;

static const char* blas_library_paths[] = {
#ifdef __linux__
    /* MKL (fastest on Intel, competitive on AMD) */
    "libmkl_rt.so",
    "libmkl_rt.so.2",
    "libmkl_rt.so.1",
    /* BLIS (often beats OpenBLAS, especially on AMD) */
    "libblis.so",
    "libblis.so.4",
    "libblis.so.3",
    /* OpenBLAS (widely available, good performance) */
    "libopenblas.so.0",
    "libopenblas.so",
    /* ATLAS, reference BLAS */
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
    "libblis.dylib",
    "libblas.dylib",
#elif defined(_WIN32)
    "mkl_rt.dll", "mkl_rt.2.dll",
    "libblis.dll",
    "openblas.dll", "libopenblas.dll",
    "blas.dll",
#endif
    NULL};

bool cml_blas_available(void) {
    for (int i = 0; blas_library_paths[i] != NULL; i++) {
        void* lib = LIB_LOAD(blas_library_paths[i]);
        if (lib) {
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

    ctx->cblas_sgemm = LIB_SYM(ctx->lib_handle, "cblas_sgemm");
    ctx->cblas_sgemv = LIB_SYM(ctx->lib_handle, "cblas_sgemv");
    ctx->cblas_saxpy = LIB_SYM(ctx->lib_handle, "cblas_saxpy");
    ctx->cblas_sscal = LIB_SYM(ctx->lib_handle, "cblas_sscal");
    ctx->cblas_sdot  = LIB_SYM(ctx->lib_handle, "cblas_sdot");
    ctx->cblas_snrm2 = LIB_SYM(ctx->lib_handle, "cblas_snrm2");
    ctx->cblas_dgemm = LIB_SYM(ctx->lib_handle, "cblas_dgemm");

    if (!ctx->cblas_sgemm) {
        ctx->sgemm_ = LIB_SYM(ctx->lib_handle, "sgemm_");
    }
}

static void tune_blas_threading(CMLBlasContext* ctx) {
    if (!ctx || !ctx->lib_handle)
        return;

    /* MKL: set threading via mkl_set_num_threads */
    void (*mkl_set_num_threads)(int) = LIB_SYM(ctx->lib_handle, "MKL_Set_Num_Threads");
    if (!mkl_set_num_threads)
        mkl_set_num_threads = LIB_SYM(ctx->lib_handle, "mkl_set_num_threads");
    if (mkl_set_num_threads) {
        const char* env = getenv("MKL_NUM_THREADS");
        if (!env) {
            /* Use all available cores for MKL */
            int ncores = 1;
#ifdef __linux__
            ncores = sysconf(_SC_NPROCESSORS_ONLN);
            if (ncores < 1) ncores = 1;
#endif
            mkl_set_num_threads(ncores);
            LOG_INFO("MKL threads set to %d", ncores);
        }
        ctx->is_mkl = true;
        return;
    }

    /* OpenBLAS: set threading via openblas_set_num_threads.
       OpenBLAS built with OpenMP ignores this and reads OMP_NUM_THREADS,
       so we must also set that env var to propagate the thread count. */
    void (*openblas_set)(int) = LIB_SYM(ctx->lib_handle, "openblas_set_num_threads");
    if (openblas_set) {
        if (!getenv("OPENBLAS_NUM_THREADS") && !getenv("OMP_NUM_THREADS")) {
            int ncores = 1;
#ifdef __linux__
            ncores = sysconf(_SC_NPROCESSORS_ONLN);
            if (ncores < 1) ncores = 1;
#endif
            openblas_set(ncores);
            char ncores_str[16];
            snprintf(ncores_str, sizeof(ncores_str), "%d", ncores);
            setenv("OMP_NUM_THREADS", ncores_str, 0);
            setenv("GOTO_NUM_THREADS", ncores_str, 0);
            LOG_INFO("OpenBLAS threads set to %d", ncores);
        }
        ctx->is_openblas = true;
        return;
    }

    /* BLIS: set threading via bli_thread_set_num_threads */
    void (*blis_set)(int) = LIB_SYM(ctx->lib_handle, "bli_thread_set_num_threads");
    if (blis_set) {
        if (!getenv("BLIS_NUM_THREADS") && !getenv("OMP_NUM_THREADS")) {
            int ncores = 1;
#ifdef __linux__
            ncores = sysconf(_SC_NPROCESSORS_ONLN);
            if (ncores < 1) ncores = 1;
#endif
            blis_set(ncores);
            LOG_INFO("BLIS threads set to %d", ncores);
        }
        ctx->is_blis = true;
    }
}

CMLBlasContext* cml_blas_init(void) {
    CMLBlasContext* ctx = calloc(1, sizeof(CMLBlasContext));
    if (!ctx) {
        LOG_ERROR("Failed to allocate BLAS context");
        return NULL;
    }

    const char* env_blas = getenv("CML_BLAS_LIB");
    if (env_blas && env_blas[0] != '\0') {
        ctx->lib_handle = LIB_LOAD(env_blas);
        if (ctx->lib_handle) {
            load_cblas_functions(ctx);
            if (ctx->cblas_sgemm || ctx->sgemm_) {
                strncpy(ctx->lib_name, env_blas, sizeof(ctx->lib_name) - 1);
                ctx->initialized = true;
                tune_blas_threading(ctx);
                LOG_INFO("Loaded BLAS library (from CML_BLAS_LIB): %s", ctx->lib_name);
                return ctx;
            }
            LIB_CLOSE(ctx->lib_handle);
            ctx->lib_handle = NULL;
        }
        LOG_WARNING("Failed to load BLAS from CML_BLAS_LIB=%s, trying defaults", env_blas);
    }

    for (int i = 0; blas_library_paths[i] != NULL; i++) {
        ctx->lib_handle = LIB_LOAD(blas_library_paths[i]);
        if (ctx->lib_handle) {
            load_cblas_functions(ctx);

            // Verify we have at least sgemm
            if (ctx->cblas_sgemm || ctx->sgemm_) {
                strncpy(ctx->lib_name, blas_library_paths[i], sizeof(ctx->lib_name) - 1);
                ctx->initialized = true;
                tune_blas_threading(ctx);
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

    blas_lock();
    if (ctx->lib_handle) {
        LIB_CLOSE(ctx->lib_handle);
    }

    if (ctx == g_blas_ctx) {
        g_blas_ctx = NULL;
    }
    blas_unlock();

    free(ctx);
}

CMLBlasContext* cml_blas_get_context(void) {
    /* Fast path: context already initialized — no lock needed */
    CMLBlasContext* ctx = atomic_load_explicit(
        (_Atomic(CMLBlasContext*)*)&g_blas_ctx, memory_order_acquire);
    if (ctx)
        return ctx;

    /* Slow path: first call, initialize under lock */
    blas_lock();
    ctx = atomic_load_explicit((_Atomic(CMLBlasContext*)*)&g_blas_ctx, memory_order_relaxed);
    if (!ctx) {
        ctx = cml_blas_init();
        atomic_store_explicit((_Atomic(CMLBlasContext*)*)&g_blas_ctx, ctx, memory_order_release);
    }
    blas_unlock();
    return ctx;
}

#if defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>

/* Small-matrix GEMM: avoids BLAS thread pool overhead for M,N,K < 64 */
static void sgemm_avx_small(const float* A, const float* B, float* C,
                             int M, int N, int K, float alpha, float beta) {
    /* Apply beta to C */
    if (beta == 0.0f) {
        memset(C, 0, (size_t)M * N * sizeof(float));
    } else if (beta != 1.0f) {
        size_t total = (size_t)M * N;
        __m256 vbeta = _mm256_set1_ps(beta);
        size_t i = 0;
        for (; i + 8 <= total; i += 8)
            _mm256_storeu_ps(C + i, _mm256_mul_ps(_mm256_loadu_ps(C + i), vbeta));
        for (; i < total; i++)
            C[i] *= beta;
    }

    /* C += alpha * A @ B, using register blocking */
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            __m256 a_mk = _mm256_set1_ps(alpha * A[m * K + k]);
            int n = 0;
            for (; n + 16 <= N; n += 16) {
                /* Process 16 elements (2x unrolled) */
                __m256 c0 = _mm256_loadu_ps(C + m * N + n);
                __m256 c1 = _mm256_loadu_ps(C + m * N + n + 8);
                __m256 b0 = _mm256_loadu_ps(B + k * N + n);
                __m256 b1 = _mm256_loadu_ps(B + k * N + n + 8);
#ifdef __FMA__
                c0 = _mm256_fmadd_ps(a_mk, b0, c0);
                c1 = _mm256_fmadd_ps(a_mk, b1, c1);
#else
                c0 = _mm256_add_ps(c0, _mm256_mul_ps(a_mk, b0));
                c1 = _mm256_add_ps(c1, _mm256_mul_ps(a_mk, b1));
#endif
                _mm256_storeu_ps(C + m * N + n, c0);
                _mm256_storeu_ps(C + m * N + n + 8, c1);
            }
            for (; n + 8 <= N; n += 8) {
                __m256 c_val = _mm256_loadu_ps(C + m * N + n);
                __m256 b_val = _mm256_loadu_ps(B + k * N + n);
#ifdef __FMA__
                c_val = _mm256_fmadd_ps(a_mk, b_val, c_val);
#else
                c_val = _mm256_add_ps(c_val, _mm256_mul_ps(a_mk, b_val));
#endif
                _mm256_storeu_ps(C + m * N + n, c_val);
            }
            /* Scalar tail */
            float a_scalar = alpha * A[m * K + k];
            for (; n < N; n++)
                C[m * N + n] += a_scalar * B[k * N + n];
        }
    }
}
#endif /* __AVX2__ || __AVX__ */

/* Threshold: use AVX kernel for small matrices, BLAS for large ones.
 * Below this threshold, BLAS thread pool overhead > compute time. */
#define SMALL_GEMM_THRESHOLD (64 * 64 * 64)

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

    /* Small matrix fast path: AVX2 microkernel avoids BLAS thread overhead */
#if defined(__AVX2__) || defined(__AVX__)
    if ((long long)M * N * K < SMALL_GEMM_THRESHOLD) {
        sgemm_avx_small(A, B, C, M, N, K, alpha, beta);
        return 0;
    }
#endif

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
    if (!x || !y || n <= 0)
        return 0.0f;

    if (!ctx)
        ctx = cml_blas_get_context();

    if (ctx && ctx->cblas_sdot) {
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
    if (!x || n <= 0)
        return 0.0f;

    if (!ctx)
        ctx = cml_blas_get_context();

    if (ctx && ctx->cblas_snrm2) {
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
