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

static inline void blas_lock(void) { pthread_mutex_lock(&g_blas_lock); }
static inline void blas_unlock(void) { pthread_mutex_unlock(&g_blas_lock); }

static CMLBlasContext* g_blas_ctx = NULL;

/* ILP64 scipy_openblas64 paths — probed first on Linux because it is faster
 * than the system OpenBLAS (no OpenMP thread overhead, better-tuned kernels). */
static const char* ilp64_library_paths[] = {
#ifdef __linux__
    "/home/arrry/.local/lib/python3.14/site-packages/numpy.libs/libscipy_openblas64_-32a4b2a6.so",
    "/usr/local/lib/python3/dist-packages/numpy.libs/libscipy_openblas64_.so",
    "libscipy_openblas64_.so",
#endif
    NULL};

static const char* blas_library_paths[] = {
#ifdef __linux__
    /* MKL (fastest on Intel, competitive on AMD) */
    "libmkl_rt.so", "libmkl_rt.so.2", "libmkl_rt.so.1",
    /* BLIS (often beats OpenBLAS, especially on AMD) */
    "libblis.so", "libblis.so.4", "libblis.so.3",
    /* OpenBLAS */
    "libopenblas.so.0", "libopenblas.so",
    /* ATLAS, reference BLAS */
    "libatlas.so", "libcblas.so.3", "libcblas.so", "libblas.so.3", "libblas.so",
#elif defined(__APPLE__)
    "/System/Library/Frameworks/Accelerate.framework/Accelerate",
    "libopenblas.dylib",
    "/opt/homebrew/opt/openblas/lib/libopenblas.dylib",
    "/usr/local/opt/openblas/lib/libopenblas.dylib",
    "libblis.dylib",
    "libblas.dylib",
#elif defined(_WIN32)
    "mkl_rt.dll", "mkl_rt.2.dll", "libblis.dll", "openblas.dll", "libopenblas.dll", "blas.dll",
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

    if (!ctx->cblas_sgemm)
        ctx->sgemm_ = LIB_SYM(ctx->lib_handle, "sgemm_");
}

static bool load_ilp64_functions(CMLBlasContext* ctx) {
    if (!ctx || !ctx->lib_handle)
        return false;

    ctx->ilp64_sgemm = LIB_SYM(ctx->lib_handle, "scipy_cblas_sgemm64_");
    ctx->ilp64_sgemv = LIB_SYM(ctx->lib_handle, "scipy_cblas_sgemv64_");
    ctx->ilp64_saxpy = LIB_SYM(ctx->lib_handle, "scipy_cblas_saxpy64_");
    ctx->ilp64_sscal = LIB_SYM(ctx->lib_handle, "scipy_cblas_sscal64_");
    ctx->ilp64_sdot  = LIB_SYM(ctx->lib_handle, "scipy_cblas_sdot64_");
    ctx->ilp64_snrm2 = LIB_SYM(ctx->lib_handle, "scipy_cblas_snrm264_");
    ctx->ilp64_set_threads = LIB_SYM(ctx->lib_handle, "scipy_openblas_set_num_threads64_");

    return ctx->ilp64_sgemm != NULL;
}

static int default_thread_count(void) {
    int ncores = 1;
#ifdef __linux__
    ncores = sysconf(_SC_NPROCESSORS_ONLN);
    if (ncores < 1) ncores = 1;
#endif
    return ncores;
}

static void tune_blas_threading(CMLBlasContext* ctx) {
    if (!ctx || !ctx->lib_handle)
        return;

    ctx->max_threads = default_thread_count();
    ctx->cur_threads = ctx->max_threads;

    /* MKL: set threading via mkl_set_num_threads */
    void (*mkl_set_num_threads)(int) = LIB_SYM(ctx->lib_handle, "MKL_Set_Num_Threads");
    if (!mkl_set_num_threads)
        mkl_set_num_threads = LIB_SYM(ctx->lib_handle, "mkl_set_num_threads");
    if (mkl_set_num_threads) {
        const char* env = getenv("MKL_NUM_THREADS");
        if (!env) {
            mkl_set_num_threads(ctx->max_threads);
            LOG_INFO("MKL threads set to %d", ctx->max_threads);
        }
        ctx->fn_set_threads = mkl_set_num_threads;
        ctx->is_mkl = true;
        return;
    }

    /* OpenBLAS */
    void (*openblas_set)(int) = LIB_SYM(ctx->lib_handle, "openblas_set_num_threads");
    if (openblas_set) {
        if (!getenv("OPENBLAS_NUM_THREADS") && !getenv("OMP_NUM_THREADS")) {
            openblas_set(ctx->max_threads);
        }
        ctx->fn_set_threads = openblas_set;
        ctx->is_openblas = true;
        return;
    }

    /* BLIS */
    void (*blis_set)(int) = LIB_SYM(ctx->lib_handle, "bli_thread_set_num_threads");
    if (blis_set) {
        if (!getenv("BLIS_NUM_THREADS") && !getenv("OMP_NUM_THREADS")) {
            blis_set(ctx->max_threads);
        }
        ctx->fn_set_threads = blis_set;
        ctx->is_blis = true;
    }
}

/* Intentionally not switching per-call — OpenBLAS thread pool switching
 * is expensive (~50µs per openblas_set_num_threads call) and the global
 * thread count set at init time works well for large-matrix benchmarks.
 * Per-call adaptive threading made conv2d 6× slower in testing. */
static inline void blas_set_threads_for_size(CMLBlasContext* ctx, long long flops) {
    (void)ctx; (void)flops; /* no-op: use init-time thread count */
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
            /* Try ILP64 first (scipy_openblas64 layout) */
            if (load_ilp64_functions(ctx)) {
                strncpy(ctx->lib_name, env_blas, sizeof(ctx->lib_name) - 1);
                ctx->initialized = true;
                ctx->is_ilp64    = true;
                if (ctx->ilp64_set_threads) ctx->ilp64_set_threads(default_thread_count());
                fprintf(stderr, "[CML] BLAS (ILP64) loaded: %s\n", ctx->lib_name);
                return ctx;
            }
            load_cblas_functions(ctx);
            if (ctx->cblas_sgemm || ctx->sgemm_) {
                strncpy(ctx->lib_name, env_blas, sizeof(ctx->lib_name) - 1);
                ctx->initialized = true;
                tune_blas_threading(ctx);
                fprintf(stderr, "[CML] BLAS loaded: %s\n", ctx->lib_name);
                return ctx;
            }
            LIB_CLOSE(ctx->lib_handle);
            ctx->lib_handle = NULL;
        }
        LOG_WARNING("Failed to load BLAS from CML_BLAS_LIB=%s, trying defaults", env_blas);
    }

    /* Probe ILP64 paths before LP64 — scipy_openblas64 is significantly faster
     * on Linux due to no OpenMP threading overhead. */
    for (int i = 0; ilp64_library_paths[i] != NULL; i++) {
        ctx->lib_handle = LIB_LOAD(ilp64_library_paths[i]);
        if (ctx->lib_handle) {
            if (load_ilp64_functions(ctx)) {
                strncpy(ctx->lib_name, ilp64_library_paths[i], sizeof(ctx->lib_name) - 1);
                ctx->initialized = true;
                ctx->is_ilp64    = true;
                if (ctx->ilp64_set_threads) ctx->ilp64_set_threads(default_thread_count());
                fprintf(stderr, "[CML] BLAS (ILP64) loaded: %s\n", ctx->lib_name);
                return ctx;
            }
            LIB_CLOSE(ctx->lib_handle);
            ctx->lib_handle = NULL;
        }
    }

    for (int i = 0; blas_library_paths[i] != NULL; i++) {
        ctx->lib_handle = LIB_LOAD(blas_library_paths[i]);
        if (ctx->lib_handle) {
            load_cblas_functions(ctx);
            if (ctx->cblas_sgemm || ctx->sgemm_) {
                strncpy(ctx->lib_name, blas_library_paths[i], sizeof(ctx->lib_name) - 1);
                ctx->initialized = true;
                tune_blas_threading(ctx);
                fprintf(stderr, "[CML] BLAS loaded: %s\n", ctx->lib_name);
                return ctx;
            }
            LIB_CLOSE(ctx->lib_handle);
            ctx->lib_handle = NULL;
        }
    }

    fprintf(stderr, "[CML] WARNING: No BLAS library found — using scalar fallback\n");
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

    free(ctx->pack_a_buf);
    free(ctx->pack_b_buf);
    free(ctx);
}

void cml_blas_set_num_threads(int n) {
    if (n < 1)
        n = 1;
    CMLBlasContext* ctx = cml_blas_get_context();
    if (!ctx || !ctx->initialized)
        return;
    bool applied = false;
    if (ctx->is_ilp64) {
        if (ctx->ilp64_set_threads) {
            ctx->ilp64_set_threads((int64_t)n);
            applied = true;
        }
    } else if (ctx->fn_set_threads) {
        ctx->fn_set_threads(n);
        applied = true;
    }
    if (applied)
        ctx->cur_threads = n;
}

int cml_blas_get_num_threads(void) {
    CMLBlasContext* ctx = cml_blas_get_context();
    if (!ctx)
        return 1;
    return ctx->cur_threads > 0 ? ctx->cur_threads : 1;
}

CMLBlasContext* cml_blas_get_context(void) {
    /* Fast path: context already initialized — no lock needed */
    CMLBlasContext* ctx =
        atomic_load_explicit((_Atomic(CMLBlasContext*)*)&g_blas_ctx, memory_order_acquire);
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
static void sgemm_avx_small(const float* A, const float* B, float* C, int M, int N, int K,
                            float alpha, float beta) {
    /* Apply beta to C */
    if (beta == 0.0f) {
        memset(C, 0, (size_t)M * N * sizeof(float));
    } else if (beta != 1.0f) {
        size_t total = (size_t)M * N;
        __m256 vbeta = _mm256_set1_ps(beta);
        size_t i     = 0;
        for (; i + 8 <= total; i += 8)
            _mm256_storeu_ps(C + i, _mm256_mul_ps(_mm256_loadu_ps(C + i), vbeta));
        for (; i < total; i++)
            C[i] *= beta;
    }

    /* C += alpha * A @ B, using register blocking */
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            __m256 a_mk = _mm256_set1_ps(alpha * A[m * K + k]);
            int n       = 0;
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

/* ── Cache-blocked GEMM with B-packing (Approach 2) ─────────────────────────
 * Three-level BLIS-style tiling: NC/KC/MC match L3/L2/L1 cache hierarchy.
 * Hot loop: 6×16 FMA micro-kernel, B is pre-packed into sequential [k][n]
 * layout so the inner loop has no strided reads. */
#ifdef __FMA__
#define PACKED_MR    6
#define PACKED_NR   16    /* 2 × __m256 per k step */
#define PACKED_MC  120    /* L1/L2: MC × KC A-panel fits alongside packed B */
#define PACKED_KC  256    /* L2:    KC × NC B-panel ≤ ~256 KB */
#define PACKED_NC 2048    /* L3:    NC multiple of NR for clean packing */

/* Pack a (KC_cur × NR) column panel of B into sequential [k][n] order.
 * Columns beyond NR_cur are zero-padded so the microkernel always sees NR. */
static void pack_B_panel(const float* src, float* dst, int KC_cur, int NR_cur, int ldb) {
    for (int k = 0; k < KC_cur; k++) {
        int n = 0;
        for (; n < NR_cur; n++) dst[k * PACKED_NR + n] = src[k * ldb + n];
        for (; n < PACKED_NR; n++) dst[k * PACKED_NR + n] = 0.0f;
    }
}

/* Pack a (MR_cur × KC_cur) row panel of A transposed to [k][m] order.
 * Rows beyond MR_cur are zero-padded. */
static void pack_A_panel(const float* src, float* dst, int MR_cur, int KC_cur, int lda) {
    for (int k = 0; k < KC_cur; k++) {
        int m = 0;
        for (; m < MR_cur; m++) dst[k * PACKED_MR + m] = src[m * lda + k];
        for (; m < PACKED_MR; m++) dst[k * PACKED_MR + m] = 0.0f;
    }
}

/* 6×16 FMA micro-kernel.
 * A_p: [KC × MR] packed, stride PACKED_MR per k.
 * B_p: [KC × NR] packed, stride PACKED_NR per k.
 * Both are fully sequential — L1 cache never misses in the hot loop.
 * k loop unrolled ×4 to hide 4-cycle FMA latency (Haswell/Skylake). */
static void sgemm_ukr_6x16(float* C, int ldc,
                            const float* A_p, const float* B_p,
                            int KC, float alpha, float beta) {
    __m256 c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;

    if (beta == 0.0f) {
        c00=c01=c10=c11=c20=c21=c30=c31=c40=c41=c50=c51 = _mm256_setzero_ps();
    } else {
        __m256 vb = _mm256_set1_ps(beta);
        c00=_mm256_mul_ps(_mm256_loadu_ps(C+0*ldc+0),vb); c01=_mm256_mul_ps(_mm256_loadu_ps(C+0*ldc+8),vb);
        c10=_mm256_mul_ps(_mm256_loadu_ps(C+1*ldc+0),vb); c11=_mm256_mul_ps(_mm256_loadu_ps(C+1*ldc+8),vb);
        c20=_mm256_mul_ps(_mm256_loadu_ps(C+2*ldc+0),vb); c21=_mm256_mul_ps(_mm256_loadu_ps(C+2*ldc+8),vb);
        c30=_mm256_mul_ps(_mm256_loadu_ps(C+3*ldc+0),vb); c31=_mm256_mul_ps(_mm256_loadu_ps(C+3*ldc+8),vb);
        c40=_mm256_mul_ps(_mm256_loadu_ps(C+4*ldc+0),vb); c41=_mm256_mul_ps(_mm256_loadu_ps(C+4*ldc+8),vb);
        c50=_mm256_mul_ps(_mm256_loadu_ps(C+5*ldc+0),vb); c51=_mm256_mul_ps(_mm256_loadu_ps(C+5*ldc+8),vb);
    }

    int k = 0;
#define UKR_STEP(kk)                                                                               \
    do {                                                                                           \
        __m256 b0_ = _mm256_loadu_ps(B_p + (kk)*PACKED_NR + 0);                                  \
        __m256 b1_ = _mm256_loadu_ps(B_p + (kk)*PACKED_NR + 8);                                  \
        __m256 a_;                                                                                 \
        a_=_mm256_set1_ps(A_p[(kk)*PACKED_MR+0]); c00=_mm256_fmadd_ps(a_,b0_,c00); c01=_mm256_fmadd_ps(a_,b1_,c01); \
        a_=_mm256_set1_ps(A_p[(kk)*PACKED_MR+1]); c10=_mm256_fmadd_ps(a_,b0_,c10); c11=_mm256_fmadd_ps(a_,b1_,c11); \
        a_=_mm256_set1_ps(A_p[(kk)*PACKED_MR+2]); c20=_mm256_fmadd_ps(a_,b0_,c20); c21=_mm256_fmadd_ps(a_,b1_,c21); \
        a_=_mm256_set1_ps(A_p[(kk)*PACKED_MR+3]); c30=_mm256_fmadd_ps(a_,b0_,c30); c31=_mm256_fmadd_ps(a_,b1_,c31); \
        a_=_mm256_set1_ps(A_p[(kk)*PACKED_MR+4]); c40=_mm256_fmadd_ps(a_,b0_,c40); c41=_mm256_fmadd_ps(a_,b1_,c41); \
        a_=_mm256_set1_ps(A_p[(kk)*PACKED_MR+5]); c50=_mm256_fmadd_ps(a_,b0_,c50); c51=_mm256_fmadd_ps(a_,b1_,c51); \
    } while (0)
    for (; k + 4 <= KC; k += 4) { UKR_STEP(k); UKR_STEP(k+1); UKR_STEP(k+2); UKR_STEP(k+3); }
    for (; k < KC; k++)          { UKR_STEP(k); }
#undef UKR_STEP

    if (alpha == 1.0f) {
        _mm256_storeu_ps(C+0*ldc+0,c00); _mm256_storeu_ps(C+0*ldc+8,c01);
        _mm256_storeu_ps(C+1*ldc+0,c10); _mm256_storeu_ps(C+1*ldc+8,c11);
        _mm256_storeu_ps(C+2*ldc+0,c20); _mm256_storeu_ps(C+2*ldc+8,c21);
        _mm256_storeu_ps(C+3*ldc+0,c30); _mm256_storeu_ps(C+3*ldc+8,c31);
        _mm256_storeu_ps(C+4*ldc+0,c40); _mm256_storeu_ps(C+4*ldc+8,c41);
        _mm256_storeu_ps(C+5*ldc+0,c50); _mm256_storeu_ps(C+5*ldc+8,c51);
    } else {
        __m256 va = _mm256_set1_ps(alpha);
        _mm256_storeu_ps(C+0*ldc+0,_mm256_mul_ps(va,c00)); _mm256_storeu_ps(C+0*ldc+8,_mm256_mul_ps(va,c01));
        _mm256_storeu_ps(C+1*ldc+0,_mm256_mul_ps(va,c10)); _mm256_storeu_ps(C+1*ldc+8,_mm256_mul_ps(va,c11));
        _mm256_storeu_ps(C+2*ldc+0,_mm256_mul_ps(va,c20)); _mm256_storeu_ps(C+2*ldc+8,_mm256_mul_ps(va,c21));
        _mm256_storeu_ps(C+3*ldc+0,_mm256_mul_ps(va,c30)); _mm256_storeu_ps(C+3*ldc+8,_mm256_mul_ps(va,c31));
        _mm256_storeu_ps(C+4*ldc+0,_mm256_mul_ps(va,c40)); _mm256_storeu_ps(C+4*ldc+8,_mm256_mul_ps(va,c41));
        _mm256_storeu_ps(C+5*ldc+0,_mm256_mul_ps(va,c50)); _mm256_storeu_ps(C+5*ldc+8,_mm256_mul_ps(va,c51));
    }
}

/* Edge micro-kernel for tiles smaller than MR×NR (boundaries only).
 * Routes through the full 6×16 kernel via a padded stack buffer to avoid
 * a separate scalar path. */
static void sgemm_ukr_edge(float* C, int ldc,
                           const float* A_p, const float* B_p,
                           int MR_cur, int NR_cur, int KC, float alpha, float beta) {
    float buf[PACKED_MR * PACKED_NR] __attribute__((aligned(32)));
    memset(buf, 0, sizeof(buf));
    if (beta != 0.0f)
        for (int r = 0; r < MR_cur; r++)
            for (int c = 0; c < NR_cur; c++)
                buf[r * PACKED_NR + c] = C[r * ldc + c] * beta;
    sgemm_ukr_6x16(buf, PACKED_NR, A_p, B_p, KC, alpha, 1.0f);
    for (int r = 0; r < MR_cur; r++)
        for (int c = 0; c < NR_cur; c++)
            C[r * ldc + c] = buf[r * PACKED_NR + c];
}

/* Three-level tiled GEMM: C = alpha*A*B + beta*C.
 * a_pack: ≥ PACKED_MC * PACKED_KC floats (aligned to 32 bytes).
 * b_pack: ≥ ceil(N/NR)*NR * PACKED_KC floats (aligned to 32 bytes).
 * Caller owns allocation and lifetime of both scratch buffers. */
static void sgemm_packed_avx(const float* A, const float* B, float* C,
                              int M, int N, int K, float alpha, float beta,
                              float* a_pack, float* b_pack) {
    for (int jc = 0; jc < N; jc += PACKED_NC) {
        int NC_cur = ((jc + PACKED_NC) > N) ? (N - jc) : PACKED_NC;

        for (int pc = 0; pc < K; pc += PACKED_KC) {
            int KC_cur = ((pc + PACKED_KC) > K) ? (K - pc) : PACKED_KC;
            float use_beta = (pc == 0) ? beta : 1.0f;

            /* Pack B strip (KC_cur × NC_cur) into NR-wide panels */
            for (int jr = 0; jr < NC_cur; jr += PACKED_NR) {
                int NR_cur = ((jr + PACKED_NR) > NC_cur) ? (NC_cur - jr) : PACKED_NR;
                pack_B_panel(B + pc * N + jc + jr,
                             b_pack + (jr / PACKED_NR) * KC_cur * PACKED_NR,
                             KC_cur, NR_cur, N);
            }

            for (int ic = 0; ic < M; ic += PACKED_MC) {
                int MC_cur = ((ic + PACKED_MC) > M) ? (M - ic) : PACKED_MC;

                /* Pack A strip (MC_cur × KC_cur) into MR-wide panels */
                for (int ir = 0; ir < MC_cur; ir += PACKED_MR) {
                    int MR_cur = ((ir + PACKED_MR) > MC_cur) ? (MC_cur - ir) : PACKED_MR;
                    pack_A_panel(A + (ic + ir) * K + pc,
                                 a_pack + (ir / PACKED_MR) * KC_cur * PACKED_MR,
                                 MR_cur, KC_cur, K);
                }

                /* Micro-kernel sweep over the MC × NC output tile */
                for (int jr = 0; jr < NC_cur; jr += PACKED_NR) {
                    int NR_cur = ((jr + PACKED_NR) > NC_cur) ? (NC_cur - jr) : PACKED_NR;
                    const float* B_panel = b_pack + (jr / PACKED_NR) * KC_cur * PACKED_NR;

                    for (int ir = 0; ir < MC_cur; ir += PACKED_MR) {
                        int MR_cur = ((ir + PACKED_MR) > MC_cur) ? (MC_cur - ir) : PACKED_MR;
                        const float* A_panel = a_pack + (ir / PACKED_MR) * KC_cur * PACKED_MR;
                        float* C_tile = C + (ic + ir) * N + (jc + jr);

                        if (MR_cur == PACKED_MR && NR_cur == PACKED_NR)
                            sgemm_ukr_6x16(C_tile, N, A_panel, B_panel, KC_cur, alpha, use_beta);
                        else
                            sgemm_ukr_edge(C_tile, N, A_panel, B_panel,
                                           MR_cur, NR_cur, KC_cur, alpha, use_beta);
                    }
                }
            }
        }
    }
}

/* Threshold above which scipy_openblas64 beats the packed kernel.
 * Below this limit, even ILP64 BLAS has thread-spinup overhead that the
 * packed kernel avoids by staying single-threaded. */
#define MEDIUM_GEMM_THRESHOLD (4LL * 1024 * 1024)

#endif /* __FMA__ */
#endif /* __AVX2__ || __AVX__ */

/* Threshold: use AVX kernel for small matrices, BLAS for large ones.
 * Below this threshold, BLAS thread pool overhead > compute time. */
/* AVX2 microkernel threshold: very small matrices always use it. */
#define SMALL_GEMM_THRESHOLD (64 * 64 * 64)

/* Inference batch threshold: also use AVX2 when M is small (batch ≤ 128) and
 * the weight footprint (N*K) fits in L2 cache (~256KB = 65536 floats).
 * Avoids OpenBLAS OpenMP thread-wake overhead for typical MLP inference shapes
 * (e.g. batch=64, 784→128 hits OpenBLAS OpenMP but AVX2 is 3× faster here). */
/* Inference shapes: M≤128 (small batch), N×K ≤ 110K floats (weight fits in L2).
 * Covers MLP: batch=64, 784→128 (N*K=100352) and 128→10 (N*K=1280).
 * Excludes conv2d: M=16 but N*K=194400 stays on OpenBLAS. */
#define INFERENCE_BATCH_THRESHOLD 128
#define INFERENCE_WEIGHT_THRESHOLD 110000

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

    /* Small matrix fast path: AVX2 microkernel beats both LP64 and ILP64 BLAS
     * for matrices small enough to avoid fork-join overhead. */
#if defined(__AVX2__) || defined(__AVX__)
    {
        long long flops = (long long)M * N * K;
        bool use_avx = flops < SMALL_GEMM_THRESHOLD ||
                       (!ctx->is_ilp64 &&
                        M <= INFERENCE_BATCH_THRESHOLD &&
                        (long long)N * K <= INFERENCE_WEIGHT_THRESHOLD);
        if (use_avx) {
            sgemm_avx_small(A, B, C, M, N, K, alpha, beta);
            return 0;
        }
#ifdef __FMA__
        /* Medium-matrix path: packed 6×16 kernel.
         * For ILP64: only use when both flops are small AND N fits in one NC-panel
         * (N > PACKED_NC means ILP64's multi-threading beats the single-threaded
         * packed kernel even at small flops — e.g. im2col GEMM M=16, N=7200, K=27).
         * Without ILP64: always use packed (far better than LP64 OpenBLAS-OpenMP). */
        bool use_packed = !ctx->is_ilp64 ||
                          (flops < MEDIUM_GEMM_THRESHOLD && N <= PACKED_NC);
        if (use_packed) {
            int nc_alloc = ((N + PACKED_NR - 1) / PACKED_NR) * PACKED_NR;
            if (nc_alloc > PACKED_NC) nc_alloc = PACKED_NC;
            size_t need_a = (size_t)PACKED_MC * PACKED_KC * sizeof(float);
            size_t need_b = (size_t)nc_alloc * PACKED_KC * sizeof(float);
            /* Grow context-resident buffers lazily — amortises alloc across calls */
            if (ctx->pack_a_size < need_a) {
                free(ctx->pack_a_buf);
                ctx->pack_a_buf  = (float*)aligned_alloc(32, need_a);
                ctx->pack_a_size = ctx->pack_a_buf ? need_a : 0;
            }
            if (ctx->pack_b_size < need_b) {
                free(ctx->pack_b_buf);
                ctx->pack_b_buf  = (float*)aligned_alloc(32, need_b);
                ctx->pack_b_size = ctx->pack_b_buf ? need_b : 0;
            }
            if (ctx->pack_a_buf && ctx->pack_b_buf) {
                sgemm_packed_avx(A, B, C, M, N, K, alpha, beta,
                                 ctx->pack_a_buf, ctx->pack_b_buf);
                return 0;
            }
        }
#endif /* __FMA__ */
    }
#endif /* __AVX2__ || __AVX__ */

    if (ctx->is_ilp64 && ctx->ilp64_sgemm) {
        int64_t ord = CML_BLAS_ROW_MAJOR, nt = CML_BLAS_NO_TRANS;
        ctx->ilp64_sgemm(ord, nt, nt, M, N, K, alpha, A, K, B, N, beta, C, N);
        return 0;
    }
    if (ctx->cblas_sgemm) {
        ctx->cblas_sgemm(CML_BLAS_ROW_MAJOR, CML_BLAS_NO_TRANS, CML_BLAS_NO_TRANS, M, N, K, alpha,
                         A, K, B, N, beta, C, N);
        return 0;
    }
    if (ctx->sgemm_) {
        char tA = 'N', tB = 'N';
        ctx->sgemm_(&tA, &tB, &N, &M, &K, &alpha, B, &N, A, &K, &beta, C, &N);
        return 0;
    }
    LOG_ERROR("No sgemm function available");
    return -1;
}

int cml_blas_sgemm_ex(CMLBlasContext* ctx, const float* A, const float* B, float* C, int M, int N,
                      int K, float alpha, float beta, bool transA, bool transB) {
    if (!ctx)
        ctx = cml_blas_get_context();
    if (!ctx || !ctx->initialized || !A || !B || !C || M <= 0 || N <= 0 || K <= 0)
        return -1;

    int ta  = transA ? CML_BLAS_TRANS : CML_BLAS_NO_TRANS;
    int tb  = transB ? CML_BLAS_TRANS : CML_BLAS_NO_TRANS;
    int lda = transA ? M : K;
    int ldb = transB ? K : N;

    if (ctx->is_ilp64 && ctx->ilp64_sgemm) {
        ctx->ilp64_sgemm((int64_t)CML_BLAS_ROW_MAJOR, (int64_t)ta, (int64_t)tb,
                         (int64_t)M, (int64_t)N, (int64_t)K,
                         alpha, A, (int64_t)lda, B, (int64_t)ldb, beta, C, (int64_t)N);
        return 0;
    }
    if (ctx->cblas_sgemm) {
        ctx->cblas_sgemm(CML_BLAS_ROW_MAJOR, ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
        return 0;
    }
    if (ctx->sgemm_) {
        char fta = transB ? 'T' : 'N';
        char ftb = transA ? 'T' : 'N';
        int flda = transB ? K : N;
        int fldb = transA ? M : K;
        ctx->sgemm_(&fta, &ftb, &N, &M, &K, &alpha, B, &flda, A, &fldb, &beta, C, &N);
        return 0;
    }
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

    if (ctx->is_ilp64 && ctx->ilp64_sgemv) {
        ctx->ilp64_sgemv((int64_t)CML_BLAS_ROW_MAJOR, (int64_t)CML_BLAS_NO_TRANS,
                         (int64_t)M, (int64_t)N, alpha, A, (int64_t)N, x, (int64_t)1,
                         beta, y, (int64_t)1);
        return 0;
    }
    if (ctx->cblas_sgemv) {
        ctx->cblas_sgemv(CML_BLAS_ROW_MAJOR, CML_BLAS_NO_TRANS, M, N, alpha, A, N, x, 1, beta, y,
                         1);
        return 0;
    }

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

    if (ctx->is_ilp64 && ctx->ilp64_saxpy) {
        ctx->ilp64_saxpy((int64_t)n, alpha, x, (int64_t)1, y, (int64_t)1);
        return 0;
    }
    if (ctx->cblas_saxpy) {
        ctx->cblas_saxpy(n, alpha, x, 1, y, 1);
        return 0;
    }

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

    if (ctx->is_ilp64 && ctx->ilp64_sscal) {
        ctx->ilp64_sscal((int64_t)n, alpha, x, (int64_t)1);
        return 0;
    }
    if (ctx->cblas_sscal) {
        ctx->cblas_sscal(n, alpha, x, 1);
        return 0;
    }

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

    if (ctx && ctx->is_ilp64 && ctx->ilp64_sdot) {
        return ctx->ilp64_sdot((int64_t)n, x, (int64_t)1, y, (int64_t)1);
    }
    if (ctx && ctx->cblas_sdot) {
        return ctx->cblas_sdot(n, x, 1, y, 1);
    }

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

    if (ctx && ctx->is_ilp64 && ctx->ilp64_snrm2) {
        return ctx->ilp64_snrm2((int64_t)n, x, (int64_t)1);
    }
    if (ctx && ctx->cblas_snrm2) {
        return ctx->cblas_snrm2(n, x, 1);
    }

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
