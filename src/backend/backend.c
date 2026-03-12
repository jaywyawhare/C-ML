#include "backend/backend.h"
#include "backend/opencl_backend.h"
#include "core/logging.h"
#include "backend/device.h"
#include "ops/simd_utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <cpuid.h>
#endif

#ifdef __linux__
#include <dlfcn.h>
#define CML_DLOPEN(path, mode) dlopen(path, mode)
#define CML_DLSYM(handle, symbol) dlsym(handle, symbol)
#define CML_DLCLOSE(handle) dlclose(handle)
#ifndef RTLD_LAZY
#define RTLD_LAZY 1
#endif
#elif defined(__APPLE__)
#include <dlfcn.h>
#define CML_DLOPEN(path, mode) dlopen(path, mode)
#define CML_DLSYM(handle, symbol) dlsym(handle, symbol)
#define CML_DLCLOSE(handle) dlclose(handle)
#ifndef RTLD_LAZY
#define RTLD_LAZY 1
#endif
#elif defined(_WIN32)
#include <windows.h>
#define CML_DLOPEN(path, mode) LoadLibraryA(path)
#define CML_DLSYM(handle, symbol) GetProcAddress((HMODULE)handle, symbol)
#define CML_DLCLOSE(handle) FreeLibrary((HMODULE)handle)
#define RTLD_LAZY 0
#else
#define CML_DLOPEN(path, mode) NULL
#define CML_DLSYM(handle, symbol) NULL
#define CML_DLCLOSE(handle) ((void)0)
#define RTLD_LAZY 0
#endif

static pthread_mutex_t g_backend_lock;
static bool g_backend_lock_initialized = false;

static inline void backend_lock(void) {
    if (g_backend_lock_initialized) pthread_mutex_lock(&g_backend_lock);
}
static inline void backend_unlock(void) {
    if (g_backend_lock_initialized) pthread_mutex_unlock(&g_backend_lock);
}

static Backend* g_current_backend = NULL;

// Scalar backend implementation (default)
static void scalar_matmul(const void* a, const void* b, void* out, int m, int n, int k,
                          DType dtype) {
    if (!a || !b || !out || m <= 0 || n <= 0 || k <= 0) {
        LOG_ERROR("Invalid parameters for scalar_matmul");
        return;
    }

    if (dtype == DTYPE_FLOAT32) {
        const float* a_f = (const float*)a;
        const float* b_f = (const float*)b;
        float* out_f     = (float*)out;

        // C = A @ B where A is [m, k] and B is [k, n], C is [m, n]
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int l = 0; l < k; l++) {
                    sum += a_f[i * k + l] * b_f[l * n + j];
                }
                out_f[i * n + j] = sum;
            }
        }
    } else {
        LOG_ERROR("Unsupported dtype for scalar_matmul: %d", dtype);
    }
}

static void scalar_add(const void* a, const void* b, void* out, size_t n, DType dtype) {
    if (!a || !b || !out || n == 0) {
        LOG_ERROR("Invalid parameters for scalar_add");
        return;
    }

    if (dtype == DTYPE_FLOAT32) {
        const float* a_f = (const float*)a;
        const float* b_f = (const float*)b;
        float* out_f     = (float*)out;

        for (size_t i = 0; i < n; i++) {
            out_f[i] = a_f[i] + b_f[i];
        }
    } else {
        LOG_ERROR("Unsupported dtype for scalar_add: %d", dtype);
    }
}

static void scalar_mul(const void* a, const void* b, void* out, size_t n, DType dtype) {
    if (!a || !b || !out || n == 0) {
        LOG_ERROR("Invalid parameters for scalar_mul");
        return;
    }

    if (dtype == DTYPE_FLOAT32) {
        const float* a_f = (const float*)a;
        const float* b_f = (const float*)b;
        float* out_f     = (float*)out;

        for (size_t i = 0; i < n; i++) {
            out_f[i] = a_f[i] * b_f[i];
        }
    } else {
        LOG_ERROR("Unsupported dtype for scalar_mul: %d", dtype);
    }
}

static void scalar_relu(const void* a, void* out, size_t n, DType dtype) {
    if (!a || !out || n == 0) {
        LOG_ERROR("Invalid parameters for scalar_relu");
        return;
    }

    if (dtype == DTYPE_FLOAT32) {
        const float* a_f = (const float*)a;
        float* out_f     = (float*)out;

        for (size_t i = 0; i < n; i++) {
            out_f[i] = a_f[i] > 0.0f ? a_f[i] : 0.0f;
        }
    } else {
        LOG_ERROR("Unsupported dtype for scalar_relu: %d", dtype);
    }
}

static void scalar_sigmoid(const void* a, void* out, size_t n, DType dtype) {
    if (!a || !out || n == 0) {
        LOG_ERROR("Invalid parameters for scalar_sigmoid");
        return;
    }

    if (dtype == DTYPE_FLOAT32) {
        const float* a_f = (const float*)a;
        float* out_f     = (float*)out;

        for (size_t i = 0; i < n; i++) {
            // sigmoid(x) = 1 / (1 + exp(-x))
            float x  = a_f[i];
            out_f[i] = 1.0f / (1.0f + expf(-x));
        }
    } else {
        LOG_ERROR("Unsupported dtype for scalar_sigmoid: %d", dtype);
    }
}

static void scalar_sum(const void* a, void* out, size_t n, DType dtype) {
    if (!a || !out || n == 0) {
        LOG_ERROR("Invalid parameters for scalar_sum");
        return;
    }

    if (dtype == DTYPE_FLOAT32) {
        const float* a_f = (const float*)a;
        float* out_f     = (float*)out;

        float sum = 0.0f;
        for (size_t i = 0; i < n; i++) {
            sum += a_f[i];
        }
        *out_f = sum;
    } else {
        LOG_ERROR("Unsupported dtype for scalar_sum: %d", dtype);
    }
}

static void scalar_mean(const void* a, void* out, size_t n, DType dtype) {
    if (!a || !out || n == 0) {
        LOG_ERROR("Invalid parameters for scalar_mean");
        return;
    }

    if (dtype == DTYPE_FLOAT32) {
        const float* a_f = (const float*)a;
        float* out_f     = (float*)out;

        float sum = 0.0f;
        for (size_t i = 0; i < n; i++) {
            sum += a_f[i];
        }
        *out_f = sum / (float)n;
    } else {
        LOG_ERROR("Unsupported dtype for scalar_mean: %d", dtype);
    }
}

static void scalar_matmul_add(const void* a, const void* b, const void* c, void* out, int m, int n,
                              int k, DType dtype) {
    if (!a || !b || !c || !out || m <= 0 || n <= 0 || k <= 0) {
        LOG_ERROR("Invalid parameters for scalar_matmul_add");
        return;
    }

    // First compute matmul: out = a @ b
    scalar_matmul(a, b, out, m, n, k, dtype);

    // Then add c: out = out + c
    if (dtype == DTYPE_FLOAT32) {
        const float* c_f = (const float*)c;
        float* out_f     = (float*)out;

        for (int i = 0; i < m * n; i++) {
            out_f[i] += c_f[i];
        }
    }
}

static BackendOps scalar_ops = {.matmul     = scalar_matmul,
                                .matmul_add = scalar_matmul_add,
                                .add        = scalar_add,
                                .mul        = scalar_mul,
                                .relu       = scalar_relu,
                                .sigmoid    = scalar_sigmoid,
                                .sum        = scalar_sum,
                                .mean       = scalar_mean};

// SIMD-optimized backend implementations (AVX-256)
#ifdef __SSE__
#include <immintrin.h>
#include <emmintrin.h>

// SIMD-optimized add
static void simd_add(const void* a, const void* b, void* out, size_t n, DType dtype) {
    if (!a || !b || !out || n == 0) {
        LOG_ERROR("Invalid parameters for simd_add");
        return;
    }

    if (dtype == DTYPE_FLOAT32) {
        const float* a_f = (const float*)a;
        const float* b_f = (const float*)b;
        float* out_f     = (float*)out;

#ifdef __AVX__
        // AVX-256: process 8 floats at a time
        size_t simd_end = (n / 8) * 8;
        size_t i        = 0;
        for (; i < simd_end; i += 8) {
            __m256 va   = _mm256_loadu_ps(&a_f[i]);
            __m256 vb   = _mm256_loadu_ps(&b_f[i]);
            __m256 vout = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(&out_f[i], vout);
        }
        // Scalar remainder
        for (; i < n; i++) {
            out_f[i] = a_f[i] + b_f[i];
        }
#else
        // SSE: process 4 floats at a time
        size_t simd_end = (n / 4) * 4;
        size_t i        = 0;
        for (; i < simd_end; i += 4) {
            __m128 va   = _mm_loadu_ps(&a_f[i]);
            __m128 vb   = _mm_loadu_ps(&b_f[i]);
            __m128 vout = _mm_add_ps(va, vb);
            _mm_storeu_ps(&out_f[i], vout);
        }
        // Scalar remainder
        for (; i < n; i++) {
            out_f[i] = a_f[i] + b_f[i];
        }
#endif
    } else {
        LOG_ERROR("Unsupported dtype for simd_add: %d", dtype);
    }
}

// SIMD-optimized mul
static void simd_mul(const void* a, const void* b, void* out, size_t n, DType dtype) {
    if (!a || !b || !out || n == 0) {
        LOG_ERROR("Invalid parameters for simd_mul");
        return;
    }

    if (dtype == DTYPE_FLOAT32) {
        const float* a_f = (const float*)a;
        const float* b_f = (const float*)b;
        float* out_f     = (float*)out;

#ifdef __AVX__
        size_t simd_end = (n / 8) * 8;
        size_t i        = 0;
        for (; i < simd_end; i += 8) {
            __m256 va   = _mm256_loadu_ps(&a_f[i]);
            __m256 vb   = _mm256_loadu_ps(&b_f[i]);
            __m256 vout = _mm256_mul_ps(va, vb);
            _mm256_storeu_ps(&out_f[i], vout);
        }
        for (; i < n; i++) {
            out_f[i] = a_f[i] * b_f[i];
        }
#else
        size_t simd_end = (n / 4) * 4;
        size_t i        = 0;
        for (; i < simd_end; i += 4) {
            __m128 va   = _mm_loadu_ps(&a_f[i]);
            __m128 vb   = _mm_loadu_ps(&b_f[i]);
            __m128 vout = _mm_mul_ps(va, vb);
            _mm_storeu_ps(&out_f[i], vout);
        }
        for (; i < n; i++) {
            out_f[i] = a_f[i] * b_f[i];
        }
#endif
    } else {
        LOG_ERROR("Unsupported dtype for simd_mul: %d", dtype);
    }
}

// SIMD-optimized relu
static void simd_relu(const void* x, void* out, size_t n, DType dtype) {
    if (!x || !out || n == 0) {
        LOG_ERROR("Invalid parameters for simd_relu");
        return;
    }

    if (dtype == DTYPE_FLOAT32) {
        const float* x_f = (const float*)x;
        float* out_f     = (float*)out;

#ifdef __AVX__
        __m256 zero     = _mm256_setzero_ps();
        size_t simd_end = (n / 8) * 8;
        size_t i        = 0;
        for (; i < simd_end; i += 8) {
            __m256 vx   = _mm256_loadu_ps(&x_f[i]);
            __m256 vout = _mm256_max_ps(vx, zero);
            _mm256_storeu_ps(&out_f[i], vout);
        }
        for (; i < n; i++) {
            out_f[i] = x_f[i] > 0.0f ? x_f[i] : 0.0f;
        }
#else
        __m128 zero     = _mm_setzero_ps();
        size_t simd_end = (n / 4) * 4;
        size_t i        = 0;
        for (; i < simd_end; i += 4) {
            __m128 vx   = _mm_loadu_ps(&x_f[i]);
            __m128 vout = _mm_max_ps(vx, zero);
            _mm_storeu_ps(&out_f[i], vout);
        }
        for (; i < n; i++) {
            out_f[i] = x_f[i] > 0.0f ? x_f[i] : 0.0f;
        }
#endif
    } else {
        LOG_ERROR("Unsupported dtype for simd_relu: %d", dtype);
    }
}

// SIMD-optimized sigmoid
static void simd_sigmoid(const void* x, void* out, size_t n, DType dtype) {
    if (!x || !out || n == 0) {
        LOG_ERROR("Invalid parameters for simd_sigmoid");
        return;
    }

    if (dtype == DTYPE_FLOAT32) {
        const float* x_f = (const float*)x;
        float* out_f     = (float*)out;

        // Use scalar for sigmoid (exp is complex, SIMD exp approximation can be added later)
        for (size_t i = 0; i < n; i++) {
            float val = x_f[i];
            // Clamp to prevent overflow
            val      = val > 88.0f ? 88.0f : (val < -88.0f ? -88.0f : val);
            out_f[i] = 1.0f / (1.0f + expf(-val));
        }
    } else {
        LOG_ERROR("Unsupported dtype for simd_sigmoid: %d", dtype);
    }
}

// SIMD-optimized sum
static void simd_sum(const void* x, void* out, size_t n, DType dtype) {
    if (!x || !out || n == 0) {
        LOG_ERROR("Invalid parameters for simd_sum");
        return;
    }

    if (dtype == DTYPE_FLOAT32) {
        const float* x_f = (const float*)x;
        float* out_f     = (float*)out;

        *out_f = simd_sum_float(x_f, n);
    } else {
        LOG_ERROR("Unsupported dtype for simd_sum: %d", dtype);
    }
}

// SIMD-optimized mean
static void simd_mean(const void* x, void* out, size_t n, DType dtype) {
    if (!x || !out || n == 0) {
        LOG_ERROR("Invalid parameters for simd_mean");
        return;
    }

    if (dtype == DTYPE_FLOAT32) {
        const float* x_f = (const float*)x;
        float* out_f     = (float*)out;

        *out_f = simd_sum_float(x_f, n) / (float)n;
    } else {
        LOG_ERROR("Unsupported dtype for simd_mean: %d", dtype);
    }
}

// SIMD-optimized matmul (tiled approach)
static void simd_matmul(const void* a, const void* b, void* out, int m, int n, int k, DType dtype) {
    if (!a || !b || !out || m <= 0 || n <= 0 || k <= 0) {
        LOG_ERROR("Invalid parameters for simd_matmul");
        return;
    }

    if (dtype == DTYPE_FLOAT32) {
        const float* a_f = (const float*)a;
        const float* b_f = (const float*)b;
        float* out_f     = (float*)out;

#ifdef __AVX__
        // AVX-256 optimized matmul with tiling
        const int TILE_SIZE = 64; // Tile size for cache optimization

        // Initialize output to zero
        memset(out_f, 0, m * n * sizeof(float));

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j += 8) {
                __m256 sum_vec = _mm256_setzero_ps();
                int j_end      = (j + 8 < n) ? j + 8 : n;

                for (int kk = 0; kk < k; kk++) {
                    __m256 a_vec = _mm256_set1_ps(a_f[i * k + kk]);
                    if (j_end - j == 8) {
                        __m256 b_vec = _mm256_loadu_ps(&b_f[kk * n + j]);
#ifdef __FMA__
                        sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
#else
                        sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(a_vec, b_vec));
#endif
                    } else {
                        // Handle remainder
                        for (int jj = j; jj < j_end; jj++) {
                            out_f[i * n + jj] += a_f[i * k + kk] * b_f[kk * n + jj];
                        }
                    }
                }

                if (j_end - j == 8) {
                    _mm256_storeu_ps(&out_f[i * n + j], sum_vec);
                }
            }
        }
#else
        // SSE optimized matmul
        // Initialize output to zero
        memset(out_f, 0, (size_t)m * (size_t)n * sizeof(float));

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j += 4) {
                __m128 sum_vec = _mm_setzero_ps();
                int j_end      = (j + 4 < n) ? j + 4 : n;

                for (int kk = 0; kk < k; kk++) {
                    __m128 a_vec = _mm_set1_ps(a_f[i * k + kk]);
                    if (j_end - j == 4) {
                        __m128 b_vec = _mm_loadu_ps(&b_f[kk * n + j]);
                        sum_vec      = _mm_add_ps(sum_vec, _mm_mul_ps(a_vec, b_vec));
                    } else {
                        for (int jj = j; jj < j_end; jj++) {
                            out_f[i * n + jj] += a_f[i * k + kk] * b_f[kk * n + jj];
                        }
                    }
                }

                if (j_end - j == 4) {
                    _mm_storeu_ps(&out_f[i * n + j], sum_vec);
                }
            }
        }
#endif
    } else {
        LOG_ERROR("Unsupported dtype for simd_matmul: %d", dtype);
    }
}

// SIMD-optimized matmul_add
static void simd_matmul_add(const void* a, const void* b, const void* c, void* out, int m, int n,
                            int k, DType dtype) {
    // First compute matmul
    simd_matmul(a, b, out, m, n, k, dtype);

    // Then add bias
    if (dtype == DTYPE_FLOAT32 && c) {
        const float* c_f = (const float*)c;
        float* out_f     = (float*)out;

#ifdef __AVX__
        size_t total    = m * n;
        size_t simd_end = (total / 8) * 8;
        size_t i        = 0;
        for (; i < simd_end; i += 8) {
            __m256 vout = _mm256_loadu_ps(&out_f[i]);
            __m256 vc   = _mm256_loadu_ps(&c_f[i]);
            _mm256_storeu_ps(&out_f[i], _mm256_add_ps(vout, vc));
        }
        for (; i < total; i++) {
            out_f[i] += c_f[i];
        }
#else
        size_t total    = (size_t)m * (size_t)n;
        size_t simd_end = (total / 4) * 4;
        size_t i        = 0;
        for (; i < simd_end; i += 4) {
            __m128 vout = _mm_loadu_ps(&out_f[i]);
            __m128 vc   = _mm_loadu_ps(&c_f[i]);
            _mm_storeu_ps(&out_f[i], _mm_add_ps(vout, vc));
        }
        for (; i < total; i++) {
            out_f[i] += c_f[i];
        }
#endif
    }
}

static BackendOps simd_ops = {.matmul     = simd_matmul,
                              .matmul_add = simd_matmul_add,
                              .add        = simd_add,
                              .mul        = simd_mul,
                              .relu       = simd_relu,
                              .sigmoid    = simd_sigmoid,
                              .sum        = simd_sum,
                              .mean       = simd_mean};

#endif // __SSE__

Backend* backend_get_current(void) {
    backend_lock();
    if (!g_current_backend) {
        backend_unlock();
        backend_init(BACKEND_SCALAR);
        backend_lock();
    }
    Backend* result = g_current_backend;
    backend_unlock();
    return result;
}

int backend_set(BackendType type) { return backend_init(type); }

int backend_init(BackendType type) {
    if (!g_backend_lock_initialized) {
        pthread_mutex_init(&g_backend_lock, NULL);
        g_backend_lock_initialized = true;
    }

    backend_lock();
    if (g_current_backend && g_current_backend->type == type) {
        backend_unlock();
        return 0; // Already initialized
    }

    if (g_current_backend) {
        if (g_current_backend->context) {
            free(g_current_backend->context);
        }
        free(g_current_backend);
        g_current_backend = NULL;
    }

    g_current_backend = malloc(sizeof(Backend));
    if (!g_current_backend) {
        backend_unlock();
        LOG_ERROR("Failed to allocate backend");
        return -1;
    }

    g_current_backend->type    = type;
    g_current_backend->context = NULL;

    // Set operations based on backend type
    switch (type) {
    case BACKEND_SCALAR:
        g_current_backend->ops = scalar_ops;
        break;
#ifdef __SSE__
    case BACKEND_SSE:
    case BACKEND_AVX:
        // Use SIMD-optimized operations
        g_current_backend->ops = simd_ops;
        LOG_DEBUG("Backend %d using SIMD-optimized operations", type);
        break;
    case BACKEND_METAL:
        // Metal backend enhancement:
        // - Uses unified memory architecture on Apple Silicon (M1/M2/M3+)
        // - SIMD operations leverage CPU SIMD on unified memory (very efficient)
        // - Future: Can integrate Metal Performance Shaders (MPS) for GPU acceleration
        // - Future: Can use Metal compute shaders for custom kernels
        if (device_metal_available()) {
            g_current_backend->ops = simd_ops;

            // Try to detect Metal Performance Shaders availability
            // MPS is available on macOS 12.0+ and provides optimized GPU kernels
#ifdef __APPLE__
            void* mps_framework =
                CML_DLOPEN("/System/Library/Frameworks/MetalPerformanceShaders.framework/"
                           "MetalPerformanceShaders",
                           RTLD_LAZY);
            if (mps_framework) {
                LOG_INFO("Metal backend initialized: SIMD-optimized ops + MPS framework detected "
                         "(GPU acceleration available)");
                CML_DLCLOSE(mps_framework);
            } else {
                LOG_INFO("Metal backend initialized: SIMD-optimized operations (unified memory, "
                         "CPU SIMD)");
            }
#else
            LOG_INFO("Metal backend initialized: SIMD-optimized operations (unified memory)");
#endif
        } else {
            g_current_backend->ops = scalar_ops;
            LOG_WARNING("Metal not available, using scalar fallback");
        }
        break;
    case BACKEND_ROCM:
        // ROCm backend enhancement:
        // - Uses HIP runtime for GPU memory management
        // - SIMD operations run on CPU (data copied from GPU if needed)
        // - Future: Can integrate rocBLAS for optimized matrix operations
        // - Future: Can use HIP kernels for custom GPU operations
        if (device_rocm_available()) {
            g_current_backend->ops = simd_ops;

            // Try to detect rocBLAS availability for optimized matmul
            const char* rocblas_libs[] = {
#ifdef __linux__
                "librocblas.so", "librocblas.so.0",
#elif defined(__APPLE__)
                "librocblas.dylib",
#elif defined(_WIN32)
                "rocblas.dll",
#endif
                NULL};

            bool rocblas_available = false;
            for (int i = 0; rocblas_libs[i] != NULL; i++) {
                void* rocblas_lib = CML_DLOPEN(rocblas_libs[i], RTLD_LAZY);
                if (rocblas_lib) {
                    void* symbol = CML_DLSYM(rocblas_lib, "rocblas_sgemm");
                    if (symbol) {
                        rocblas_available = true;
                        LOG_INFO("ROCm backend initialized: SIMD-optimized ops + rocBLAS detected "
                                 "(GPU matmul available)");
                        CML_DLCLOSE(rocblas_lib);
                        break;
                    }
                    CML_DLCLOSE(rocblas_lib);
                }
            }

            if (!rocblas_available) {
                LOG_INFO(
                    "ROCm backend initialized: SIMD-optimized operations (CPU SIMD, GPU memory)");
            }
        } else {
            g_current_backend->ops = scalar_ops;
            LOG_WARNING("ROCm not available, using scalar fallback");
        }
        break;
    case BACKEND_BLAS:
    case BACKEND_CUDA:
        // BLAS and CUDA use scalar fallback for now (can be enhanced with BLAS/CUBLAS)
        g_current_backend->ops = scalar_ops;
        LOG_DEBUG("Backend %d using scalar fallback (optimized version not available)", type);
        break;
    case BACKEND_OPENCL:
        if (opencl_backend_init() == 0) {
            g_current_backend->ops = opencl_backend_get_ops();
            LOG_INFO("OpenCL backend initialized");
        } else {
            g_current_backend->ops = scalar_ops;
            LOG_WARNING("OpenCL not available, using scalar fallback");
        }
        break;
#else
    case BACKEND_SSE:
    case BACKEND_AVX:
    case BACKEND_BLAS:
    case BACKEND_CUDA:
    case BACKEND_METAL:
    case BACKEND_ROCM:
        // SIMD not available, use scalar fallback
        g_current_backend->ops = scalar_ops;
        LOG_DEBUG("Backend %d using scalar fallback (SIMD not available)", type);
        break;
    case BACKEND_OPENCL:
        if (opencl_backend_init() == 0) {
            g_current_backend->ops = opencl_backend_get_ops();
            LOG_INFO("OpenCL backend initialized");
        } else {
            g_current_backend->ops = scalar_ops;
            LOG_WARNING("OpenCL not available, using scalar fallback");
        }
        break;
#endif
    }

    backend_unlock();
    LOG_INFO("Backend initialized: %d", type);
    return 0;
}

void backend_cleanup(void) {
    backend_lock();
    if (g_current_backend) {
        if (g_current_backend->context) {
            free(g_current_backend->context);
        }
        free(g_current_backend);
        g_current_backend = NULL;
    }
    backend_unlock();
}

static bool check_cpu_feature_sse(void) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        // Check SSE bit (bit 25 in EDX)
        return (edx & (1 << 25)) != 0;
    }
#endif
    return false;
}

static bool check_cpu_feature_avx(void) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        // Check AVX bit (bit 28 in ECX)
        if ((ecx & (1 << 28)) == 0) {
            return false;
        }
        // Check OSXSAVE bit (bit 27 in ECX)
        if ((ecx & (1 << 27)) == 0) {
            return false;
        }
        // Check XGETBV support
        unsigned int xcr0 = 0;
        __asm__("xgetbv" : "=a"(xcr0) : "c"(0) : "edx");
        // Check if XMM and YMM states are enabled
        return (xcr0 & 0x6) == 0x6;
    }
#endif
    return false;
}

static bool check_blas_available(void) {
    const char* blas_symbols[] = {"cblas_sgemm", "cblas_dgemm", "sgemm_", "dgemm_", NULL};

    const char* blas_libs[] = {
#ifdef __linux__
        "libopenblas.so", "libblas.so", "libmkl_rt.so", "libatlas.so",
#elif defined(__APPLE__)
        "libopenblas.dylib",
        "libblas.dylib",
        "libmkl_rt.dylib",
        "libatlas.dylib",
        "/System/Library/Frameworks/Accelerate.framework/Accelerate",
#elif defined(_WIN32)
        "openblas.dll", "blas.dll", "mkl_rt.dll", "atlas.dll",
#endif
        NULL};

    // Try to load BLAS libraries
    for (int i = 0; blas_libs[i] != NULL; i++) {
        void* blas_lib = CML_DLOPEN(blas_libs[i], RTLD_LAZY);
        if (blas_lib) {
            // Check for BLAS symbols
            for (int j = 0; blas_symbols[j] != NULL; j++) {
                void* symbol = CML_DLSYM(blas_lib, blas_symbols[j]);
                if (symbol) {
                    CML_DLCLOSE(blas_lib);
                    LOG_DEBUG("BLAS detected: %s", blas_libs[i]);
                    return true;
                }
            }
            CML_DLCLOSE(blas_lib);
        }
    }

    return false;
}

bool backend_is_available(BackendType type) {
    switch (type) {
    case BACKEND_SCALAR:
        return true; // Always available
    case BACKEND_SSE:
        return check_cpu_feature_sse();
    case BACKEND_AVX:
        return check_cpu_feature_avx();
    case BACKEND_BLAS:
        return check_blas_available();
    case BACKEND_CUDA:
        return device_cuda_available();
    case BACKEND_METAL:
        return device_metal_available();
    case BACKEND_ROCM:
        return device_rocm_available();
    case BACKEND_OPENCL:
        return opencl_backend_is_available();
    default:
        return false;
    }
}
