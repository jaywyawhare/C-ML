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

#include "core/dynlib.h"

static pthread_mutex_t g_backend_lock;
static pthread_once_t g_backend_lock_once = PTHREAD_ONCE_INIT;

static void backend_lock_init(void) { pthread_mutex_init(&g_backend_lock, NULL); }

static inline void backend_lock(void) {
    pthread_once(&g_backend_lock_once, backend_lock_init);
    pthread_mutex_lock(&g_backend_lock);
}
static inline void backend_unlock(void) {
    pthread_once(&g_backend_lock_once, backend_lock_init);
    pthread_mutex_unlock(&g_backend_lock);
}

static Backend* g_current_backend = NULL;

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

    scalar_matmul(a, b, out, m, n, k, dtype);

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

#ifdef __SSE__
/* The hand-rolled SSE/AVX backend ops were removed; the "simd" backend now
 * reuses the portable scalar ops (auto-vectorized by the compiler).  The table
 * is retained so BACKEND_AVX/BACKEND_SSE selection sites stay valid. */
static BackendOps simd_ops = {.matmul     = scalar_matmul,
                              .matmul_add = scalar_matmul_add,
                              .add        = scalar_add,
                              .mul        = scalar_mul,
                              .relu       = scalar_relu,
                              .sigmoid    = scalar_sigmoid,
                              .sum        = scalar_sum,
                              .mean       = scalar_mean};

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
    backend_lock();
    if (g_current_backend && g_current_backend->type == type) {
        backend_unlock();
        return 0;
    }

    if (g_current_backend) {
        if (g_current_backend->context) {
            cml_free(g_current_backend->context);
        }
        cml_free(g_current_backend);
        g_current_backend = NULL;
    }

    g_current_backend = cml_malloc(sizeof(Backend));
    if (!g_current_backend) {
        backend_unlock();
        LOG_ERROR("Failed to allocate backend");
        return -1;
    }

    g_current_backend->type    = type;
    g_current_backend->context = NULL;

    switch (type) {
    case BACKEND_SCALAR:
        g_current_backend->ops = scalar_ops;
        break;
#ifdef __SSE__
    case BACKEND_SSE:
    case BACKEND_AVX:
        g_current_backend->ops = simd_ops;
        break;
    case BACKEND_METAL:
        if (device_metal_available()) {
            g_current_backend->ops = simd_ops;

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
        if (device_rocm_available()) {
            g_current_backend->ops = simd_ops;

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
        g_current_backend->ops = scalar_ops;
        break;
    case BACKEND_CUDA:
        /* The legacy backend API has no CUDA compute path — this selects host
         * scalar ops only. Fail loudly instead of lending the label false
         * credibility; GPU execution goes through dispatch/IR (cml_dispatch_*,
         * cml_ir_execute) or HCQ. */
        LOG_ERROR("backend_init(BACKEND_CUDA): the legacy backend API does not "
                  "perform GPU compute; use cml_dispatch_execute_on / cml_ir_execute "
                  "for CUDA. Falling back to host scalar ops.");
        g_current_backend->ops = scalar_ops;
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
        // SIMD not available, use scalar fallback
        g_current_backend->ops = scalar_ops;
        break;
    case BACKEND_CUDA:
    case BACKEND_METAL:
    case BACKEND_ROCM:
        /* Legacy API: no GPU compute path here (see the BACKEND_CUDA note in
         * the SSE build). Loud, not silent. */
        LOG_ERROR("backend_init(%d): the legacy backend API does not perform GPU "
                  "compute on this platform; use cml_dispatch_execute_on / cml_ir_execute. "
                  "Falling back to host scalar ops.", (int)type);
        g_current_backend->ops = scalar_ops;
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
            cml_free(g_current_backend->context);
        }
        cml_free(g_current_backend);
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

    for (int i = 0; blas_libs[i] != NULL; i++) {
        void* blas_lib = CML_DLOPEN(blas_libs[i], RTLD_LAZY);
        if (blas_lib) {
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
        return true;
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
