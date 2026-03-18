#include "ops/simd_math.h"
#include "ops/simd_utils.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define CML_X86 1
#ifdef __SSE__
#include <xmmintrin.h> // SSE
#include <emmintrin.h> // SSE2
#define CML_HAS_SSE_COMPILE 1
#endif
#ifdef __SSE4_1__
#include <smmintrin.h> // SSE4.1
#define CML_HAS_SSE4_COMPILE 1
#endif
#ifdef __AVX__
#include <immintrin.h> // AVX, AVX2, AVX-512
#define CML_HAS_AVX_COMPILE 1
#endif
#ifdef __AVX2__
#define CML_HAS_AVX2_COMPILE 1
#endif
#ifdef __AVX512F__
#define CML_HAS_AVX512_COMPILE 1
#endif
#elif defined(__aarch64__) || defined(__arm__)
#define CML_ARM 1
#ifdef __ARM_NEON
#include <arm_neon.h>
#define CML_HAS_NEON_COMPILE 1
#endif
#endif

#if defined(__linux__) || defined(__APPLE__)
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

static void* g_sleef_handle = NULL;
static int g_sleef_loaded   = 0;
static int g_sleef_tried    = 0;

#ifdef CML_HAS_AVX_COMPILE
typedef __m256 (*sleef_f8_t)(__m256);
typedef __m256 (*sleef_f8f8_t)(__m256, __m256);
#endif

#ifdef CML_HAS_AVX512_COMPILE
typedef __m512 (*sleef_f16_t)(__m512);
typedef __m512 (*sleef_f16f16_t)(__m512, __m512);
#endif

#ifdef CML_HAS_SSE_COMPILE
typedef __m128 (*sleef_f4_t)(__m128);
typedef __m128 (*sleef_f4f4_t)(__m128, __m128);
#endif

#ifdef CML_HAS_AVX_COMPILE
static sleef_f8_t sleef_expf8   = NULL;
static sleef_f8_t sleef_logf8   = NULL;
static sleef_f8_t sleef_sinf8   = NULL;
static sleef_f8_t sleef_cosf8   = NULL;
static sleef_f8_t sleef_tanf8   = NULL;
static sleef_f8_t sleef_tanhf8  = NULL;
static sleef_f8f8_t sleef_powf8 = NULL;
#endif

#ifdef CML_HAS_AVX512_COMPILE
static sleef_f16_t sleef_expf16    = NULL;
static sleef_f16_t sleef_logf16    = NULL;
static sleef_f16_t sleef_sinf16    = NULL;
static sleef_f16_t sleef_cosf16    = NULL;
static sleef_f16_t sleef_tanf16    = NULL;
static sleef_f16_t sleef_tanhf16   = NULL;
static sleef_f16f16_t sleef_powf16 = NULL;
#endif

#ifdef CML_X86
static void cpuid(int info[4], int leaf) {
#ifdef _MSC_VER
    __cpuid(info, leaf);
#else
    __asm__ __volatile__("cpuid"
                         : "=a"(info[0]), "=b"(info[1]), "=c"(info[2]), "=d"(info[3])
                         : "a"(leaf), "c"(0));
#endif
}

static void cpuid_count(int info[4], int leaf, int subleaf) {
#ifdef _MSC_VER
    __cpuidex(info, leaf, subleaf);
#else
    __asm__ __volatile__("cpuid"
                         : "=a"(info[0]), "=b"(info[1]), "=c"(info[2]), "=d"(info[3])
                         : "a"(leaf), "c"(subleaf));
#endif
}

static unsigned long long xgetbv(unsigned int index) {
#ifdef _MSC_VER
    return _xgetbv(index);
#else
    unsigned int eax, edx;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
    return ((unsigned long long)edx << 32) | eax;
#endif
}
#endif

static int try_load_sleef(void) {
    if (g_sleef_tried)
        return g_sleef_loaded;
    g_sleef_tried = 1;

    const char* sleef_paths[] = {
#ifdef __linux__
        "libsleef.so", "libsleef.so.3", "/usr/lib/libsleef.so", "/usr/local/lib/libsleef.so",
#elif defined(__APPLE__)
        "libsleef.dylib", "/usr/local/lib/libsleef.dylib", "/opt/homebrew/lib/libsleef.dylib",
#elif defined(_WIN32)
        "sleef.dll", "libsleef.dll",
#endif
        NULL};

    for (int i = 0; sleef_paths[i] != NULL; i++) {
        g_sleef_handle = LIB_LOAD(sleef_paths[i]);
        if (g_sleef_handle) {
            // Try to load AVX functions
#ifdef CML_HAS_AVX_COMPILE
            sleef_expf8  = (sleef_f8_t)LIB_SYM(g_sleef_handle, "Sleef_expf8_u10");
            sleef_logf8  = (sleef_f8_t)LIB_SYM(g_sleef_handle, "Sleef_logf8_u10");
            sleef_sinf8  = (sleef_f8_t)LIB_SYM(g_sleef_handle, "Sleef_sinf8_u10");
            sleef_cosf8  = (sleef_f8_t)LIB_SYM(g_sleef_handle, "Sleef_cosf8_u10");
            sleef_tanf8  = (sleef_f8_t)LIB_SYM(g_sleef_handle, "Sleef_tanf8_u10");
            sleef_tanhf8 = (sleef_f8_t)LIB_SYM(g_sleef_handle, "Sleef_tanhf8_u10");
            sleef_powf8  = (sleef_f8f8_t)LIB_SYM(g_sleef_handle, "Sleef_powf8_u10");
#endif
#ifdef CML_HAS_AVX512_COMPILE
            sleef_expf16  = (sleef_f16_t)LIB_SYM(g_sleef_handle, "Sleef_expf16_u10");
            sleef_logf16  = (sleef_f16_t)LIB_SYM(g_sleef_handle, "Sleef_logf16_u10");
            sleef_sinf16  = (sleef_f16_t)LIB_SYM(g_sleef_handle, "Sleef_sinf16_u10");
            sleef_cosf16  = (sleef_f16_t)LIB_SYM(g_sleef_handle, "Sleef_cosf16_u10");
            sleef_tanf16  = (sleef_f16_t)LIB_SYM(g_sleef_handle, "Sleef_tanf16_u10");
            sleef_tanhf16 = (sleef_f16_t)LIB_SYM(g_sleef_handle, "Sleef_tanhf16_u10");
            sleef_powf16  = (sleef_f16f16_t)LIB_SYM(g_sleef_handle, "Sleef_powf16_u10");
#endif
            g_sleef_loaded = 1;
            return 1;
        }
    }
    return 0;
}

static CMLSimdCaps g_caps = {0};
static int g_caps_init    = 0;

CMLSimdCaps cml_detect_simd_caps(void) {
    CMLSimdCaps caps = {0};

#ifdef CML_X86
    int info[4];
    cpuid(info, 0);
    int max_leaf = info[0];

    if (max_leaf >= 1) {
        cpuid(info, 1);
        caps.has_sse  = (info[3] & (1 << 25)) != 0; // SSE
        caps.has_sse4 = (info[2] & (1 << 19)) != 0; // SSE4.1

        bool os_uses_xsave = (info[2] & (1 << 27)) != 0;
        bool cpu_has_avx   = (info[2] & (1 << 28)) != 0;

        if (os_uses_xsave && cpu_has_avx) {
            unsigned long long xcr0 = xgetbv(0);
            bool os_saves_ymm       = (xcr0 & 6) == 6; // XMM and YMM state

            if (os_saves_ymm) {
                caps.has_avx = true;

                if (max_leaf >= 7) {
                    cpuid_count(info, 7, 0);
                    caps.has_avx2 = (info[1] & (1 << 5)) != 0;

                    bool os_saves_zmm = (xcr0 & 0xE0) == 0xE0; // ZMM state
                    if (os_saves_zmm) {
                        caps.has_avx512 = (info[1] & (1 << 16)) != 0; // AVX-512F
                    }
                }
            }
        }
    }
#endif

#ifdef CML_ARM
#ifdef CML_HAS_NEON_COMPILE
    caps.has_neon = true;
#endif
#endif

    caps.has_sleef = try_load_sleef();

    return caps;
}

const CMLSimdCaps* cml_get_simd_caps(void) {
    if (!g_caps_init) {
        g_caps      = cml_detect_simd_caps();
        g_caps_init = 1;
    }
    return &g_caps;
}

void cml_print_simd_caps(void) {
    const CMLSimdCaps* caps = cml_get_simd_caps();
    printf("\nSIMD Capabilities\n");
    printf("SSE:     %s\n", caps->has_sse ? "Yes" : "No");
    printf("SSE4.1:  %s\n", caps->has_sse4 ? "Yes" : "No");
    printf("AVX:     %s\n", caps->has_avx ? "Yes" : "No");
    printf("AVX2:    %s\n", caps->has_avx2 ? "Yes" : "No");
    printf("AVX-512: %s\n", caps->has_avx512 ? "Yes" : "No");
    printf("NEON:    %s\n", caps->has_neon ? "Yes" : "No");
    printf("SLEEF:   %s\n", caps->has_sleef ? "Yes" : "No");
    printf("\n");
}

#define EXP_LN2 0.6931471805599453f
#define EXP_INV_LN2 1.4426950408889634f
#define EXP_C0 1.0f
#define EXP_C1 1.0f
#define EXP_C2 0.5f
#define EXP_C3 0.16666666666666666f
#define EXP_C4 0.041666666666666664f
#define EXP_C5 0.008333333333333333f
#define EXP_C6 0.001388888888888889f

#define LOG_LN2 0.6931471805599453f

#ifdef CML_HAS_AVX512_COMPILE

static inline __m512 exp_poly_avx512(__m512 x) {
    // Clamp input to prevent overflow/underflow
    __m512 max_val = _mm512_set1_ps(88.0f);
    __m512 min_val = _mm512_set1_ps(-88.0f);
    x              = _mm512_max_ps(_mm512_min_ps(x, max_val), min_val);

    const __m512 ln2     = _mm512_set1_ps(EXP_LN2);
    const __m512 inv_ln2 = _mm512_set1_ps(EXP_INV_LN2);

    // Range reduction: x = k*ln(2) + r, |r| < ln(2)/2
    __m512 k = _mm512_roundscale_ps(_mm512_mul_ps(x, inv_ln2), _MM_FROUND_TO_NEAREST_INT);
    __m512 r = _mm512_fnmadd_ps(k, ln2, x);

    // Polynomial approximation for exp(r)
    __m512 c6 = _mm512_set1_ps(EXP_C6);
    __m512 c5 = _mm512_set1_ps(EXP_C5);
    __m512 c4 = _mm512_set1_ps(EXP_C4);
    __m512 c3 = _mm512_set1_ps(EXP_C3);
    __m512 c2 = _mm512_set1_ps(EXP_C2);
    __m512 c1 = _mm512_set1_ps(EXP_C1);
    __m512 c0 = _mm512_set1_ps(EXP_C0);

    __m512 result = _mm512_fmadd_ps(c6, r, c5);
    result        = _mm512_fmadd_ps(result, r, c4);
    result        = _mm512_fmadd_ps(result, r, c3);
    result        = _mm512_fmadd_ps(result, r, c2);
    result        = _mm512_fmadd_ps(result, r, c1);
    result        = _mm512_fmadd_ps(result, r, c0);

    // Scale by 2^k
    __m512i ki   = _mm512_cvtps_epi32(k);
    ki           = _mm512_add_epi32(ki, _mm512_set1_epi32(127));
    ki           = _mm512_slli_epi32(ki, 23);
    __m512 scale = _mm512_castsi512_ps(ki);

    return _mm512_mul_ps(result, scale);
}

static inline __m512 log_poly_avx512(__m512 x) {
    // Extract exponent and mantissa
    __m512i xi        = _mm512_castps_si512(x);
    __m512i exp_mask  = _mm512_set1_epi32(0x7F800000);
    __m512i mant_mask = _mm512_set1_epi32(0x007FFFFF);

    __m512i exp_bits  = _mm512_and_si512(xi, exp_mask);
    __m512i mant_bits = _mm512_and_si512(xi, mant_mask);

    // e = exponent - 127
    __m512i e_int = _mm512_srli_epi32(exp_bits, 23);
    __m512 e      = _mm512_cvtepi32_ps(_mm512_sub_epi32(e_int, _mm512_set1_epi32(127)));

    // m = 1.mantissa (normalized to [1, 2))
    __m512i m_bits = _mm512_or_si512(mant_bits, _mm512_set1_epi32(0x3F800000));
    __m512 m       = _mm512_castsi512_ps(m_bits);

    // log(x) = e*ln(2) + log(m), log(m) using polynomial
    __m512 t = _mm512_sub_ps(m, _mm512_set1_ps(1.0f));

    // Polynomial for log(1+t) where t in [0, 1)
    __m512 c7 = _mm512_set1_ps(-0.125f);
    __m512 c6 = _mm512_set1_ps(0.142857142857f);
    __m512 c5 = _mm512_set1_ps(-0.166666666667f);
    __m512 c4 = _mm512_set1_ps(0.2f);
    __m512 c3 = _mm512_set1_ps(-0.25f);
    __m512 c2 = _mm512_set1_ps(0.333333333333f);
    __m512 c1 = _mm512_set1_ps(-0.5f);
    __m512 c0 = _mm512_set1_ps(1.0f);

    __m512 log_m = _mm512_fmadd_ps(c7, t, c6);
    log_m        = _mm512_fmadd_ps(log_m, t, c5);
    log_m        = _mm512_fmadd_ps(log_m, t, c4);
    log_m        = _mm512_fmadd_ps(log_m, t, c3);
    log_m        = _mm512_fmadd_ps(log_m, t, c2);
    log_m        = _mm512_fmadd_ps(log_m, t, c1);
    log_m        = _mm512_fmadd_ps(log_m, t, c0);
    log_m        = _mm512_mul_ps(log_m, t);

    return _mm512_fmadd_ps(e, _mm512_set1_ps(LOG_LN2), log_m);
}

static void simd_exp_f32_avx512(const float* in, float* out, size_t n) {
    const CMLSimdCaps* caps = cml_get_simd_caps();
    size_t i                = 0;

    if (caps->has_sleef && sleef_expf16) {
        for (; i + 16 <= n; i += 16) {
            __m512 v = _mm512_loadu_ps(&in[i]);
            _mm512_storeu_ps(&out[i], sleef_expf16(v));
        }
    } else {
        for (; i + 16 <= n; i += 16) {
            __m512 v = _mm512_loadu_ps(&in[i]);
            _mm512_storeu_ps(&out[i], exp_poly_avx512(v));
        }
    }
    for (; i < n; i++)
        out[i] = expf(in[i]);
}

static void simd_log_f32_avx512(const float* in, float* out, size_t n) {
    const CMLSimdCaps* caps = cml_get_simd_caps();
    size_t i                = 0;

    if (caps->has_sleef && sleef_logf16) {
        for (; i + 16 <= n; i += 16) {
            __m512 v = _mm512_loadu_ps(&in[i]);
            _mm512_storeu_ps(&out[i], sleef_logf16(v));
        }
    } else {
        for (; i + 16 <= n; i += 16) {
            __m512 v = _mm512_loadu_ps(&in[i]);
            _mm512_storeu_ps(&out[i], log_poly_avx512(v));
        }
    }
    for (; i < n; i++)
        out[i] = logf(in[i] + 1e-8f);
}

static void simd_sqrt_f32_avx512(const float* in, float* out, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(&in[i]);
        _mm512_storeu_ps(&out[i], _mm512_sqrt_ps(v));
    }
    for (; i < n; i++)
        out[i] = sqrtf(in[i]);
}

static void simd_rsqrt_f32_avx512(const float* in, float* out, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v     = _mm512_loadu_ps(&in[i]);
        __m512 rsqrt = _mm512_rsqrt14_ps(v);
        // Newton-Raphson refinement
        __m512 half  = _mm512_set1_ps(0.5f);
        __m512 three = _mm512_set1_ps(3.0f);
        rsqrt        = _mm512_mul_ps(_mm512_mul_ps(half, rsqrt),
                                     _mm512_fnmadd_ps(v, _mm512_mul_ps(rsqrt, rsqrt), three));
        _mm512_storeu_ps(&out[i], rsqrt);
    }
    for (; i < n; i++)
        out[i] = 1.0f / sqrtf(in[i]);
}

static void simd_recip_f32_avx512(const float* in, float* out, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v   = _mm512_loadu_ps(&in[i]);
        __m512 rcp = _mm512_rcp14_ps(v);
        // Newton-Raphson refinement
        __m512 two = _mm512_set1_ps(2.0f);
        rcp        = _mm512_mul_ps(rcp, _mm512_fnmadd_ps(v, rcp, two));
        _mm512_storeu_ps(&out[i], rcp);
    }
    for (; i < n; i++)
        out[i] = 1.0f / in[i];
}

static void simd_abs_f32_avx512(const float* in, float* out, size_t n) {
    size_t i     = 0;
    __m512i mask = _mm512_set1_epi32(0x7FFFFFFF);
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(&in[i]);
        _mm512_storeu_ps(&out[i],
                         _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(v), mask)));
    }
    for (; i < n; i++)
        out[i] = fabsf(in[i]);
}

static void simd_neg_f32_avx512(const float* in, float* out, size_t n) {
    size_t i    = 0;
    __m512 zero = _mm512_setzero_ps();
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(&in[i]);
        _mm512_storeu_ps(&out[i], _mm512_sub_ps(zero, v));
    }
    for (; i < n; i++)
        out[i] = -in[i];
}

static void simd_sin_f32_avx512(const float* in, float* out, size_t n) {
    const CMLSimdCaps* caps = cml_get_simd_caps();
    size_t i                = 0;
    if (caps->has_sleef && sleef_sinf16) {
        for (; i + 16 <= n; i += 16) {
            __m512 v = _mm512_loadu_ps(&in[i]);
            _mm512_storeu_ps(&out[i], sleef_sinf16(v));
        }
    }
    for (; i < n; i++)
        out[i] = sinf(in[i]);
}

static void simd_cos_f32_avx512(const float* in, float* out, size_t n) {
    const CMLSimdCaps* caps = cml_get_simd_caps();
    size_t i                = 0;
    if (caps->has_sleef && sleef_cosf16) {
        for (; i + 16 <= n; i += 16) {
            __m512 v = _mm512_loadu_ps(&in[i]);
            _mm512_storeu_ps(&out[i], sleef_cosf16(v));
        }
    }
    for (; i < n; i++)
        out[i] = cosf(in[i]);
}

static void simd_tan_f32_avx512(const float* in, float* out, size_t n) {
    const CMLSimdCaps* caps = cml_get_simd_caps();
    size_t i                = 0;
    if (caps->has_sleef && sleef_tanf16) {
        for (; i + 16 <= n; i += 16) {
            __m512 v = _mm512_loadu_ps(&in[i]);
            _mm512_storeu_ps(&out[i], sleef_tanf16(v));
        }
    }
    for (; i < n; i++)
        out[i] = tanf(in[i]);
}

static void simd_tanh_f32_avx512(const float* in, float* out, size_t n) {
    const CMLSimdCaps* caps = cml_get_simd_caps();
    size_t i                = 0;
    if (caps->has_sleef && sleef_tanhf16) {
        for (; i + 16 <= n; i += 16) {
            __m512 v = _mm512_loadu_ps(&in[i]);
            _mm512_storeu_ps(&out[i], sleef_tanhf16(v));
        }
    } else {
        // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        __m512 two = _mm512_set1_ps(2.0f);
        __m512 one = _mm512_set1_ps(1.0f);
        for (; i + 16 <= n; i += 16) {
            __m512 v     = _mm512_loadu_ps(&in[i]);
            __m512 exp2x = exp_poly_avx512(_mm512_mul_ps(two, v));
            __m512 num   = _mm512_sub_ps(exp2x, one);
            __m512 den   = _mm512_add_ps(exp2x, one);
            _mm512_storeu_ps(&out[i], _mm512_div_ps(num, den));
        }
    }
    for (; i < n; i++)
        out[i] = tanhf(in[i]);
}

static void simd_sigmoid_f32_avx512(const float* in, float* out, size_t n) {
    size_t i   = 0;
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 neg = _mm512_set1_ps(-1.0f);
    for (; i + 16 <= n; i += 16) {
        __m512 v         = _mm512_loadu_ps(&in[i]);
        __m512 exp_neg_x = exp_poly_avx512(_mm512_mul_ps(neg, v));
        __m512 result    = _mm512_div_ps(one, _mm512_add_ps(one, exp_neg_x));
        _mm512_storeu_ps(&out[i], result);
    }
    for (; i < n; i++)
        out[i] = 1.0f / (1.0f + expf(-in[i]));
}

static void simd_pow_f32_avx512(const float* a, const float* b, float* out, size_t n) {
    const CMLSimdCaps* caps = cml_get_simd_caps();
    size_t i                = 0;
    if (caps->has_sleef && sleef_powf16) {
        for (; i + 16 <= n; i += 16) {
            __m512 va = _mm512_loadu_ps(&a[i]);
            __m512 vb = _mm512_loadu_ps(&b[i]);
            _mm512_storeu_ps(&out[i], sleef_powf16(va, vb));
        }
    } else {
        // pow(a, b) = exp(b * log(a))
        for (; i + 16 <= n; i += 16) {
            __m512 va    = _mm512_loadu_ps(&a[i]);
            __m512 vb    = _mm512_loadu_ps(&b[i]);
            __m512 log_a = log_poly_avx512(va);
            _mm512_storeu_ps(&out[i], exp_poly_avx512(_mm512_mul_ps(vb, log_a)));
        }
    }
    for (; i < n; i++)
        out[i] = powf(a[i], b[i]);
}

static void simd_cmplt_f32_avx512(const float* a, const float* b, float* out, size_t n) {
    size_t i    = 0;
    __m512 one  = _mm512_set1_ps(1.0f);
    __m512 zero = _mm512_setzero_ps();
    for (; i + 16 <= n; i += 16) {
        __m512 va      = _mm512_loadu_ps(&a[i]);
        __m512 vb      = _mm512_loadu_ps(&b[i]);
        __mmask16 mask = _mm512_cmp_ps_mask(va, vb, _CMP_LT_OQ);
        _mm512_storeu_ps(&out[i], _mm512_mask_blend_ps(mask, zero, one));
    }
    for (; i < n; i++)
        out[i] = (a[i] < b[i]) ? 1.0f : 0.0f;
}

static void simd_cmpgt_f32_avx512(const float* a, const float* b, float* out, size_t n) {
    size_t i    = 0;
    __m512 one  = _mm512_set1_ps(1.0f);
    __m512 zero = _mm512_setzero_ps();
    for (; i + 16 <= n; i += 16) {
        __m512 va      = _mm512_loadu_ps(&a[i]);
        __m512 vb      = _mm512_loadu_ps(&b[i]);
        __mmask16 mask = _mm512_cmp_ps_mask(va, vb, _CMP_GT_OQ);
        _mm512_storeu_ps(&out[i], _mm512_mask_blend_ps(mask, zero, one));
    }
    for (; i < n; i++)
        out[i] = (a[i] > b[i]) ? 1.0f : 0.0f;
}

static void simd_min_f32_avx512(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        _mm512_storeu_ps(&out[i], _mm512_min_ps(va, vb));
    }
    for (; i < n; i++)
        out[i] = (a[i] < b[i]) ? a[i] : b[i];
}

static void simd_max_f32_avx512(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        _mm512_storeu_ps(&out[i], _mm512_max_ps(va, vb));
    }
    for (; i < n; i++)
        out[i] = (a[i] > b[i]) ? a[i] : b[i];
}

static void simd_where_f32_avx512(const float* cond, const float* a, const float* b, float* out,
                                  size_t n) {
    size_t i    = 0;
    __m512 zero = _mm512_setzero_ps();
    for (; i + 16 <= n; i += 16) {
        __m512 vc      = _mm512_loadu_ps(&cond[i]);
        __m512 va      = _mm512_loadu_ps(&a[i]);
        __m512 vb      = _mm512_loadu_ps(&b[i]);
        __mmask16 mask = _mm512_cmp_ps_mask(vc, zero, _CMP_NEQ_OQ);
        _mm512_storeu_ps(&out[i], _mm512_mask_blend_ps(mask, vb, va));
    }
    for (; i < n; i++)
        out[i] = (cond[i] != 0.0f) ? a[i] : b[i];
}

static void simd_add_f32_avx512(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        _mm512_storeu_ps(&out[i], _mm512_add_ps(va, vb));
    }
    for (; i < n; i++)
        out[i] = a[i] + b[i];
}

static void simd_sub_f32_avx512(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        _mm512_storeu_ps(&out[i], _mm512_sub_ps(va, vb));
    }
    for (; i < n; i++)
        out[i] = a[i] - b[i];
}

static void simd_mul_f32_avx512(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        _mm512_storeu_ps(&out[i], _mm512_mul_ps(va, vb));
    }
    for (; i < n; i++)
        out[i] = a[i] * b[i];
}

static void simd_div_f32_avx512(const float* a, const float* b, float* out, size_t n) {
    size_t i   = 0;
    __m512 eps = _mm512_set1_ps(1e-8f);
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        _mm512_storeu_ps(&out[i], _mm512_div_ps(va, _mm512_add_ps(vb, eps)));
    }
    for (; i < n; i++)
        out[i] = a[i] / (b[i] + 1e-8f);
}

#endif // CML_HAS_AVX512_COMPILE

#ifdef CML_HAS_AVX_COMPILE

static inline __m256 exp_poly_avx(__m256 x) {
    // Clamp input to prevent overflow/underflow
    // exp(88.7) ~ 3.4e38 (max float), exp(-87.3) ~ 1e-38 (min normal float)
    __m256 max_val = _mm256_set1_ps(88.0f);
    __m256 min_val = _mm256_set1_ps(-88.0f);
    x              = _mm256_max_ps(_mm256_min_ps(x, max_val), min_val);

    const __m256 ln2     = _mm256_set1_ps(EXP_LN2);
    const __m256 inv_ln2 = _mm256_set1_ps(EXP_INV_LN2);

    __m256 k = _mm256_round_ps(_mm256_mul_ps(x, inv_ln2), _MM_FROUND_TO_NEAREST_INT);
    __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(k, ln2));

    __m256 c6 = _mm256_set1_ps(EXP_C6);
    __m256 c5 = _mm256_set1_ps(EXP_C5);
    __m256 c4 = _mm256_set1_ps(EXP_C4);
    __m256 c3 = _mm256_set1_ps(EXP_C3);
    __m256 c2 = _mm256_set1_ps(EXP_C2);
    __m256 c1 = _mm256_set1_ps(EXP_C1);
    __m256 c0 = _mm256_set1_ps(EXP_C0);

#ifdef CML_HAS_AVX2_COMPILE
    __m256 result = _mm256_fmadd_ps(c6, r, c5);
    result        = _mm256_fmadd_ps(result, r, c4);
    result        = _mm256_fmadd_ps(result, r, c3);
    result        = _mm256_fmadd_ps(result, r, c2);
    result        = _mm256_fmadd_ps(result, r, c1);
    result        = _mm256_fmadd_ps(result, r, c0);
#else
    __m256 result = _mm256_add_ps(_mm256_mul_ps(c6, r), c5);
    result        = _mm256_add_ps(_mm256_mul_ps(result, r), c4);
    result        = _mm256_add_ps(_mm256_mul_ps(result, r), c3);
    result        = _mm256_add_ps(_mm256_mul_ps(result, r), c2);
    result        = _mm256_add_ps(_mm256_mul_ps(result, r), c1);
    result        = _mm256_add_ps(_mm256_mul_ps(result, r), c0);
#endif

    __m256i ki   = _mm256_cvtps_epi32(k);
    ki           = _mm256_add_epi32(ki, _mm256_set1_epi32(127));
    ki           = _mm256_slli_epi32(ki, 23);
    __m256 scale = _mm256_castsi256_ps(ki);

    return _mm256_mul_ps(result, scale);
}

static inline __m256 log_poly_avx(__m256 x) {
    __m256i xi        = _mm256_castps_si256(x);
    __m256i exp_mask  = _mm256_set1_epi32(0x7F800000);
    __m256i mant_mask = _mm256_set1_epi32(0x007FFFFF);

    __m256i exp_bits  = _mm256_and_si256(xi, exp_mask);
    __m256i mant_bits = _mm256_and_si256(xi, mant_mask);

    __m256i e_int = _mm256_srli_epi32(exp_bits, 23);
    __m256 e      = _mm256_cvtepi32_ps(_mm256_sub_epi32(e_int, _mm256_set1_epi32(127)));

    __m256i m_bits = _mm256_or_si256(mant_bits, _mm256_set1_epi32(0x3F800000));
    __m256 m       = _mm256_castsi256_ps(m_bits);

    __m256 t = _mm256_sub_ps(m, _mm256_set1_ps(1.0f));

    __m256 c7 = _mm256_set1_ps(-0.125f);
    __m256 c6 = _mm256_set1_ps(0.142857142857f);
    __m256 c5 = _mm256_set1_ps(-0.166666666667f);
    __m256 c4 = _mm256_set1_ps(0.2f);
    __m256 c3 = _mm256_set1_ps(-0.25f);
    __m256 c2 = _mm256_set1_ps(0.333333333333f);
    __m256 c1 = _mm256_set1_ps(-0.5f);
    __m256 c0 = _mm256_set1_ps(1.0f);

#ifdef CML_HAS_AVX2_COMPILE
    __m256 log_m = _mm256_fmadd_ps(c7, t, c6);
    log_m        = _mm256_fmadd_ps(log_m, t, c5);
    log_m        = _mm256_fmadd_ps(log_m, t, c4);
    log_m        = _mm256_fmadd_ps(log_m, t, c3);
    log_m        = _mm256_fmadd_ps(log_m, t, c2);
    log_m        = _mm256_fmadd_ps(log_m, t, c1);
    log_m        = _mm256_fmadd_ps(log_m, t, c0);
#else
    __m256 log_m = _mm256_add_ps(_mm256_mul_ps(c7, t), c6);
    log_m        = _mm256_add_ps(_mm256_mul_ps(log_m, t), c5);
    log_m        = _mm256_add_ps(_mm256_mul_ps(log_m, t), c4);
    log_m        = _mm256_add_ps(_mm256_mul_ps(log_m, t), c3);
    log_m        = _mm256_add_ps(_mm256_mul_ps(log_m, t), c2);
    log_m        = _mm256_add_ps(_mm256_mul_ps(log_m, t), c1);
    log_m        = _mm256_add_ps(_mm256_mul_ps(log_m, t), c0);
#endif
    log_m = _mm256_mul_ps(log_m, t);

#ifdef CML_HAS_AVX2_COMPILE
    return _mm256_fmadd_ps(e, _mm256_set1_ps(LOG_LN2), log_m);
#else
    return _mm256_add_ps(_mm256_mul_ps(e, _mm256_set1_ps(LOG_LN2)), log_m);
#endif
}

static void simd_exp_f32_avx(const float* in, float* out, size_t n) {
    const CMLSimdCaps* caps = cml_get_simd_caps();
    size_t i                = 0;

    if (caps->has_sleef && sleef_expf8) {
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(&in[i]);
            _mm256_storeu_ps(&out[i], sleef_expf8(v));
        }
    } else {
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(&in[i]);
            _mm256_storeu_ps(&out[i], exp_poly_avx(v));
        }
    }
    for (; i < n; i++)
        out[i] = expf(in[i]);
}

static void simd_log_f32_avx(const float* in, float* out, size_t n) {
    const CMLSimdCaps* caps = cml_get_simd_caps();
    size_t i                = 0;

    if (caps->has_sleef && sleef_logf8) {
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(&in[i]);
            _mm256_storeu_ps(&out[i], sleef_logf8(v));
        }
    } else {
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(&in[i]);
            _mm256_storeu_ps(&out[i], log_poly_avx(v));
        }
    }
    for (; i < n; i++)
        out[i] = logf(in[i] + 1e-8f);
}

static void simd_sqrt_f32_avx(const float* in, float* out, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(&in[i]);
        _mm256_storeu_ps(&out[i], _mm256_sqrt_ps(v));
    }
    for (; i < n; i++)
        out[i] = sqrtf(in[i]);
}

static void simd_rsqrt_f32_avx(const float* in, float* out, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v     = _mm256_loadu_ps(&in[i]);
        __m256 rsqrt = _mm256_rsqrt_ps(v);
        // Newton-Raphson
        __m256 half  = _mm256_set1_ps(0.5f);
        __m256 three = _mm256_set1_ps(3.0f);
#ifdef CML_HAS_AVX2_COMPILE
        rsqrt = _mm256_mul_ps(_mm256_mul_ps(half, rsqrt),
                              _mm256_fnmadd_ps(v, _mm256_mul_ps(rsqrt, rsqrt), three));
#else
        rsqrt = _mm256_mul_ps(_mm256_mul_ps(half, rsqrt),
                              _mm256_sub_ps(three, _mm256_mul_ps(v, _mm256_mul_ps(rsqrt, rsqrt))));
#endif
        _mm256_storeu_ps(&out[i], rsqrt);
    }
    for (; i < n; i++)
        out[i] = 1.0f / sqrtf(in[i]);
}

static void simd_recip_f32_avx(const float* in, float* out, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v   = _mm256_loadu_ps(&in[i]);
        __m256 rcp = _mm256_rcp_ps(v);
        // Newton-Raphson
        __m256 two = _mm256_set1_ps(2.0f);
#ifdef CML_HAS_AVX2_COMPILE
        rcp = _mm256_mul_ps(rcp, _mm256_fnmadd_ps(v, rcp, two));
#else
        rcp = _mm256_mul_ps(rcp, _mm256_sub_ps(two, _mm256_mul_ps(v, rcp)));
#endif
        _mm256_storeu_ps(&out[i], rcp);
    }
    for (; i < n; i++)
        out[i] = 1.0f / in[i];
}

static void simd_abs_f32_avx(const float* in, float* out, size_t n) {
    size_t i     = 0;
    __m256i mask = _mm256_set1_epi32(0x7FFFFFFF);
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(&in[i]);
        _mm256_storeu_ps(&out[i], _mm256_and_ps(v, _mm256_castsi256_ps(mask)));
    }
    for (; i < n; i++)
        out[i] = fabsf(in[i]);
}

static void simd_neg_f32_avx(const float* in, float* out, size_t n) {
    size_t i         = 0;
    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(&in[i]);
        _mm256_storeu_ps(&out[i], _mm256_xor_ps(v, sign_mask));
    }
    for (; i < n; i++)
        out[i] = -in[i];
}

static void simd_sin_f32_avx(const float* in, float* out, size_t n) {
    const CMLSimdCaps* caps = cml_get_simd_caps();
    size_t i                = 0;
    if (caps->has_sleef && sleef_sinf8) {
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(&in[i]);
            _mm256_storeu_ps(&out[i], sleef_sinf8(v));
        }
    }
    for (; i < n; i++)
        out[i] = sinf(in[i]);
}

static void simd_cos_f32_avx(const float* in, float* out, size_t n) {
    const CMLSimdCaps* caps = cml_get_simd_caps();
    size_t i                = 0;
    if (caps->has_sleef && sleef_cosf8) {
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(&in[i]);
            _mm256_storeu_ps(&out[i], sleef_cosf8(v));
        }
    }
    for (; i < n; i++)
        out[i] = cosf(in[i]);
}

static void simd_tan_f32_avx(const float* in, float* out, size_t n) {
    const CMLSimdCaps* caps = cml_get_simd_caps();
    size_t i                = 0;
    if (caps->has_sleef && sleef_tanf8) {
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(&in[i]);
            _mm256_storeu_ps(&out[i], sleef_tanf8(v));
        }
    }
    for (; i < n; i++)
        out[i] = tanf(in[i]);
}

static void simd_tanh_f32_avx(const float* in, float* out, size_t n) {
    const CMLSimdCaps* caps = cml_get_simd_caps();
    size_t i                = 0;
    if (caps->has_sleef && sleef_tanhf8) {
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(&in[i]);
            _mm256_storeu_ps(&out[i], sleef_tanhf8(v));
        }
    } else {
        __m256 two = _mm256_set1_ps(2.0f);
        __m256 one = _mm256_set1_ps(1.0f);
        for (; i + 8 <= n; i += 8) {
            __m256 v     = _mm256_loadu_ps(&in[i]);
            __m256 exp2x = exp_poly_avx(_mm256_mul_ps(two, v));
            __m256 num   = _mm256_sub_ps(exp2x, one);
            __m256 den   = _mm256_add_ps(exp2x, one);
            _mm256_storeu_ps(&out[i], _mm256_div_ps(num, den));
        }
    }
    for (; i < n; i++)
        out[i] = tanhf(in[i]);
}

static void simd_sigmoid_f32_avx(const float* in, float* out, size_t n) {
    size_t i   = 0;
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 neg = _mm256_set1_ps(-1.0f);
    for (; i + 8 <= n; i += 8) {
        __m256 v         = _mm256_loadu_ps(&in[i]);
        __m256 exp_neg_x = exp_poly_avx(_mm256_mul_ps(neg, v));
        __m256 result    = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg_x));
        _mm256_storeu_ps(&out[i], result);
    }
    for (; i < n; i++)
        out[i] = 1.0f / (1.0f + expf(-in[i]));
}

static void simd_pow_f32_avx(const float* a, const float* b, float* out, size_t n) {
    const CMLSimdCaps* caps = cml_get_simd_caps();
    size_t i                = 0;
    if (caps->has_sleef && sleef_powf8) {
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);
            _mm256_storeu_ps(&out[i], sleef_powf8(va, vb));
        }
    } else {
        for (; i + 8 <= n; i += 8) {
            __m256 va    = _mm256_loadu_ps(&a[i]);
            __m256 vb    = _mm256_loadu_ps(&b[i]);
            __m256 log_a = log_poly_avx(va);
            _mm256_storeu_ps(&out[i], exp_poly_avx(_mm256_mul_ps(vb, log_a)));
        }
    }
    for (; i < n; i++)
        out[i] = powf(a[i], b[i]);
}

static void simd_cmplt_f32_avx(const float* a, const float* b, float* out, size_t n) {
    size_t i   = 0;
    __m256 one = _mm256_set1_ps(1.0f);
    for (; i + 8 <= n; i += 8) {
        __m256 va  = _mm256_loadu_ps(&a[i]);
        __m256 vb  = _mm256_loadu_ps(&b[i]);
        __m256 cmp = _mm256_cmp_ps(va, vb, _CMP_LT_OQ);
        _mm256_storeu_ps(&out[i], _mm256_and_ps(cmp, one));
    }
    for (; i < n; i++)
        out[i] = (a[i] < b[i]) ? 1.0f : 0.0f;
}

static void simd_cmpgt_f32_avx(const float* a, const float* b, float* out, size_t n) {
    size_t i   = 0;
    __m256 one = _mm256_set1_ps(1.0f);
    for (; i + 8 <= n; i += 8) {
        __m256 va  = _mm256_loadu_ps(&a[i]);
        __m256 vb  = _mm256_loadu_ps(&b[i]);
        __m256 cmp = _mm256_cmp_ps(va, vb, _CMP_GT_OQ);
        _mm256_storeu_ps(&out[i], _mm256_and_ps(cmp, one));
    }
    for (; i < n; i++)
        out[i] = (a[i] > b[i]) ? 1.0f : 0.0f;
}

static void simd_min_f32_avx(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        _mm256_storeu_ps(&out[i], _mm256_min_ps(va, vb));
    }
    for (; i < n; i++)
        out[i] = (a[i] < b[i]) ? a[i] : b[i];
}

static void simd_max_f32_avx(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        _mm256_storeu_ps(&out[i], _mm256_max_ps(va, vb));
    }
    for (; i < n; i++)
        out[i] = (a[i] > b[i]) ? a[i] : b[i];
}

static void simd_where_f32_avx(const float* cond, const float* a, const float* b, float* out,
                               size_t n) {
    size_t i    = 0;
    __m256 zero = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 vc   = _mm256_loadu_ps(&cond[i]);
        __m256 va   = _mm256_loadu_ps(&a[i]);
        __m256 vb   = _mm256_loadu_ps(&b[i]);
        __m256 mask = _mm256_cmp_ps(vc, zero, _CMP_NEQ_OQ);
        _mm256_storeu_ps(&out[i], _mm256_blendv_ps(vb, va, mask));
    }
    for (; i < n; i++)
        out[i] = (cond[i] != 0.0f) ? a[i] : b[i];
}

static void simd_add_f32_avx(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        _mm256_storeu_ps(&out[i], _mm256_add_ps(va, vb));
    }
    for (; i < n; i++)
        out[i] = a[i] + b[i];
}

static void simd_sub_f32_avx(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        _mm256_storeu_ps(&out[i], _mm256_sub_ps(va, vb));
    }
    for (; i < n; i++)
        out[i] = a[i] - b[i];
}

static void simd_mul_f32_avx(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        _mm256_storeu_ps(&out[i], _mm256_mul_ps(va, vb));
    }
    for (; i < n; i++)
        out[i] = a[i] * b[i];
}

static void simd_div_f32_avx(const float* a, const float* b, float* out, size_t n) {
    size_t i   = 0;
    __m256 eps = _mm256_set1_ps(1e-8f);
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        _mm256_storeu_ps(&out[i], _mm256_div_ps(va, _mm256_add_ps(vb, eps)));
    }
    for (; i < n; i++)
        out[i] = a[i] / (b[i] + 1e-8f);
}

#endif // CML_HAS_AVX_COMPILE

#ifdef CML_HAS_SSE_COMPILE

static void simd_exp_f32_sse(const float* in, float* out, size_t n) {
    // SSE polynomial exp
    size_t i = 0;
    for (; i < n; i++)
        out[i] = expf(in[i]);
}

static void simd_log_f32_sse(const float* in, float* out, size_t n) {
    size_t i = 0;
    for (; i < n; i++)
        out[i] = logf(in[i] + 1e-8f);
}

static void simd_sqrt_f32_sse(const float* in, float* out, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 v = _mm_loadu_ps(&in[i]);
        _mm_storeu_ps(&out[i], _mm_sqrt_ps(v));
    }
    for (; i < n; i++)
        out[i] = sqrtf(in[i]);
}

static void simd_rsqrt_f32_sse(const float* in, float* out, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 v     = _mm_loadu_ps(&in[i]);
        __m128 rsqrt = _mm_rsqrt_ps(v);
        __m128 half  = _mm_set1_ps(0.5f);
        __m128 three = _mm_set1_ps(3.0f);
        rsqrt        = _mm_mul_ps(_mm_mul_ps(half, rsqrt),
                                  _mm_sub_ps(three, _mm_mul_ps(v, _mm_mul_ps(rsqrt, rsqrt))));
        _mm_storeu_ps(&out[i], rsqrt);
    }
    for (; i < n; i++)
        out[i] = 1.0f / sqrtf(in[i]);
}

static void simd_recip_f32_sse(const float* in, float* out, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 v   = _mm_loadu_ps(&in[i]);
        __m128 rcp = _mm_rcp_ps(v);
        __m128 two = _mm_set1_ps(2.0f);
        rcp        = _mm_mul_ps(rcp, _mm_sub_ps(two, _mm_mul_ps(v, rcp)));
        _mm_storeu_ps(&out[i], rcp);
    }
    for (; i < n; i++)
        out[i] = 1.0f / in[i];
}

static void simd_abs_f32_sse(const float* in, float* out, size_t n) {
    size_t i     = 0;
    __m128i mask = _mm_set1_epi32(0x7FFFFFFF);
    for (; i + 4 <= n; i += 4) {
        __m128 v = _mm_loadu_ps(&in[i]);
        _mm_storeu_ps(&out[i], _mm_and_ps(v, _mm_castsi128_ps(mask)));
    }
    for (; i < n; i++)
        out[i] = fabsf(in[i]);
}

static void simd_neg_f32_sse(const float* in, float* out, size_t n) {
    size_t i         = 0;
    __m128 sign_mask = _mm_set1_ps(-0.0f);
    for (; i + 4 <= n; i += 4) {
        __m128 v = _mm_loadu_ps(&in[i]);
        _mm_storeu_ps(&out[i], _mm_xor_ps(v, sign_mask));
    }
    for (; i < n; i++)
        out[i] = -in[i];
}

static void simd_add_f32_sse(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        _mm_storeu_ps(&out[i], _mm_add_ps(va, vb));
    }
    for (; i < n; i++)
        out[i] = a[i] + b[i];
}

static void simd_sub_f32_sse(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        _mm_storeu_ps(&out[i], _mm_sub_ps(va, vb));
    }
    for (; i < n; i++)
        out[i] = a[i] - b[i];
}

static void simd_mul_f32_sse(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        _mm_storeu_ps(&out[i], _mm_mul_ps(va, vb));
    }
    for (; i < n; i++)
        out[i] = a[i] * b[i];
}

static void simd_div_f32_sse(const float* a, const float* b, float* out, size_t n) {
    size_t i   = 0;
    __m128 eps = _mm_set1_ps(1e-8f);
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        _mm_storeu_ps(&out[i], _mm_div_ps(va, _mm_add_ps(vb, eps)));
    }
    for (; i < n; i++)
        out[i] = a[i] / (b[i] + 1e-8f);
}

static void simd_min_f32_sse(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        _mm_storeu_ps(&out[i], _mm_min_ps(va, vb));
    }
    for (; i < n; i++)
        out[i] = (a[i] < b[i]) ? a[i] : b[i];
}

static void simd_max_f32_sse(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        _mm_storeu_ps(&out[i], _mm_max_ps(va, vb));
    }
    for (; i < n; i++)
        out[i] = (a[i] > b[i]) ? a[i] : b[i];
}

#endif // CML_HAS_SSE_COMPILE

#ifdef CML_HAS_NEON_COMPILE

static void simd_exp_f32_neon(const float* in, float* out, size_t n) {
    size_t i = 0;
    for (; i < n; i++)
        out[i] = expf(in[i]);
}

static void simd_log_f32_neon(const float* in, float* out, size_t n) {
    size_t i = 0;
    for (; i < n; i++)
        out[i] = logf(in[i] + 1e-8f);
}

static void simd_sqrt_f32_neon(const float* in, float* out, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(&in[i]);
        vst1q_f32(&out[i], vsqrtq_f32(v));
    }
    for (; i < n; i++)
        out[i] = sqrtf(in[i]);
}

static void simd_rsqrt_f32_neon(const float* in, float* out, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v     = vld1q_f32(&in[i]);
        float32x4_t rsqrt = vrsqrteq_f32(v);
        rsqrt             = vmulq_f32(vrsqrtsq_f32(vmulq_f32(v, rsqrt), rsqrt), rsqrt);
        vst1q_f32(&out[i], rsqrt);
    }
    for (; i < n; i++)
        out[i] = 1.0f / sqrtf(in[i]);
}

static void simd_recip_f32_neon(const float* in, float* out, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v   = vld1q_f32(&in[i]);
        float32x4_t rcp = vrecpeq_f32(v);
        rcp             = vmulq_f32(vrecpsq_f32(v, rcp), rcp);
        vst1q_f32(&out[i], rcp);
    }
    for (; i < n; i++)
        out[i] = 1.0f / in[i];
}

static void simd_abs_f32_neon(const float* in, float* out, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(&in[i]);
        vst1q_f32(&out[i], vabsq_f32(v));
    }
    for (; i < n; i++)
        out[i] = fabsf(in[i]);
}

static void simd_neg_f32_neon(const float* in, float* out, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(&in[i]);
        vst1q_f32(&out[i], vnegq_f32(v));
    }
    for (; i < n; i++)
        out[i] = -in[i];
}

static void simd_add_f32_neon(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        vst1q_f32(&out[i], vaddq_f32(va, vb));
    }
    for (; i < n; i++)
        out[i] = a[i] + b[i];
}

static void simd_sub_f32_neon(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        vst1q_f32(&out[i], vsubq_f32(va, vb));
    }
    for (; i < n; i++)
        out[i] = a[i] - b[i];
}

static void simd_mul_f32_neon(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        vst1q_f32(&out[i], vmulq_f32(va, vb));
    }
    for (; i < n; i++)
        out[i] = a[i] * b[i];
}

static void simd_div_f32_neon(const float* a, const float* b, float* out, size_t n) {
    size_t i        = 0;
    float32x4_t eps = vdupq_n_f32(1e-8f);
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        vst1q_f32(&out[i], vdivq_f32(va, vaddq_f32(vb, eps)));
    }
    for (; i < n; i++)
        out[i] = a[i] / (b[i] + 1e-8f);
}

static void simd_min_f32_neon(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        vst1q_f32(&out[i], vminq_f32(va, vb));
    }
    for (; i < n; i++)
        out[i] = (a[i] < b[i]) ? a[i] : b[i];
}

static void simd_max_f32_neon(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        vst1q_f32(&out[i], vmaxq_f32(va, vb));
    }
    for (; i < n; i++)
        out[i] = (a[i] > b[i]) ? a[i] : b[i];
}

#endif // CML_HAS_NEON_COMPILE

static void simd_exp_f32_scalar(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; i++)
        out[i] = expf(in[i]);
}

static void simd_log_f32_scalar(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; i++)
        out[i] = logf(in[i] + 1e-8f);
}

static void simd_sqrt_f32_scalar(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; i++)
        out[i] = sqrtf(in[i]);
}

static void simd_rsqrt_f32_scalar(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; i++)
        out[i] = 1.0f / sqrtf(in[i]);
}

static void simd_recip_f32_scalar(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; i++)
        out[i] = 1.0f / in[i];
}

static void simd_abs_f32_scalar(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; i++)
        out[i] = fabsf(in[i]);
}

static void simd_neg_f32_scalar(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; i++)
        out[i] = -in[i];
}

static void simd_sin_f32_scalar(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; i++)
        out[i] = sinf(in[i]);
}

static void simd_cos_f32_scalar(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; i++)
        out[i] = cosf(in[i]);
}

static void simd_tan_f32_scalar(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; i++)
        out[i] = tanf(in[i]);
}

static void simd_tanh_f32_scalar(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; i++)
        out[i] = tanhf(in[i]);
}

static void simd_sigmoid_f32_scalar(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; i++)
        out[i] = 1.0f / (1.0f + expf(-in[i]));
}

static void simd_pow_f32_scalar(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; i++)
        out[i] = powf(a[i], b[i]);
}

static void simd_cmplt_f32_scalar(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; i++)
        out[i] = (a[i] < b[i]) ? 1.0f : 0.0f;
}

static void simd_cmpgt_f32_scalar(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; i++)
        out[i] = (a[i] > b[i]) ? 1.0f : 0.0f;
}

static void simd_min_f32_scalar(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; i++)
        out[i] = (a[i] < b[i]) ? a[i] : b[i];
}

static void simd_max_f32_scalar(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; i++)
        out[i] = (a[i] > b[i]) ? a[i] : b[i];
}

static void simd_where_f32_scalar(const float* cond, const float* a, const float* b, float* out,
                                  size_t n) {
    for (size_t i = 0; i < n; i++)
        out[i] = (cond[i] != 0.0f) ? a[i] : b[i];
}

static void simd_add_f32_scalar(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; i++)
        out[i] = a[i] + b[i];
}

static void simd_sub_f32_scalar(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; i++)
        out[i] = a[i] - b[i];
}

static void simd_mul_f32_scalar(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; i++)
        out[i] = a[i] * b[i];
}

static void simd_div_f32_scalar(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; i++)
        out[i] = a[i] / (b[i] + 1e-8f);
}

void simd_exp_f32(const float* in, float* out, size_t n) {
    if (!in || !out || n == 0)
        return;
    const CMLSimdCaps* caps = cml_get_simd_caps();

#ifdef CML_HAS_AVX512_COMPILE
    if (caps->has_avx512) {
        simd_exp_f32_avx512(in, out, n);
        return;
    }
#endif
#ifdef CML_HAS_AVX_COMPILE
    if (caps->has_avx) {
        simd_exp_f32_avx(in, out, n);
        return;
    }
#endif
#ifdef CML_HAS_SSE_COMPILE
    if (caps->has_sse) {
        simd_exp_f32_sse(in, out, n);
        return;
    }
#endif
#ifdef CML_HAS_NEON_COMPILE
    if (caps->has_neon) {
        simd_exp_f32_neon(in, out, n);
        return;
    }
#endif
    simd_exp_f32_scalar(in, out, n);
}

void simd_log_f32(const float* in, float* out, size_t n) {
    if (!in || !out || n == 0)
        return;
    const CMLSimdCaps* caps = cml_get_simd_caps();

#ifdef CML_HAS_AVX512_COMPILE
    if (caps->has_avx512) {
        simd_log_f32_avx512(in, out, n);
        return;
    }
#endif
#ifdef CML_HAS_AVX_COMPILE
    if (caps->has_avx) {
        simd_log_f32_avx(in, out, n);
        return;
    }
#endif
#ifdef CML_HAS_SSE_COMPILE
    if (caps->has_sse) {
        simd_log_f32_sse(in, out, n);
        return;
    }
#endif
#ifdef CML_HAS_NEON_COMPILE
    if (caps->has_neon) {
        simd_log_f32_neon(in, out, n);
        return;
    }
#endif
    simd_log_f32_scalar(in, out, n);
}

void simd_sqrt_f32(const float* in, float* out, size_t n) {
    if (!in || !out || n == 0)
        return;
    const CMLSimdCaps* caps = cml_get_simd_caps();

#ifdef CML_HAS_AVX512_COMPILE
    if (caps->has_avx512) {
        simd_sqrt_f32_avx512(in, out, n);
        return;
    }
#endif
#ifdef CML_HAS_AVX_COMPILE
    if (caps->has_avx) {
        simd_sqrt_f32_avx(in, out, n);
        return;
    }
#endif
#ifdef CML_HAS_SSE_COMPILE
    if (caps->has_sse) {
        simd_sqrt_f32_sse(in, out, n);
        return;
    }
#endif
#ifdef CML_HAS_NEON_COMPILE
    if (caps->has_neon) {
        simd_sqrt_f32_neon(in, out, n);
        return;
    }
#endif
    simd_sqrt_f32_scalar(in, out, n);
}

void simd_rsqrt_f32(const float* in, float* out, size_t n) {
    if (!in || !out || n == 0)
        return;
    const CMLSimdCaps* caps = cml_get_simd_caps();

#ifdef CML_HAS_AVX512_COMPILE
    if (caps->has_avx512) {
        simd_rsqrt_f32_avx512(in, out, n);
        return;
    }
#endif
#ifdef CML_HAS_AVX_COMPILE
    if (caps->has_avx) {
        simd_rsqrt_f32_avx(in, out, n);
        return;
    }
#endif
#ifdef CML_HAS_SSE_COMPILE
    if (caps->has_sse) {
        simd_rsqrt_f32_sse(in, out, n);
        return;
    }
#endif
#ifdef CML_HAS_NEON_COMPILE
    if (caps->has_neon) {
        simd_rsqrt_f32_neon(in, out, n);
        return;
    }
#endif
    simd_rsqrt_f32_scalar(in, out, n);
}

void simd_recip_f32(const float* in, float* out, size_t n) {
    if (!in || !out || n == 0)
        return;
    const CMLSimdCaps* caps = cml_get_simd_caps();

#ifdef CML_HAS_AVX512_COMPILE
    if (caps->has_avx512) {
        simd_recip_f32_avx512(in, out, n);
        return;
    }
#endif
#ifdef CML_HAS_AVX_COMPILE
    if (caps->has_avx) {
        simd_recip_f32_avx(in, out, n);
        return;
    }
#endif
#ifdef CML_HAS_SSE_COMPILE
    if (caps->has_sse) {
        simd_recip_f32_sse(in, out, n);
        return;
    }
#endif
#ifdef CML_HAS_NEON_COMPILE
    if (caps->has_neon) {
        simd_recip_f32_neon(in, out, n);
        return;
    }
#endif
    simd_recip_f32_scalar(in, out, n);
}

void simd_abs_f32(const float* in, float* out, size_t n) {
    if (!in || !out || n == 0)
        return;
    const CMLSimdCaps* caps = cml_get_simd_caps();

#ifdef CML_HAS_AVX512_COMPILE
    if (caps->has_avx512) {
        simd_abs_f32_avx512(in, out, n);
        return;
    }
#endif
#ifdef CML_HAS_AVX_COMPILE
    if (caps->has_avx) {
        simd_abs_f32_avx(in, out, n);
        return;
    }
#endif
#ifdef CML_HAS_SSE_COMPILE
    if (caps->has_sse) {
        simd_abs_f32_sse(in, out, n);
        return;
    }
#endif
#ifdef CML_HAS_NEON_COMPILE
    if (caps->has_neon) {
        simd_abs_f32_neon(in, out, n);
        return;
    }
#endif
    simd_abs_f32_scalar(in, out, n);
}

void simd_neg_f32(const float* in, float* out, size_t n) {
    if (!in || !out || n == 0)
        return;
    const CMLSimdCaps* caps = cml_get_simd_caps();

#ifdef CML_HAS_AVX512_COMPILE
    if (caps->has_avx512) {
        simd_neg_f32_avx512(in, out, n);
        return;
    }
#endif
#ifdef CML_HAS_AVX_COMPILE
    if (caps->has_avx) {
        simd_neg_f32_avx(in, out, n);
        return;
    }
#endif
#ifdef CML_HAS_SSE_COMPILE
    if (caps->has_sse) {
        simd_neg_f32_sse(in, out, n);
        return;
    }
#endif
#ifdef CML_HAS_NEON_COMPILE
    if (caps->has_neon) {
        simd_neg_f32_neon(in, out, n);
        return;
    }
#endif
    simd_neg_f32_scalar(in, out, n);
}

void simd_sin_f32(const float* in, float* out, size_t n) {
    if (!in || !out || n == 0)
        return;
    const CMLSimdCaps* caps = cml_get_simd_caps();
    (void)caps;

#ifdef CML_HAS_AVX512_COMPILE
    if (caps->has_avx512) {
        simd_sin_f32_avx512(in, out, n);
        return;
    }
#endif
#ifdef CML_HAS_AVX_COMPILE
    if (caps->has_avx) {
        simd_sin_f32_avx(in, out, n);
        return;
    }
#endif
    simd_sin_f32_scalar(in, out, n);
}

void simd_cos_f32(const float* in, float* out, size_t n) {
    if (!in || !out || n == 0)
        return;
    const CMLSimdCaps* caps = cml_get_simd_caps();
    (void)caps;

#ifdef CML_HAS_AVX512_COMPILE
    if (caps->has_avx512) {
        simd_cos_f32_avx512(in, out, n);
        return;
    }
#endif
#ifdef CML_HAS_AVX_COMPILE
    if (caps->has_avx) {
        simd_cos_f32_avx(in, out, n);
        return;
    }
#endif
    simd_cos_f32_scalar(in, out, n);
}

void simd_tan_f32(const float* in, float* out, size_t n) {
    if (!in || !out || n == 0)
        return;
    const CMLSimdCaps* caps = cml_get_simd_caps();
    (void)caps;

#ifdef CML_HAS_AVX512_COMPILE
    if (caps->has_avx512) {
        simd_tan_f32_avx512(in, out, n);
        return;
    }
#endif
#ifdef CML_HAS_AVX_COMPILE
    if (caps->has_avx) {
        simd_tan_f32_avx(in, out, n);
        return;
    }
#endif
    simd_tan_f32_scalar(in, out, n);
}

void simd_tanh_f32(const float* in, float* out, size_t n) {
    if (!in || !out || n == 0)
        return;
    const CMLSimdCaps* caps = cml_get_simd_caps();
    (void)caps;

#ifdef CML_HAS_AVX512_COMPILE
    if (caps->has_avx512) {
        simd_tanh_f32_avx512(in, out, n);
        return;
    }
#endif
#ifdef CML_HAS_AVX_COMPILE
    if (caps->has_avx) {
        simd_tanh_f32_avx(in, out, n);
        return;
    }
#endif
    simd_tanh_f32_scalar(in, out, n);
}

void simd_sigmoid_f32(const float* in, float* out, size_t n) {
    if (!in || !out || n == 0)
        return;
    const CMLSimdCaps* caps = cml_get_simd_caps();
    (void)caps;

#ifdef CML_HAS_AVX512_COMPILE
    if (caps->has_avx512) {
        simd_sigmoid_f32_avx512(in, out, n);
        return;
    }
#endif
#ifdef CML_HAS_AVX_COMPILE
    if (caps->has_avx) {
        simd_sigmoid_f32_avx(in, out, n);
        return;
    }
#endif
    simd_sigmoid_f32_scalar(in, out, n);
}

void simd_pow_f32(const float* a, const float* b, float* out, size_t n) {
    if (!a || !b || !out || n == 0)
        return;
    const CMLSimdCaps* caps = cml_get_simd_caps();
    (void)caps;

#ifdef CML_HAS_AVX512_COMPILE
    if (caps->has_avx512) {
        simd_pow_f32_avx512(a, b, out, n);
        return;
    }
#endif
#ifdef CML_HAS_AVX_COMPILE
    if (caps->has_avx) {
        simd_pow_f32_avx(a, b, out, n);
        return;
    }
#endif
    simd_pow_f32_scalar(a, b, out, n);
}

void simd_cmplt_f32(const float* a, const float* b, float* out, size_t n) {
    if (!a || !b || !out || n == 0)
        return;
    const CMLSimdCaps* caps = cml_get_simd_caps();
    (void)caps;

#ifdef CML_HAS_AVX512_COMPILE
    if (caps->has_avx512) {
        simd_cmplt_f32_avx512(a, b, out, n);
        return;
    }
#endif
#ifdef CML_HAS_AVX_COMPILE
    if (caps->has_avx) {
        simd_cmplt_f32_avx(a, b, out, n);
        return;
    }
#endif
    simd_cmplt_f32_scalar(a, b, out, n);
}

void simd_cmpgt_f32(const float* a, const float* b, float* out, size_t n) {
    if (!a || !b || !out || n == 0)
        return;
    const CMLSimdCaps* caps = cml_get_simd_caps();
    (void)caps;

#ifdef CML_HAS_AVX512_COMPILE
    if (caps->has_avx512) {
        simd_cmpgt_f32_avx512(a, b, out, n);
        return;
    }
#endif
#ifdef CML_HAS_AVX_COMPILE
    if (caps->has_avx) {
        simd_cmpgt_f32_avx(a, b, out, n);
        return;
    }
#endif
    simd_cmpgt_f32_scalar(a, b, out, n);
}

void simd_min_f32(const float* a, const float* b, float* out, size_t n) {
    if (!a || !b || !out || n == 0)
        return;
    const CMLSimdCaps* caps = cml_get_simd_caps();

#ifdef CML_HAS_AVX512_COMPILE
    if (caps->has_avx512) {
        simd_min_f32_avx512(a, b, out, n);
        return;
    }
#endif
#ifdef CML_HAS_AVX_COMPILE
    if (caps->has_avx) {
        simd_min_f32_avx(a, b, out, n);
        return;
    }
#endif
#ifdef CML_HAS_SSE_COMPILE
    if (caps->has_sse) {
        simd_min_f32_sse(a, b, out, n);
        return;
    }
#endif
#ifdef CML_HAS_NEON_COMPILE
    if (caps->has_neon) {
        simd_min_f32_neon(a, b, out, n);
        return;
    }
#endif
    simd_min_f32_scalar(a, b, out, n);
}

void simd_max_f32(const float* a, const float* b, float* out, size_t n) {
    if (!a || !b || !out || n == 0)
        return;
    const CMLSimdCaps* caps = cml_get_simd_caps();

#ifdef CML_HAS_AVX512_COMPILE
    if (caps->has_avx512) {
        simd_max_f32_avx512(a, b, out, n);
        return;
    }
#endif
#ifdef CML_HAS_AVX_COMPILE
    if (caps->has_avx) {
        simd_max_f32_avx(a, b, out, n);
        return;
    }
#endif
#ifdef CML_HAS_SSE_COMPILE
    if (caps->has_sse) {
        simd_max_f32_sse(a, b, out, n);
        return;
    }
#endif
#ifdef CML_HAS_NEON_COMPILE
    if (caps->has_neon) {
        simd_max_f32_neon(a, b, out, n);
        return;
    }
#endif
    simd_max_f32_scalar(a, b, out, n);
}

void simd_where_f32(const float* cond, const float* a, const float* b, float* out, size_t n) {
    if (!cond || !a || !b || !out || n == 0)
        return;
    const CMLSimdCaps* caps = cml_get_simd_caps();
    (void)caps;

#ifdef CML_HAS_AVX512_COMPILE
    if (caps->has_avx512) {
        simd_where_f32_avx512(cond, a, b, out, n);
        return;
    }
#endif
#ifdef CML_HAS_AVX_COMPILE
    if (caps->has_avx) {
        simd_where_f32_avx(cond, a, b, out, n);
        return;
    }
#endif
    simd_where_f32_scalar(cond, a, b, out, n);
}

void simd_add_f32(const float* a, const float* b, float* out, size_t n) {
    if (!a || !b || !out || n == 0)
        return;
    const CMLSimdCaps* caps = cml_get_simd_caps();

#ifdef CML_HAS_AVX512_COMPILE
    if (caps->has_avx512) {
        simd_add_f32_avx512(a, b, out, n);
        return;
    }
#endif
#ifdef CML_HAS_AVX_COMPILE
    if (caps->has_avx) {
        simd_add_f32_avx(a, b, out, n);
        return;
    }
#endif
#ifdef CML_HAS_SSE_COMPILE
    if (caps->has_sse) {
        simd_add_f32_sse(a, b, out, n);
        return;
    }
#endif
#ifdef CML_HAS_NEON_COMPILE
    if (caps->has_neon) {
        simd_add_f32_neon(a, b, out, n);
        return;
    }
#endif
    simd_add_f32_scalar(a, b, out, n);
}

void simd_sub_f32(const float* a, const float* b, float* out, size_t n) {
    if (!a || !b || !out || n == 0)
        return;
    const CMLSimdCaps* caps = cml_get_simd_caps();

#ifdef CML_HAS_AVX512_COMPILE
    if (caps->has_avx512) {
        simd_sub_f32_avx512(a, b, out, n);
        return;
    }
#endif
#ifdef CML_HAS_AVX_COMPILE
    if (caps->has_avx) {
        simd_sub_f32_avx(a, b, out, n);
        return;
    }
#endif
#ifdef CML_HAS_SSE_COMPILE
    if (caps->has_sse) {
        simd_sub_f32_sse(a, b, out, n);
        return;
    }
#endif
#ifdef CML_HAS_NEON_COMPILE
    if (caps->has_neon) {
        simd_sub_f32_neon(a, b, out, n);
        return;
    }
#endif
    simd_sub_f32_scalar(a, b, out, n);
}

void simd_mul_f32(const float* a, const float* b, float* out, size_t n) {
    if (!a || !b || !out || n == 0)
        return;
    const CMLSimdCaps* caps = cml_get_simd_caps();

#ifdef CML_HAS_AVX512_COMPILE
    if (caps->has_avx512) {
        simd_mul_f32_avx512(a, b, out, n);
        return;
    }
#endif
#ifdef CML_HAS_AVX_COMPILE
    if (caps->has_avx) {
        simd_mul_f32_avx(a, b, out, n);
        return;
    }
#endif
#ifdef CML_HAS_SSE_COMPILE
    if (caps->has_sse) {
        simd_mul_f32_sse(a, b, out, n);
        return;
    }
#endif
#ifdef CML_HAS_NEON_COMPILE
    if (caps->has_neon) {
        simd_mul_f32_neon(a, b, out, n);
        return;
    }
#endif
    simd_mul_f32_scalar(a, b, out, n);
}

void simd_div_f32(const float* a, const float* b, float* out, size_t n) {
    if (!a || !b || !out || n == 0)
        return;
    const CMLSimdCaps* caps = cml_get_simd_caps();

#ifdef CML_HAS_AVX512_COMPILE
    if (caps->has_avx512) {
        simd_div_f32_avx512(a, b, out, n);
        return;
    }
#endif
#ifdef CML_HAS_AVX_COMPILE
    if (caps->has_avx) {
        simd_div_f32_avx(a, b, out, n);
        return;
    }
#endif
#ifdef CML_HAS_SSE_COMPILE
    if (caps->has_sse) {
        simd_div_f32_sse(a, b, out, n);
        return;
    }
#endif
#ifdef CML_HAS_NEON_COMPILE
    if (caps->has_neon) {
        simd_div_f32_neon(a, b, out, n);
        return;
    }
#endif
    simd_div_f32_scalar(a, b, out, n);
}

void simd_transpose_f32(const float* in, float* out, int rows, int cols) {
    if (!in || !out || rows <= 0 || cols <= 0)
        return;

    const int BLOCK = 8;

    for (int i0 = 0; i0 < rows; i0 += BLOCK) {
        for (int j0 = 0; j0 < cols; j0 += BLOCK) {
            int i_end = (i0 + BLOCK < rows) ? i0 + BLOCK : rows;
            int j_end = (j0 + BLOCK < cols) ? j0 + BLOCK : cols;

            for (int i = i0; i < i_end; i++) {
                for (int j = j0; j < j_end; j++) {
                    out[j * rows + i] = in[i * cols + j];
                }
            }
        }
    }
}

void simd_add_scalar_f32(const float* a, float scalar, float* out, size_t n) {
    if (!a || !out || n == 0)
        return;
    size_t i = 0;

#ifdef __AVX512F__
    __m512 vscalar = _mm512_set1_ps(scalar);
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        _mm512_storeu_ps(&out[i], _mm512_add_ps(va, vscalar));
    }
#endif
#ifdef __AVX__
    __m256 vscalar256 = _mm256_set1_ps(scalar);
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        _mm256_storeu_ps(&out[i], _mm256_add_ps(va, vscalar256));
    }
#endif
#ifdef __SSE__
    __m128 vscalar128 = _mm_set1_ps(scalar);
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        _mm_storeu_ps(&out[i], _mm_add_ps(va, vscalar128));
    }
#endif
    for (; i < n; i++) {
        out[i] = a[i] + scalar;
    }
}

void simd_mul_scalar_f32(const float* a, float scalar, float* out, size_t n) {
    if (!a || !out || n == 0)
        return;
    size_t i = 0;

#ifdef __AVX512F__
    __m512 vscalar = _mm512_set1_ps(scalar);
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        _mm512_storeu_ps(&out[i], _mm512_mul_ps(va, vscalar));
    }
#endif
#ifdef __AVX__
    __m256 vscalar256 = _mm256_set1_ps(scalar);
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        _mm256_storeu_ps(&out[i], _mm256_mul_ps(va, vscalar256));
    }
#endif
#ifdef __SSE__
    __m128 vscalar128 = _mm_set1_ps(scalar);
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        _mm_storeu_ps(&out[i], _mm_mul_ps(va, vscalar128));
    }
#endif
    for (; i < n; i++) {
        out[i] = a[i] * scalar;
    }
}

void simd_add_broadcast_f32(const float* a, size_t a_n, const float* b, size_t b_n, float* out,
                            size_t out_n) {
    if (!a || !b || !out || out_n == 0)
        return;

    // Fast path: scalar broadcast
    if (a_n == 1) {
        simd_add_scalar_f32(b, a[0], out, out_n);
        return;
    }
    if (b_n == 1) {
        simd_add_scalar_f32(a, b[0], out, out_n);
        return;
    }

    // Fast path: same size
    if (a_n == b_n && a_n == out_n) {
        simd_add_f32(a, b, out, out_n);
        return;
    }

    // General broadcast with SIMD acceleration for aligned cases
    // When b_n divides out_n evenly and a_n == out_n, we can vectorize
    if (a_n == out_n && out_n % b_n == 0) {
        size_t repeats = out_n / b_n;
        for (size_t r = 0; r < repeats; r++) {
            simd_add_f32(&a[r * b_n], b, &out[r * b_n], b_n);
        }
        return;
    }

    // Fallback: scalar broadcast
    for (size_t i = 0; i < out_n; i++) {
        size_t ai = (a_n == 1) ? 0 : i % a_n;
        size_t bi = (b_n == 1) ? 0 : i % b_n;
        out[i]    = a[ai] + b[bi];
    }
}

void simd_mul_broadcast_f32(const float* a, size_t a_n, const float* b, size_t b_n, float* out,
                            size_t out_n) {
    if (!a || !b || !out || out_n == 0)
        return;

    // Fast path: scalar broadcast
    if (a_n == 1) {
        simd_mul_scalar_f32(b, a[0], out, out_n);
        return;
    }
    if (b_n == 1) {
        simd_mul_scalar_f32(a, b[0], out, out_n);
        return;
    }

    // Fast path: same size
    if (a_n == b_n && a_n == out_n) {
        simd_mul_f32(a, b, out, out_n);
        return;
    }

    // General broadcast with SIMD acceleration
    if (a_n == out_n && out_n % b_n == 0) {
        size_t repeats = out_n / b_n;
        for (size_t r = 0; r < repeats; r++) {
            simd_mul_f32(&a[r * b_n], b, &out[r * b_n], b_n);
        }
        return;
    }

    // Fallback: scalar
    for (size_t i = 0; i < out_n; i++) {
        size_t ai = (a_n == 1) ? 0 : i % a_n;
        size_t bi = (b_n == 1) ? 0 : i % b_n;
        out[i]    = a[ai] * b[bi];
    }
}

void simd_max_broadcast_f32(const float* a, size_t a_n, const float* b, size_t b_n, float* out,
                            size_t out_n) {
    if (!a || !b || !out || out_n == 0)
        return;

    // Fast path: same size
    if (a_n == b_n && a_n == out_n) {
        simd_max_f32(a, b, out, out_n);
        return;
    }

    // Scalar broadcast with SIMD
    if (b_n == 1) {
        float scalar = b[0];
        size_t i     = 0;
#ifdef __AVX__
        __m256 vscalar = _mm256_set1_ps(scalar);
        for (; i + 8 <= out_n && i + 8 <= a_n; i += 8) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            _mm256_storeu_ps(&out[i], _mm256_max_ps(va, vscalar));
        }
#endif
        for (; i < out_n; i++) {
            size_t ai = (a_n == 1) ? 0 : i % a_n;
            out[i]    = (a[ai] > scalar) ? a[ai] : scalar;
        }
        return;
    }

    // General broadcast with SIMD
    if (a_n == out_n && out_n % b_n == 0) {
        size_t repeats = out_n / b_n;
        for (size_t r = 0; r < repeats; r++) {
            simd_max_f32(&a[r * b_n], b, &out[r * b_n], b_n);
        }
        return;
    }

    // Fallback
    for (size_t i = 0; i < out_n; i++) {
        size_t ai = (a_n == 1) ? 0 : i % a_n;
        size_t bi = (b_n == 1) ? 0 : i % b_n;
        out[i]    = (a[ai] > b[bi]) ? a[ai] : b[bi];
    }
}

#include "backend/threadpool.h"

static size_t g_parallel_threshold = 10000; // Min elements for parallel execution

void simd_set_parallel_threshold(size_t threshold) { g_parallel_threshold = threshold; }

typedef struct {
    const float* a;
    const float* b;
    float* out;
} ParallelBinaryData;

static void parallel_add_task(void* data, size_t start, size_t end) {
    ParallelBinaryData* d = (ParallelBinaryData*)data;
    simd_add_f32(&d->a[start], &d->b[start], &d->out[start], end - start);
}

void simd_add_f32_parallel(const float* a, const float* b, float* out, size_t n) {
    if (!a || !b || !out || n == 0)
        return;

    // Use single-threaded for small arrays
    if (n < g_parallel_threshold) {
        simd_add_f32(a, b, out, n);
        return;
    }

    ThreadPool* pool = threadpool_get_global();
    if (!pool) {
        simd_add_f32(a, b, out, n);
        return;
    }

    ParallelBinaryData data = {a, b, out};
    threadpool_parallel_for(pool, parallel_add_task, &data, n);
}

static void parallel_mul_task(void* data, size_t start, size_t end) {
    ParallelBinaryData* d = (ParallelBinaryData*)data;
    simd_mul_f32(&d->a[start], &d->b[start], &d->out[start], end - start);
}

void simd_mul_f32_parallel(const float* a, const float* b, float* out, size_t n) {
    if (!a || !b || !out || n == 0)
        return;

    if (n < g_parallel_threshold) {
        simd_mul_f32(a, b, out, n);
        return;
    }

    ThreadPool* pool = threadpool_get_global();
    if (!pool) {
        simd_mul_f32(a, b, out, n);
        return;
    }

    ParallelBinaryData data = {a, b, out};
    threadpool_parallel_for(pool, parallel_mul_task, &data, n);
}

typedef struct {
    const float* in;
    float* out;
} ParallelUnaryData;

static void parallel_exp_task(void* data, size_t start, size_t end) {
    ParallelUnaryData* d = (ParallelUnaryData*)data;
    simd_exp_f32(&d->in[start], &d->out[start], end - start);
}

void simd_exp_f32_parallel(const float* in, float* out, size_t n) {
    if (!in || !out || n == 0)
        return;

    if (n < g_parallel_threshold) {
        simd_exp_f32(in, out, n);
        return;
    }

    ThreadPool* pool = threadpool_get_global();
    if (!pool) {
        simd_exp_f32(in, out, n);
        return;
    }

    ParallelUnaryData data = {in, out};
    threadpool_parallel_for(pool, parallel_exp_task, &data, n);
}

typedef struct {
    const float* data;
    float* partial_sums;
    size_t num_threads;
} ParallelSumData;

static void parallel_sum_task(void* data, size_t start, size_t end) {
    ParallelSumData* d = (ParallelSumData*)data;
    // Compute partial sum for this chunk
    float sum = simd_sum_float(&d->data[start], end - start);
    // Store in partial sums array (thread-safe: each thread writes to different index)
    // Approximate thread index from start position
    size_t chunk_size = (end - start);
    if (chunk_size > 0) {
        size_t thread_idx = start / chunk_size;
        if (thread_idx < d->num_threads) {
            d->partial_sums[thread_idx] = sum;
        }
    }
}

float simd_sum_f32_parallel(const float* data, size_t n) {
    if (!data || n == 0)
        return 0.0f;

    if (n < g_parallel_threshold) {
        return simd_sum_float(data, n);
    }

    ThreadPool* pool = threadpool_get_global();
    if (!pool) {
        return simd_sum_float(data, n);
    }

    size_t num_threads = threadpool_get_num_threads(pool);
    if (num_threads == 0)
        num_threads = 1;

    float* partial_sums = calloc(num_threads, sizeof(float));
    if (!partial_sums) {
        return simd_sum_float(data, n);
    }

    ParallelSumData pdata = {data, partial_sums, num_threads};
    threadpool_parallel_for(pool, parallel_sum_task, &pdata, n);

    // Reduce partial sums
    float total = 0.0f;
    for (size_t i = 0; i < num_threads; i++) {
        total += partial_sums[i];
    }
    free(partial_sums);

    return total;
}
