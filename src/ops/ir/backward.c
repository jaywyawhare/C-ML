#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "tensor/tensor.h"
#include "core/logging.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "ops/uops.h"
#include "backend/blas.h"
#include "ops/simd_math.h"

#ifdef __SSE__
#include <xmmintrin.h>
#include <emmintrin.h>
#endif
#ifdef __AVX__
#include <immintrin.h>
#endif

/* Access the shared BLAS context from execution.c */
extern CMLBlasContext* get_blas_context(void);

int cml_ir_build_backward(CMLGraph_t ir, struct IRNode* output_node) {
    if (!ir || !output_node) {
        LOG_ERROR("Invalid arguments to cml_ir_build_backward");
        return -1;
    }

    output_node->is_used = true;
    return 0;
}

static Tensor* ensure_grad(Tensor* t) {
    if (!t)
        return NULL;
    if (!t->grad) {
        TensorConfig config = {
            .dtype = t->dtype, .device = t->device, .has_dtype = true, .has_device = true};
        t->grad = tensor_zeros(t->shape, t->ndim, &config);
    }
    return t->grad;
}

static void accumulate_grad(Tensor* t, float* grad_data, size_t numel) {
    if (!t || !grad_data)
        return;
    Tensor* g = ensure_grad(t);
    if (!g || !g->data)
        return;

    float* g_data  = (float*)g->data;
    size_t t_numel = t->numel;

    if (t_numel == numel) {
        /* Fast path: SIMD-accelerated direct accumulation */
        simd_add_f32(g_data, grad_data, g_data, numel);
    } else if (t_numel == 1) {
        /* Scalar gradient: sum all */
        float sum = 0.0f;
#ifdef __AVX__
        size_t i = 0;
        __m256 vsum = _mm256_setzero_ps();
        for (; i + 8 <= numel; i += 8)
            vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(grad_data + i));
        /* Horizontal sum of 8 floats */
        __m128 lo = _mm256_castps256_ps128(vsum);
        __m128 hi = _mm256_extractf128_ps(vsum, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
        lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
        sum = _mm_cvtss_f32(lo);
        for (; i < numel; i++)
            sum += grad_data[i];
#else
        for (size_t i = 0; i < numel; i++)
            sum += grad_data[i];
#endif
        g_data[0] += sum;
    } else {
        /* Broadcast: accumulate with modulo */
        for (size_t i = 0; i < numel; i++)
            g_data[i % t_numel] += grad_data[i];
    }
}

static int cpu_backward_node(struct IRNode* node) {
    if (!node || !node->output)
        return 0;

    Tensor* out = node->output;

    if (!out->grad || !out->grad->data)
        return 0;

    float* out_grad  = (float*)out->grad->data;
    size_t out_numel = out->numel;

    Tensor* in1 = (node->num_inputs >= 1 && node->inputs) ? node->inputs[0] : NULL;
    Tensor* in2 = (node->num_inputs >= 2 && node->inputs) ? node->inputs[1] : NULL;

    switch (node->type) {
    case UOP_ADD:
        // d(a+b)/da = 1, d(a+b)/db = 1
        if (in1 && in1->requires_grad) {
            accumulate_grad(in1, out_grad, out_numel);
        }
        if (in2 && in2->requires_grad) {
            accumulate_grad(in2, out_grad, out_numel);
        }
        break;

    case UOP_SUB:
        // d(a-b)/da = 1, d(a-b)/db = -1
        if (in1 && in1->requires_grad) {
            accumulate_grad(in1, out_grad, out_numel);
        }
        if (in2 && in2->requires_grad) {
            Tensor* g2 = ensure_grad(in2);
            if (g2 && g2->data) {
                float* g2_data = (float*)g2->data;
                if (in2->numel == out_numel) {
                    /* Fast path: SIMD subtract */
                    simd_sub_f32(g2_data, out_grad, g2_data, out_numel);
                } else {
                    for (size_t i = 0; i < out_numel; i++)
                        g2_data[i % in2->numel] -= out_grad[i];
                }
            }
        }
        break;

    case UOP_MUL:
        // d(a*b)/da = b, d(a*b)/db = a
        if (in1 && in1->requires_grad && in2 && in2->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data  = (float*)g1->data;
                float* in2_data = (float*)in2->data;
                if (in1->numel == out_numel && in2->numel == out_numel) {
                    /* Fast path: no broadcast, SIMD fused multiply-add */
                    size_t i = 0;
#ifdef __AVX__
                    for (; i + 8 <= out_numel; i += 8) {
                        __m256 og = _mm256_loadu_ps(out_grad + i);
                        __m256 b  = _mm256_loadu_ps(in2_data + i);
                        __m256 cur = _mm256_loadu_ps(g1_data + i);
                        cur = _mm256_add_ps(cur, _mm256_mul_ps(og, b));
                        _mm256_storeu_ps(g1_data + i, cur);
                    }
#endif
                    for (; i < out_numel; i++)
                        g1_data[i] += out_grad[i] * in2_data[i];
                } else {
                    for (size_t i = 0; i < out_numel; i++)
                        g1_data[i % in1->numel] += out_grad[i] * in2_data[i % in2->numel];
                }
            }
        }
        if (in2 && in2->requires_grad && in1 && in1->data) {
            Tensor* g2 = ensure_grad(in2);
            if (g2 && g2->data) {
                float* g2_data  = (float*)g2->data;
                float* in1_data = (float*)in1->data;
                if (in1->numel == out_numel && in2->numel == out_numel) {
                    size_t i = 0;
#ifdef __AVX__
                    for (; i + 8 <= out_numel; i += 8) {
                        __m256 og = _mm256_loadu_ps(out_grad + i);
                        __m256 a  = _mm256_loadu_ps(in1_data + i);
                        __m256 cur = _mm256_loadu_ps(g2_data + i);
                        cur = _mm256_add_ps(cur, _mm256_mul_ps(og, a));
                        _mm256_storeu_ps(g2_data + i, cur);
                    }
#endif
                    for (; i < out_numel; i++)
                        g2_data[i] += out_grad[i] * in1_data[i];
                } else {
                    for (size_t i = 0; i < out_numel; i++)
                        g2_data[i % in2->numel] += out_grad[i] * in1_data[i % in1->numel];
                }
            }
        }
        break;

    case UOP_DIV:
        // d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
        if (in1 && in1->requires_grad && in2 && in2->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data  = (float*)g1->data;
                float* in2_data = (float*)in2->data;
                for (size_t i = 0; i < out_numel; i++) {
                    size_t i1 = i % in1->numel;
                    size_t i2 = i % in2->numel;
                    g1_data[i1] += out_grad[i] / (in2_data[i2] + 1e-8f);
                }
            }
        }
        if (in2 && in2->requires_grad && in1 && in1->data && in2->data) {
            Tensor* g2 = ensure_grad(in2);
            if (g2 && g2->data) {
                float* g2_data  = (float*)g2->data;
                float* in1_data = (float*)in1->data;
                float* in2_data = (float*)in2->data;
                for (size_t i = 0; i < out_numel; i++) {
                    size_t i1 = i % in1->numel;
                    size_t i2 = i % in2->numel;
                    float b   = in2_data[i2] + 1e-8f;
                    g2_data[i2] -= out_grad[i] * in1_data[i1] / (b * b);
                }
            }
        }
        break;

    case UOP_NEG:
        // d(-a)/da = -1
        if (in1 && in1->requires_grad) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data = (float*)g1->data;
                for (size_t i = 0; i < out_numel; i++) {
                    g1_data[i % in1->numel] -= out_grad[i];
                }
            }
        }
        break;

    case UOP_POW: {
        // d(a^b)/da = b * a^(b-1), d(a^b)/db = a^b * log(a)
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data && out->data) {
                float* g1_data  = (float*)g1->data;
                float* in1_data = (float*)in1->data;
                if (in2 && in2->numel == 1) {
                    float exp_val = ((float*)in2->data)[0];
                    for (size_t i = 0; i < out_numel; i++) {
                        size_t i1 = i % in1->numel;
                        g1_data[i1] += out_grad[i] * exp_val * powf(in1_data[i1], exp_val - 1.0f);
                    }
                }
            }
        }
        break;
    }

    case UOP_EXP:
        // d(exp(a))/da = exp(a)
        if (in1 && in1->requires_grad && out->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data  = (float*)g1->data;
                float* out_data = (float*)out->data;
                for (size_t i = 0; i < out_numel; i++) {
                    g1_data[i % in1->numel] += out_grad[i] * out_data[i];
                }
            }
        }
        break;

    case UOP_LOG:
        // d(log(a))/da = 1/a
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data  = (float*)g1->data;
                float* in1_data = (float*)in1->data;
                for (size_t i = 0; i < out_numel; i++) {
                    size_t i1 = i % in1->numel;
                    g1_data[i1] += out_grad[i] / (in1_data[i1] + 1e-8f);
                }
            }
        }
        break;

    case UOP_SQRT:
        // d(sqrt(a))/da = 0.5/sqrt(a)
        if (in1 && in1->requires_grad && out->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data  = (float*)g1->data;
                float* out_data = (float*)out->data;
                for (size_t i = 0; i < out_numel; i++) {
                    g1_data[i % in1->numel] += out_grad[i] * 0.5f / (out_data[i] + 1e-8f);
                }
            }
        }
        break;

    case UOP_SQUARE:
        // d(x^2)/dx = 2*x
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data  = (float*)g1->data;
                float* in1_data = (float*)in1->data;
                for (size_t i = 0; i < out_numel; i++) {
                    g1_data[i % in1->numel] += out_grad[i] * 2.0f * in1_data[i % in1->numel];
                }
            }
        }
        break;

    case UOP_RECIP:
        // d(1/a)/da = -1/a^2 = -recip(a)^2
        if (in1 && in1->requires_grad && out->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data  = (float*)g1->data;
                float* out_data = (float*)out->data;
                for (size_t i = 0; i < out_numel; i++) {
                    float r = out_data[i]; // recip(a) = 1/a
                    g1_data[i % in1->numel] += out_grad[i] * (-r * r);
                }
            }
        }
        break;

    case UOP_SIN:
        // d(sin(a))/da = cos(a) = sin(a + pi/2)
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data  = (float*)g1->data;
                float* in1_data = (float*)in1->data;
                for (size_t i = 0; i < out_numel; i++) {
                    size_t i1 = i % in1->numel;
                    g1_data[i1] += out_grad[i] * cosf(in1_data[i1]);
                }
            }
        }
        break;

    case UOP_WHERE:
        // where(cond, a, b): grad flows to a where cond is true, to b where false
        // No gradient for cond (discrete)
        if (node->num_inputs >= 3) {
            Tensor* cond = node->inputs[0];
            Tensor* in_a = node->inputs[1];
            Tensor* in_b = node->inputs[2];
            if (cond && cond->data) {
                float* cond_data = (float*)cond->data;
                size_t cond_numel = cond->numel;
                if (in_a && in_a->requires_grad) {
                    Tensor* ga = ensure_grad(in_a);
                    if (ga && ga->data) {
                        float* ga_data = (float*)ga->data;
                        for (size_t i = 0; i < out_numel; i++) {
                            size_t ci = (cond_numel == 1) ? 0 : i % cond_numel;
                            if (cond_data[ci] != 0.0f) {
                                ga_data[i % in_a->numel] += out_grad[i];
                            }
                        }
                    }
                }
                if (in_b && in_b->requires_grad) {
                    Tensor* gb = ensure_grad(in_b);
                    if (gb && gb->data) {
                        float* gb_data = (float*)gb->data;
                        for (size_t i = 0; i < out_numel; i++) {
                            size_t ci = (cond_numel == 1) ? 0 : i % cond_numel;
                            if (cond_data[ci] == 0.0f) {
                                gb_data[i % in_b->numel] += out_grad[i];
                            }
                        }
                    }
                }
            }
        }
        break;

    case UOP_CMPLT:
        // Comparison ops have no gradient (discrete/non-differentiable)
        break;

    case UOP_SIGMOID:
        // d(sigmoid(a))/da = sigmoid(a) * (1 - sigmoid(a))
        if (in1 && in1->requires_grad && out->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data  = (float*)g1->data;
                float* out_data = (float*)out->data;
                for (size_t i = 0; i < out_numel; i++) {
                    float s = out_data[i];
                    g1_data[i % in1->numel] += out_grad[i] * s * (1.0f - s);
                }
            }
        }
        break;

    case UOP_TANH:
        // d(tanh(a))/da = 1 - tanh^2(a)
        if (in1 && in1->requires_grad && out->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data  = (float*)g1->data;
                float* out_data = (float*)out->data;
                for (size_t i = 0; i < out_numel; i++) {
                    float t = out_data[i];
                    g1_data[i % in1->numel] += out_grad[i] * (1.0f - t * t);
                }
            }
        }
        break;

    case UOP_SUM:
    case UOP_MEAN:
        // d(sum(a))/da = 1 for all elements
        // d(mean(a))/da = 1/n for all elements
        if (in1 && in1->requires_grad) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data = (float*)g1->data;
                float scale =
                    (node->type == UOP_MEAN && in1->numel > 0) ? 1.0f / (float)in1->numel : 1.0f;
                float val = out_grad[0] * scale;
                size_t n = in1->numel;
                size_t i = 0;
#ifdef __AVX__
                __m256 vval = _mm256_set1_ps(val);
                for (; i + 8 <= n; i += 8) {
                    __m256 cur = _mm256_loadu_ps(g1_data + i);
                    _mm256_storeu_ps(g1_data + i, _mm256_add_ps(cur, vval));
                }
#elif defined(__SSE__)
                __m128 vval = _mm_set1_ps(val);
                for (; i + 4 <= n; i += 4) {
                    __m128 cur = _mm_loadu_ps(g1_data + i);
                    _mm_storeu_ps(g1_data + i, _mm_add_ps(cur, vval));
                }
#endif
                for (; i < n; i++)
                    g1_data[i] += val;
            }
        }
        break;

    case UOP_MAX:
        // d(max(a,b))/da = 1 if a >= b, else 0
        // d(max(a,b))/db = 1 if b > a, else 0
        if (in1 && in2 && in1->data && in2->data) {
            float* in1_data = (float*)in1->data;
            float* in2_data = (float*)in2->data;

            if (in1->requires_grad) {
                Tensor* g1 = ensure_grad(in1);
                if (g1 && g1->data) {
                    float* g1_data = (float*)g1->data;
                    if (in1->numel == out_numel && in2->numel == out_numel) {
                        /* Fast path: no broadcasting */
                        for (size_t i = 0; i < out_numel; i++) {
                            if (in1_data[i] >= in2_data[i])
                                g1_data[i] += out_grad[i];
                        }
                    } else if (in2->numel == 1) {
                        /* Common: relu = max(x, 0) — scalar broadcast */
                        float threshold = in2_data[0];
                        for (size_t i = 0; i < out_numel; i++) {
                            if (in1_data[i % in1->numel] >= threshold)
                                g1_data[i % in1->numel] += out_grad[i];
                        }
                    } else {
                        for (size_t i = 0; i < out_numel; i++) {
                            size_t i1 = i % in1->numel;
                            size_t i2 = i % in2->numel;
                            if (in1_data[i1] >= in2_data[i2])
                                g1_data[i1] += out_grad[i];
                        }
                    }
                }
            }
            if (in2->requires_grad) {
                Tensor* g2 = ensure_grad(in2);
                if (g2 && g2->data) {
                    float* g2_data = (float*)g2->data;
                    if (in1->numel == out_numel && in2->numel == out_numel) {
                        for (size_t i = 0; i < out_numel; i++) {
                            if (in2_data[i] > in1_data[i])
                                g2_data[i] += out_grad[i];
                        }
                    } else {
                        for (size_t i = 0; i < out_numel; i++) {
                            size_t i1 = i % in1->numel;
                            size_t i2 = i % in2->numel;
                            if (in2_data[i2] > in1_data[i1])
                                g2_data[i2] += out_grad[i];
                        }
                    }
                }
            }
        }
        break;

    case UOP_MATMUL: {
        if (!in1 || !in2 || in1->ndim < 2 || in2->ndim < 2)
            break;

        int M = in1->shape[in1->ndim - 2];
        int K = in1->shape[in1->ndim - 1];
        int N = in2->shape[in2->ndim - 1];

        CMLBlasContext* blas = get_blas_context();

        if (in1->requires_grad && in2->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data  = (float*)g1->data;
                float* in2_data = (float*)in2->data;
                if (blas && blas->initialized) {
                    cml_blas_sgemm_ex(blas, out_grad, in2_data, g1_data,
                                      M, K, N, 1.0f, 1.0f, false, true);
                } else {
                    for (int m = 0; m < M; m++) {
                        for (int k = 0; k < K; k++) {
                            float sum = 0.0f;
                            for (int n = 0; n < N; n++) {
                                sum += out_grad[m * N + n] * in2_data[k * N + n];
                            }
                            g1_data[m * K + k] += sum;
                        }
                    }
                }
            }
        }

        if (in2->requires_grad && in1->data) {
            Tensor* g2 = ensure_grad(in2);
            if (g2 && g2->data) {
                float* g2_data  = (float*)g2->data;
                float* in1_data = (float*)in1->data;
                if (blas && blas->initialized) {
                    cml_blas_sgemm_ex(blas, in1_data, out_grad, g2_data,
                                      K, N, M, 1.0f, 1.0f, true, false);
                } else {
                    for (int k = 0; k < K; k++) {
                        for (int n = 0; n < N; n++) {
                            float sum = 0.0f;
                            for (int m = 0; m < M; m++) {
                                sum += in1_data[m * K + k] * out_grad[m * N + n];
                            }
                            g2_data[k * N + n] += sum;
                        }
                    }
                }
            }
        }
        break;
    }

    case UOP_LINEAR: {
        if (!in1 || !in2 || in1->ndim < 2 || in2->ndim != 2)
            break;

        int M = in1->shape[in1->ndim - 2];
        int K = in1->shape[in1->ndim - 1];
        int N = in2->shape[0];

        CMLBlasContext* blas = get_blas_context();

        if (in1->requires_grad && in2->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data = (float*)g1->data;
                float* w_data = (float*)in2->data;
                if (blas && blas->initialized) {
                    cml_blas_sgemm(blas, out_grad, w_data, g1_data,
                                   M, K, N, 1.0f, 1.0f);
                } else {
                    for (int m = 0; m < M; m++)
                        for (int k = 0; k < K; k++) {
                            float sum = 0.0f;
                            for (int n = 0; n < N; n++)
                                sum += out_grad[m * N + n] * w_data[n * K + k];
                            g1_data[m * K + k] += sum;
                        }
                }
            }
        }

        if (in2->requires_grad && in1->data) {
            Tensor* g2 = ensure_grad(in2);
            if (g2 && g2->data) {
                float* g2_data = (float*)g2->data;
                float* i_data = (float*)in1->data;
                if (blas && blas->initialized) {
                    cml_blas_sgemm_ex(blas, out_grad, i_data, g2_data,
                                      N, K, M, 1.0f, 1.0f, true, false);
                } else {
                    for (int n = 0; n < N; n++)
                        for (int k = 0; k < K; k++) {
                            float sum = 0.0f;
                            for (int m = 0; m < M; m++)
                                sum += out_grad[m * N + n] * i_data[m * K + k];
                            g2_data[n * K + k] += sum;
                        }
                }
            }
        }

        if (node->num_inputs >= 3 && node->inputs[2]) {
            Tensor* bias_t = node->inputs[2];
            if (bias_t && bias_t->requires_grad) {
                Tensor* gb = ensure_grad(bias_t);
                if (gb && gb->data) {
                    float* gb_data = (float*)gb->data;
                    if (blas && blas->initialized) {
                        for (int m = 0; m < M; m++)
                            cml_blas_saxpy(blas, out_grad + m * N, gb_data, N, 1.0f);
                    } else {
                        for (int m = 0; m < M; m++) {
                            const float* row = out_grad + m * N;
                            for (int n = 0; n < N; n++)
                                gb_data[n] += row[n];
                        }
                    }
                }
            }
        }
        break;
    }

    case UOP_SHRINK: {
        // Forward: out[...] = in[starts[d] + ..., ...]  (contiguous sub-region)
        // Backward: scatter out_grad back into the corresponding region of in.grad
        if (in1 && in1->requires_grad) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data = (float*)g1->data;
                ShrinkParams* sp = (ShrinkParams*)node->params;
                if (sp) {
                    if (in1->ndim == 1) {
                        int start = sp->starts[0];
                        int len   = sp->ends[0] - start;
                        for (int i = 0; i < len; i++)
                            g1_data[start + i] += out_grad[i];
                    } else if (in1->ndim == 2) {
                        int in_cols  = in1->shape[1];
                        int r_start  = sp->starts[0], c_start = sp->starts[1];
                        int out_rows = sp->ends[0] - r_start;
                        int out_cols = sp->ends[1] - c_start;
                        for (int r = 0; r < out_rows; r++)
                            for (int c = 0; c < out_cols; c++)
                                g1_data[(r_start + r) * in_cols + (c_start + c)] +=
                                    out_grad[r * out_cols + c];
                    }
                }
            }
        }
        break;
    }

    case UOP_PERMUTE: {
        // For 2D transpose: grad_input = transpose(grad_output)
        // If forward is: output = transpose(input), shape [M,N] -> [N,M]
        // Then backward is: input.grad = transpose(output.grad)
        if (in1 && in1->requires_grad && in1->ndim == 2 && out->ndim == 2) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data = (float*)g1->data;
                // out.grad has shape [N, M], we need to transpose it to [M, N]
                int N = out->shape[0]; // transposed rows
                int M = out->shape[1]; // transposed cols
                // Transpose the gradient back
                for (int i = 0; i < N; i++) {
                    for (int j = 0; j < M; j++) {
                        // out_grad[i,j] -> g1[j,i]
                        g1_data[j * N + i] += out_grad[i * M + j];
                    }
                }
            }
        }
        break;
    }

    case UOP_EXPAND:
        // d(expand(a))/da = sum along expanded dimensions
        // Expand broadcasts input to larger shape; backward sums gradients
        // back to the original shape.
        if (in1 && in1->requires_grad) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data && out->data) {
                float* g1_data = (float*)g1->data;
                for (size_t i = 0; i < out_numel; i++) {
                    g1_data[i % in1->numel] += out_grad[i];
                }
            }
        }
        break;

    case UOP_CONV2D: {
        /* Backward for 2D convolution using im2col + GEMM (BLAS-accelerated).
         * Forward:  out[co, spatial] = weight[co, ci*kh*kw] @ im2col(input)[ci*kh*kw, spatial]
         * Input grad:  col_grad = weight^T @ out_grad, then col2im → input_grad
         * Weight grad: weight_grad += out_grad @ im2col(input)^T
         */
        if (!in1 || !in2)
            break;
        if (in1->ndim != 4 || in2->ndim != 4)
            break;

        int NB = in1->shape[0], C_in = in1->shape[1];
        int H = in1->shape[2], W = in1->shape[3];
        int C_out = in2->shape[0], kH = in2->shape[2], kW = in2->shape[3];
        int oH = out->shape[2], oW = out->shape[3];

        int col_h = C_in * kH * kW;
        int col_w = oH * oW;
        CMLBlasContext* blas = get_blas_context();

        if (blas && blas->initialized) {
            /* BLAS im2col+GEMM path */
            float* col_buf = (float*)malloc((size_t)col_h * col_w * sizeof(float));
            if (!col_buf) break;

            /* Input gradient: weight^T @ out_grad → col_grad, then col2im */
            if (in1->requires_grad && in2->data) {
                Tensor* g1 = ensure_grad(in1);
                if (g1 && g1->data) {
                    float* g1_data = (float*)g1->data;
                    float* w_data  = (float*)in2->data;
                    for (int n = 0; n < NB; n++) {
                        const float* og_n = out_grad + (size_t)n * C_out * oH * oW;
                        /* col_buf[col_h, col_w] = w^T[col_h, C_out] @ og[C_out, col_w] */
                        cml_blas_sgemm_ex(blas, w_data, og_n, col_buf,
                                          col_h, col_w, C_out, 1.0f, 0.0f, true, false);
                        /* col2im: scatter col_buf into input gradient */
                        for (int ci = 0; ci < C_in; ci++) {
                            for (int kh = 0; kh < kH; kh++) {
                                for (int kw = 0; kw < kW; kw++) {
                                    int col_row = (ci * kH + kh) * kW + kw;
                                    const float* col_r = col_buf + (size_t)col_row * col_w;
                                    for (int oh = 0; oh < oH; oh++) {
                                        int ih = oh + kh;
                                        float* g1_row = g1_data + ((size_t)(n * C_in + ci) * H + ih) * W;
                                        const float* cr = col_r + oh * oW;
                                        for (int ow = 0; ow < oW; ow++)
                                            g1_row[ow + kw] += cr[ow];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            /* Weight gradient: out_grad @ im2col(input)^T */
            if (in2->requires_grad && in1->data) {
                Tensor* g2 = ensure_grad(in2);
                if (g2 && g2->data) {
                    float* g2_data = (float*)g2->data;
                    float* in_data = (float*)in1->data;
                    for (int n = 0; n < NB; n++) {
                        /* im2col: input_n → col_buf */
                        for (int ci = 0; ci < C_in; ci++) {
                            const float* in_ch = in_data + ((size_t)n * C_in + ci) * H * W;
                            for (int kh = 0; kh < kH; kh++) {
                                for (int kw = 0; kw < kW; kw++) {
                                    int col_row = (ci * kH + kh) * kW + kw;
                                    float* col_r = col_buf + (size_t)col_row * col_w;
                                    for (int oh = 0; oh < oH; oh++) {
                                        const float* src = in_ch + (oh + kh) * W + kw;
                                        float* dst = col_r + oh * oW;
                                        memcpy(dst, src, (size_t)oW * sizeof(float));
                                    }
                                }
                            }
                        }
                        const float* og_n = out_grad + (size_t)n * C_out * oH * oW;
                        /* g2[C_out, col_h] += og[C_out, col_w] @ col^T[col_w, col_h] */
                        cml_blas_sgemm_ex(blas, og_n, col_buf, g2_data,
                                          C_out, col_h, col_w, 1.0f, 1.0f, false, true);
                    }
                }
            }

            free(col_buf);
        } else {
            /* Naive fallback (no BLAS) — 7-deep nested loops */
            if (in1->requires_grad && in2->data) {
                Tensor* g1 = ensure_grad(in1);
                if (g1 && g1->data) {
                    float* g1_data = (float*)g1->data;
                    float* w_data  = (float*)in2->data;
                    for (int n = 0; n < NB; n++)
                        for (int ci = 0; ci < C_in; ci++)
                            for (int h = 0; h < H; h++)
                                for (int w_idx = 0; w_idx < W; w_idx++) {
                                    float sum = 0.0f;
                                    for (int co = 0; co < C_out; co++)
                                        for (int kh = 0; kh < kH; kh++)
                                            for (int kw = 0; kw < kW; kw++) {
                                                int oh = h - kh, ow = w_idx - kw;
                                                if (oh >= 0 && oh < oH && ow >= 0 && ow < oW)
                                                    sum += out_grad[((n * C_out + co) * oH + oh) * oW + ow]
                                                         * w_data[((co * C_in + ci) * kH + kh) * kW + kw];
                                            }
                                    g1_data[((n * C_in + ci) * H + h) * W + w_idx] += sum;
                                }
                }
            }
            if (in2->requires_grad && in1->data) {
                Tensor* g2 = ensure_grad(in2);
                if (g2 && g2->data) {
                    float* g2_data = (float*)g2->data;
                    float* in_data = (float*)in1->data;
                    for (int co = 0; co < C_out; co++)
                        for (int ci = 0; ci < C_in; ci++)
                            for (int kh = 0; kh < kH; kh++)
                                for (int kw = 0; kw < kW; kw++) {
                                    float sum = 0.0f;
                                    for (int n = 0; n < NB; n++)
                                        for (int oh = 0; oh < oH; oh++)
                                            for (int ow = 0; ow < oW; ow++)
                                                sum += out_grad[((n * C_out + co) * oH + oh) * oW + ow]
                                                     * in_data[((n * C_in + ci) * H + (oh + kh)) * W + (ow + kw)];
                                    g2_data[((co * C_in + ci) * kH + kh) * kW + kw] += sum;
                                }
                }
            }
        }
        break;
    }

    case UOP_ELU: {
        // d(elu)/dx = x > 0 ? 1 : alpha * exp(x) = x > 0 ? 1 : output + alpha
        if (in1 && in1->requires_grad && in1->data && out->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data = (float*)g1->data;
                float* in1_data = (float*)in1->data;
                float* out_data = (float*)out->data;
                ClampParams* cp = (ClampParams*)node->params;
                float alpha = cp ? cp->min_val : 1.0f;
                for (size_t i = 0; i < out_numel; i++) {
                    float x = in1_data[i % in1->numel];
                    float grad = (x > 0.0f) ? 1.0f : (out_data[i] + alpha);
                    g1_data[i % in1->numel] += out_grad[i] * grad;
                }
            }
        }
        break;
    }

    case UOP_SELU: {
        // d(selu)/dx = scale * (x > 0 ? 1 : alpha * exp(x))
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data = (float*)g1->data;
                float* in1_data = (float*)in1->data;
                const float selu_alpha = 1.6732632423543772f;
                const float selu_scale = 1.0507009873554805f;
                for (size_t i = 0; i < out_numel; i++) {
                    float x = in1_data[i % in1->numel];
                    float grad = selu_scale * (x > 0.0f ? 1.0f : selu_alpha * expf(x));
                    g1_data[i % in1->numel] += out_grad[i] * grad;
                }
            }
        }
        break;
    }

    case UOP_MISH: {
        // d(mish)/dx = d/dx[x * tanh(softplus(x))]
        // = tanh(sp) + x * sech^2(sp) * sigmoid(x)
        // where sp = softplus(x) = log(1+exp(x)), sigmoid(x) = 1/(1+exp(-x))
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data = (float*)g1->data;
                float* in1_data = (float*)in1->data;
                for (size_t i = 0; i < out_numel; i++) {
                    float x = in1_data[i % in1->numel];
                    float sp = fmaxf(x, 0.0f) + logf(1.0f + expf(-fabsf(x)));
                    float tsp = tanhf(sp);
                    float sig = 1.0f / (1.0f + expf(-x));
                    float sech2 = 1.0f - tsp * tsp;
                    float grad = tsp + x * sech2 * sig;
                    g1_data[i % in1->numel] += out_grad[i] * grad;
                }
            }
        }
        break;
    }

    case UOP_SILU: {
        // d(silu)/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data = (float*)g1->data;
                float* in1_data = (float*)in1->data;
                for (size_t i = 0; i < out_numel; i++) {
                    float x = in1_data[i % in1->numel];
                    float sig = 1.0f / (1.0f + expf(-x));
                    float grad = sig * (1.0f + x * (1.0f - sig));
                    g1_data[i % in1->numel] += out_grad[i] * grad;
                }
            }
        }
        break;
    }

    case UOP_HARDSWISH: {
        // d(hardswish)/dx = x >= 3 ? 1 : x <= -3 ? 0 : (2x+3)/6
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data = (float*)g1->data;
                float* in1_data = (float*)in1->data;
                for (size_t i = 0; i < out_numel; i++) {
                    float x = in1_data[i % in1->numel];
                    float grad;
                    if (x >= 3.0f) grad = 1.0f;
                    else if (x <= -3.0f) grad = 0.0f;
                    else grad = (2.0f * x + 3.0f) / 6.0f;
                    g1_data[i % in1->numel] += out_grad[i] * grad;
                }
            }
        }
        break;
    }

    case UOP_SOFTPLUS: {
        // d(softplus)/dx = sigmoid(x)
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data = (float*)g1->data;
                float* in1_data = (float*)in1->data;
                for (size_t i = 0; i < out_numel; i++) {
                    float x = in1_data[i % in1->numel];
                    float sig = 1.0f / (1.0f + expf(-x));
                    g1_data[i % in1->numel] += out_grad[i] * sig;
                }
            }
        }
        break;
    }

    case UOP_RELU: {
        // d(relu)/dx = x > 0 ? 1 : 0
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data = (float*)g1->data;
                float* in1_data = (float*)in1->data;
                size_t i = 0;
#ifdef __AVX__
                __m256 zero = _mm256_setzero_ps();
                for (; i + 8 <= out_numel; i += 8) {
                    __m256 x = _mm256_loadu_ps(in1_data + i);
                    __m256 g = _mm256_loadu_ps(out_grad + i);
                    __m256 mask = _mm256_cmp_ps(x, zero, _CMP_GE_OQ);
                    __m256 cur = _mm256_loadu_ps(g1_data + i);
                    cur = _mm256_add_ps(cur, _mm256_and_ps(g, mask));
                    _mm256_storeu_ps(g1_data + i, cur);
                }
#elif defined(__SSE__)
                __m128 zero = _mm_setzero_ps();
                for (; i + 4 <= out_numel; i += 4) {
                    __m128 x = _mm_loadu_ps(in1_data + i);
                    __m128 g = _mm_loadu_ps(out_grad + i);
                    __m128 mask = _mm_cmpge_ps(x, zero);
                    __m128 cur = _mm_loadu_ps(g1_data + i);
                    cur = _mm_add_ps(cur, _mm_and_ps(g, mask));
                    _mm_storeu_ps(g1_data + i, cur);
                }
#endif
                for (; i < out_numel; i++) {
                    if (in1_data[i] >= 0.0f)
                        g1_data[i] += out_grad[i];
                }
            }
        }
        break;
    }

    case UOP_RELU6: {
        // d(relu6)/dx = x > 0 && x < 6 ? 1 : 0
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data = (float*)g1->data;
                float* in1_data = (float*)in1->data;
                for (size_t i = 0; i < out_numel; i++) {
                    float x = in1_data[i % in1->numel];
                    float grad = (x > 0.0f && x < 6.0f) ? 1.0f : 0.0f;
                    g1_data[i % in1->numel] += out_grad[i] * grad;
                }
            }
        }
        break;
    }

    case UOP_HARD_SIGMOID: {
        // d(hard_sigmoid)/dx = x > -3 && x < 3 ? 1/6 : 0
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data = (float*)g1->data;
                float* in1_data = (float*)in1->data;
                for (size_t i = 0; i < out_numel; i++) {
                    float x = in1_data[i % in1->numel];
                    float grad = (x > -3.0f && x < 3.0f) ? (1.0f / 6.0f) : 0.0f;
                    g1_data[i % in1->numel] += out_grad[i] * grad;
                }
            }
        }
        break;
    }

    case UOP_HARD_TANH: {
        // d(hard_tanh)/dx = -1 < x && x < 1 ? 1 : 0
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data = (float*)g1->data;
                float* in1_data = (float*)in1->data;
                for (size_t i = 0; i < out_numel; i++) {
                    float x = in1_data[i % in1->numel];
                    float grad = (x > -1.0f && x < 1.0f) ? 1.0f : 0.0f;
                    g1_data[i % in1->numel] += out_grad[i] * grad;
                }
            }
        }
        break;
    }

    case UOP_CELU: {
        // d(celu)/dx = x > 0 ? 1 : exp(x/alpha)
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data = (float*)g1->data;
                float* in1_data = (float*)in1->data;
                ClampParams* cp = (ClampParams*)node->params;
                float alpha = cp ? cp->min_val : 1.0f;
                for (size_t i = 0; i < out_numel; i++) {
                    float x = in1_data[i % in1->numel];
                    float grad = x > 0.0f ? 1.0f : expf(x / alpha);
                    g1_data[i % in1->numel] += out_grad[i] * grad;
                }
            }
        }
        break;
    }

    case UOP_LOGSIGMOID: {
        // d(logsigmoid)/dx = 1 - sigmoid(x)
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data = (float*)g1->data;
                float* in1_data = (float*)in1->data;
                for (size_t i = 0; i < out_numel; i++) {
                    float x = in1_data[i % in1->numel];
                    float sig = 1.0f / (1.0f + expf(-x));
                    g1_data[i % in1->numel] += out_grad[i] * (1.0f - sig);
                }
            }
        }
        break;
    }

    case UOP_QUICK_GELU: {
        // d(quick_gelu)/dx = d/dx[x * sigmoid(1.702*x)]
        // = sigmoid(1.702x) + x * 1.702 * sigmoid(1.702x) * (1 - sigmoid(1.702x))
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data = (float*)g1->data;
                float* in1_data = (float*)in1->data;
                for (size_t i = 0; i < out_numel; i++) {
                    float x = in1_data[i % in1->numel];
                    float sig = 1.0f / (1.0f + expf(-1.702f * x));
                    float grad = sig + x * 1.702f * sig * (1.0f - sig);
                    g1_data[i % in1->numel] += out_grad[i] * grad;
                }
            }
        }
        break;
    }

    case UOP_SOFTSIGN: {
        // d(softsign)/dx = 1 / (1 + |x|)^2
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data = (float*)g1->data;
                float* in1_data = (float*)in1->data;
                for (size_t i = 0; i < out_numel; i++) {
                    float x = in1_data[i % in1->numel];
                    float denom = 1.0f + fabsf(x);
                    g1_data[i % in1->numel] += out_grad[i] / (denom * denom);
                }
            }
        }
        break;
    }

    case UOP_LERP: {
        // lerp(a, b, t) = a + t*(b-a)
        // d/da = 1 - t, d/db = t, d/dt = b - a
        if (node->num_inputs >= 3) {
            Tensor* in_a = node->inputs[0];
            Tensor* in_b = node->inputs[1];
            Tensor* in_t = node->inputs[2];
            if (in_t && in_t->data) {
                float* t_data = (float*)in_t->data;
                if (in_a && in_a->requires_grad) {
                    Tensor* ga = ensure_grad(in_a);
                    if (ga && ga->data) {
                        float* ga_data = (float*)ga->data;
                        for (size_t i = 0; i < out_numel; i++) {
                            float t = t_data[i % in_t->numel];
                            ga_data[i % in_a->numel] += out_grad[i] * (1.0f - t);
                        }
                    }
                }
                if (in_b && in_b->requires_grad) {
                    Tensor* gb = ensure_grad(in_b);
                    if (gb && gb->data) {
                        float* gb_data = (float*)gb->data;
                        for (size_t i = 0; i < out_numel; i++) {
                            float t = t_data[i % in_t->numel];
                            gb_data[i % in_b->numel] += out_grad[i] * t;
                        }
                    }
                }
            }
        }
        break;
    }

    case UOP_GATHER: {
        // Gather backward: scatter grad_output back to input positions
        // Forward: out[i] = input[i, indices[i]]  (2D input, 1D indices, dim=1)
        // Backward: grad_input[i, indices[i]] += grad_output[i]
        if (node->num_inputs >= 2) {
            Tensor* input_t = node->inputs[0];
            Tensor* indices_t = node->inputs[1];
            if (input_t && input_t->requires_grad && input_t->data && indices_t && indices_t->data) {
                Tensor* g1 = ensure_grad(input_t);
                if (g1 && g1->data) {
                    float* g1_data = (float*)g1->data;
                    float* idx_data = (float*)indices_t->data;
                    GatherParams* params = (GatherParams*)node->params;
                    int dim = params ? params->dim : -1;
                    if (dim < 0) dim = input_t->ndim + dim;

                    if (input_t->ndim == 2 && indices_t->ndim == 1 && dim == 1) {
                        size_t n_cols = (size_t)input_t->shape[1];
                        size_t n_rows = (size_t)indices_t->numel;
                        for (size_t i = 0; i < n_rows && i < out_numel; i++) {
                            int idx = (int)idx_data[i];
                            if (idx >= 0 && idx < (int)n_cols) {
                                g1_data[i * n_cols + (size_t)idx] += out_grad[i];
                            }
                        }
                    } else {
                        LOG_DEBUG("CPU backward: UOP_GATHER generic N-dim grad not implemented");
                    }
                }
            }
        }
        break;
    }

    case UOP_IDIV:
    case UOP_MOD:
    case UOP_SORT:
    case UOP_TOPK:
    case UOP_MASKED_SELECT:
    case UOP_SPLIT:
    case UOP_CHUNK:
    case UOP_MESHGRID:
    case UOP_DIAGONAL:
        // Non-differentiable or complex-gradient ops - no gradient
        break;

    default:
        // Unsupported op - gradients not computed
        LOG_DEBUG("CPU backward: no gradient rule for op type %d", node->type);
        break;
    }

    return 0;
}

static int cpu_execute_backward(CMLGraph_t ir) {
    if (!ir)
        return -1;

    int node_count   = 0;
    struct IRNode* n = ir->head;
    while (n) {
        node_count++;
        n = n->next;
    }

    if (node_count == 0)
        return 0;

    /* Use stack buffer for small graphs to avoid malloc/free overhead */
    struct IRNode* stack_buf[64];
    struct IRNode** nodes = (node_count <= 64) ? stack_buf
                                               : malloc(node_count * sizeof(struct IRNode*));
    if (!nodes) {
        LOG_ERROR("Failed to allocate node array for backward pass");
        return -1;
    }

    n = ir->head;
    for (int i = 0; i < node_count; i++) {
        nodes[i] = n;
        n        = n->next;
    }

    /* Loss gradient is already set up by cml_ir_execute_backward — no need to redo here */

    for (int i = node_count - 1; i >= 0; i--) {
        cpu_backward_node(nodes[i]);
    }

    if (nodes != stack_buf)
        free(nodes);
    return 0;
}

int cml_ir_execute_backward(CMLGraph_t ir) {
    if (!ir) {
        LOG_ERROR("NULL IR passed to cml_ir_execute_backward");
        return -1;
    }

    struct IRNode* node = ir->tail;
    if (!node) {
        LOG_ERROR("No tail node in IR graph");
        return -1;
    }

    if (node->output && node->output->requires_grad) {
        if (!node->output->grad) {
            Tensor* output      = node->output;
            TensorConfig config = {.dtype      = output->dtype,
                                   .device     = output->device,
                                   .has_dtype  = true,
                                   .has_device = true};
            node->output->grad  = tensor_zeros(output->shape, output->ndim, &config);
            if (!node->output->grad) {
                LOG_ERROR("Failed to allocate gradient tensor");
                return -1;
            }
        }
        if (node->output->numel == 1) {
            if (node->output->dtype == DTYPE_FLOAT32) {
                *(float*)node->output->grad->data = 1.0f;
            }
        }
    }

    return cpu_execute_backward(ir);
}
