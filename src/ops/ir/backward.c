#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "tensor/tensor.h"
#include "tensor/realize.h"
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

    /* Forward linear scan from ir->head to output_node.
     * For each node, propagate requires_grad from inputs and set
     * needs_input_grad[i] so cpu_execute_backward can skip nodes
     * that don't contribute to any gradient (backward DCE). */
    struct IRNode* n = ir->head;
    while (n) {
        bool any = false;
        for (int i = 0; i < n->num_inputs && n->inputs; i++) {
            Tensor* inp = n->inputs[i];
            bool inp_grad = inp && (inp->requires_grad ||
                                    (inp->ir_node && inp->ir_node->requires_grad));
            if (i < 8) n->needs_input_grad[i] = inp_grad;
            if (inp_grad) any = true;
        }
        n->requires_grad = any;
        n->is_used = true;
        if (n == output_node) break;
        n = n->next;
    }

    return 0;
}

static Tensor* ensure_grad(Tensor* t) {
    if (!t)
        return NULL;
    if (!t->grad) {
        /* Use tensor_empty (eager allocation) + memset to zero.
         * We must NOT use tensor_zeros here: it creates a lazy IR node and
         * calling tensor_realize mid-backward would re-execute the entire
         * forward graph, causing leaks and incorrect execution order.
         * An eagerly-allocated zero buffer is the right primitive for
         * gradient accumulation. */
        TensorConfig config = {
            .dtype = t->dtype, .device = t->device, .has_dtype = true, .has_device = true};
        t->grad = tensor_empty(t->shape, t->ndim, &config);
        if (t->grad && t->grad->data)
            memset(t->grad->data, 0, t->grad->numel * cml_dtype_size(t->grad->dtype));
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
        if (in1 && in1->requires_grad) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data = (float*)g1->data;
                ReduceParams* rp = (ReduceParams*)node->params;

                /* Global reduce (no dims specified) — broadcast scalar grad */
                if (!rp || rp->num_dims == 0) {
                    float scale = (node->type == UOP_MEAN && in1->numel > 0)
                                  ? 1.0f / (float)in1->numel : 1.0f;
                    float val = out_grad[0] * scale;
                    size_t i = 0;
#ifdef __AVX__
                    __m256 vval = _mm256_set1_ps(val);
                    for (; i + 8 <= in1->numel; i += 8) {
                        __m256 cur = _mm256_loadu_ps(g1_data + i);
                        _mm256_storeu_ps(g1_data + i, _mm256_add_ps(cur, vval));
                    }
#elif defined(__SSE__)
                    __m128 vval4 = _mm_set1_ps(val);
                    for (; i + 4 <= in1->numel; i += 4) {
                        __m128 cur = _mm_loadu_ps(g1_data + i);
                        _mm_storeu_ps(g1_data + i, _mm_add_ps(cur, vval4));
                    }
#endif
                    for (; i < in1->numel; i++)
                        g1_data[i] += val;
                } else {
                    /* Dim-specific reduce: outer × reduce × inner layout.
                     * grad_input[outer, j, inner] += grad_out[outer, inner] / reduce_size */
                    int ndim = in1->ndim;
                    /* Handle first reduce dim — general single-dim path works for the
                     * common case; multi-dim reduce falls back to modulo broadcast. */
                    int rd = rp->dims[0];
                    if (rd < 0) rd += ndim;

                    size_t outer = 1;
                    for (int d = 0; d < rd; d++) outer *= (size_t)in1->shape[d];
                    size_t reduce_size = (size_t)in1->shape[rd];
                    size_t inner = 1;
                    for (int d = rd + 1; d < ndim; d++) inner *= (size_t)in1->shape[d];

                    float scale = (node->type == UOP_MEAN) ? 1.0f / (float)reduce_size : 1.0f;

                    for (size_t o = 0; o < outer; o++) {
                        for (size_t r = 0; r < reduce_size; r++) {
                            for (size_t iv = 0; iv < inner; iv++) {
                                size_t in_idx  = (o * reduce_size + r) * inner + iv;
                                size_t out_idx = o * inner + iv;
                                g1_data[in_idx] += out_grad[out_idx] * scale;
                            }
                        }
                    }
                }
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
        /* Flatten all leading batch dims — treat as [batch * M, K] @ [K, N] */
        size_t batch = in1->numel / ((size_t)M * K);

        CMLBlasContext* blas = get_blas_context();
        int M_flat = (int)(batch * (size_t)M);

        if (in1->requires_grad && in2->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data  = (float*)g1->data;
                float* in2_data = (float*)in2->data;
                if (blas && blas->initialized) {
                    /* g1[M_flat, K] = out_grad[M_flat, N] @ in2[K, N]^T */
                    cml_blas_sgemm_ex(blas, out_grad, in2_data, g1_data,
                                      M_flat, K, N, 1.0f, 1.0f, false, true);
                } else {
                    for (int m = 0; m < M_flat; m++)
                        for (int k = 0; k < K; k++) {
                            float sum = 0.0f;
                            for (int n = 0; n < N; n++)
                                sum += out_grad[m * N + n] * in2_data[k * N + n];
                            g1_data[m * K + k] += sum;
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
                    /* g2[K, N] += in1[M_flat, K]^T @ out_grad[M_flat, N] */
                    cml_blas_sgemm_ex(blas, in1_data, out_grad, g2_data,
                                      K, N, M_flat, 1.0f, 1.0f, true, false);
                } else {
                    for (int k = 0; k < K; k++)
                        for (int n = 0; n < N; n++) {
                            float sum = 0.0f;
                            for (int m = 0; m < M_flat; m++)
                                sum += in1_data[m * K + k] * out_grad[m * N + n];
                            g2_data[k * N + n] += sum;
                        }
                }
            }
        }
        break;
    }

    case UOP_LINEAR: {
        if (!in1 || !in2 || in1->ndim < 2 || in2->ndim != 2)
            break;

        /* input: [..., M, K],  weight: [N, K],  out: [..., M, N]
         * Flatten all leading batch dims: M_flat = numel/K */
        int K = in1->shape[in1->ndim - 1];
        int N = in2->shape[0];
        int M_flat = (int)(in1->numel / (size_t)K);  /* batch * M */

        CMLBlasContext* blas = get_blas_context();

        if (in1->requires_grad && in2->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data = (float*)g1->data;
                float* w_data  = (float*)in2->data;
                /* g1[M_flat, K] = out_grad[M_flat, N] @ weight[N, K] */
                if (blas && blas->initialized) {
                    cml_blas_sgemm(blas, out_grad, w_data, g1_data,
                                   M_flat, K, N, 1.0f, 1.0f);
                } else {
                    for (int m = 0; m < M_flat; m++)
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
                float* i_data  = (float*)in1->data;
                /* g2[N, K] += out_grad[M_flat, N]^T @ input[M_flat, K] */
                if (blas && blas->initialized) {
                    cml_blas_sgemm_ex(blas, out_grad, i_data, g2_data,
                                      N, K, M_flat, 1.0f, 1.0f, true, false);
                } else {
                    for (int n = 0; n < N; n++)
                        for (int k = 0; k < K; k++) {
                            float sum = 0.0f;
                            for (int m = 0; m < M_flat; m++)
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
                    /* Sum over all M_flat rows */
                    if (blas && blas->initialized) {
                        for (int m = 0; m < M_flat; m++)
                            cml_blas_saxpy(blas, out_grad + m * N, gb_data, N, 1.0f);
                    } else {
                        for (int m = 0; m < M_flat; m++) {
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
        /* Backward for grouped 2D convolution.
         * weight shape: [C_out, C_in/groups, kH, kW]
         * For each group g:
         *   input  channels: [g*cig .. (g+1)*cig)
         *   output channels: [g*cog .. (g+1)*cog)
         */
        if (!in1 || !in2)
            break;
        if (in1->ndim != 4 || in2->ndim != 4)
            break;

        Conv2DParams* cp = (Conv2DParams*)node->params;
        int groups = (cp && cp->groups > 0) ? cp->groups : 1;
        int pad_h  = (cp && cp->padding)  ? cp->padding[0]  : 0;
        int pad_w  = (cp && cp->padding)  ? cp->padding[1]  : 0;
        int str_h  = (cp && cp->stride)   ? cp->stride[0]   : 1;
        int str_w  = (cp && cp->stride)   ? cp->stride[1]   : 1;
        int dil_h  = (cp && cp->dilation) ? cp->dilation[0] : 1;
        int dil_w  = (cp && cp->dilation) ? cp->dilation[1] : 1;

        int NB = in1->shape[0], C_in = in1->shape[1];
        int H  = in1->shape[2], W    = in1->shape[3];
        int C_out = in2->shape[0];
        int kH = in2->shape[2], kW = in2->shape[3];
        int oH = out->shape[2], oW = out->shape[3];

        int cig = C_in  / groups;   /* input  channels per group */
        int cog = C_out / groups;   /* output channels per group */
        int col_h = cig * kH * kW;
        int col_w = oH * oW;

        CMLBlasContext* blas = get_blas_context();

        if (blas && blas->initialized) {
            float* col_buf = (float*)malloc((size_t)col_h * col_w * sizeof(float));
            if (!col_buf) break;

            for (int g = 0; g < groups; g++) {
                int ci_start = g * cig, co_start = g * cog;

                /* Input gradient: w_g^T @ out_grad_g → col → col2im */
                if (in1->requires_grad && in2->data) {
                    Tensor* g1 = ensure_grad(in1);
                    if (g1 && g1->data) {
                        float* g1_data = (float*)g1->data;
                        float* w_data  = (float*)in2->data;
                        /* Weight slice for group g: [cog, cig, kH, kW] starting at co_start */
                        const float* wg = w_data + (size_t)co_start * cig * kH * kW;
                        for (int n = 0; n < NB; n++) {
                            const float* og_n = out_grad
                                + ((size_t)n * C_out + co_start) * oH * oW;
                            /* col[col_h, col_w] = wg^T[cog, col_h]^T @ og_n[cog, col_w] */
                            cml_blas_sgemm_ex(blas, wg, og_n, col_buf,
                                              col_h, col_w, cog, 1.0f, 0.0f, true, false);
                            /* col2im with stride/dilation/padding */
                            for (int ci = 0; ci < cig; ci++) {
                                for (int kh = 0; kh < kH; kh++) {
                                    for (int kw = 0; kw < kW; kw++) {
                                        int col_row = (ci * kH + kh) * kW + kw;
                                        const float* col_r = col_buf + (size_t)col_row * col_w;
                                        for (int oh = 0; oh < oH; oh++) {
                                            int ih = oh * str_h - pad_h + kh * dil_h;
                                            if (ih < 0 || ih >= H) continue;
                                            float* g1_row = g1_data
                                                + ((size_t)(n * C_in + ci_start + ci) * H + ih) * W;
                                            const float* cr = col_r + oh * oW;
                                            for (int ow = 0; ow < oW; ow++) {
                                                int iw = ow * str_w - pad_w + kw * dil_w;
                                                if (iw >= 0 && iw < W)
                                                    g1_row[iw] += cr[ow];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                /* Weight gradient: out_grad_g @ im2col(input_g)^T */
                if (in2->requires_grad && in1->data) {
                    Tensor* g2 = ensure_grad(in2);
                    if (g2 && g2->data) {
                        float* g2_data = (float*)g2->data;
                        float* in_data = (float*)in1->data;
                        float* wg_grad = g2_data + (size_t)co_start * cig * kH * kW;
                        for (int n = 0; n < NB; n++) {
                            /* im2col for group g's input channels, with stride/dilation/padding */
                            memset(col_buf, 0, (size_t)col_h * col_w * sizeof(float));
                            for (int ci = 0; ci < cig; ci++) {
                                const float* in_ch = in_data
                                    + ((size_t)n * C_in + ci_start + ci) * H * W;
                                for (int kh = 0; kh < kH; kh++) {
                                    for (int kw = 0; kw < kW; kw++) {
                                        int col_row = (ci * kH + kh) * kW + kw;
                                        float* col_r = col_buf + (size_t)col_row * col_w;
                                        for (int oh = 0; oh < oH; oh++) {
                                            int ih = oh * str_h - pad_h + kh * dil_h;
                                            if (ih < 0 || ih >= H) continue;
                                            for (int ow = 0; ow < oW; ow++) {
                                                int iw = ow * str_w - pad_w + kw * dil_w;
                                                col_r[oh * oW + ow] = (iw >= 0 && iw < W)
                                                    ? in_ch[ih * W + iw] : 0.f;
                                            }
                                        }
                                    }
                                }
                            }
                            const float* og_n = out_grad
                                + ((size_t)n * C_out + co_start) * oH * oW;
                            /* wg_grad[cog, col_h] += og_n[cog, col_w] @ col^T */
                            cml_blas_sgemm_ex(blas, og_n, col_buf, wg_grad,
                                              cog, col_h, col_w, 1.0f, 1.0f, false, true);
                        }
                    }
                }
            }

            free(col_buf);
        } else {
            /* Naive fallback with groups, stride, dilation, padding */
            if (in1->requires_grad && in2->data) {
                Tensor* g1 = ensure_grad(in1);
                if (g1 && g1->data) {
                    float* g1_data = (float*)g1->data;
                    float* w_data  = (float*)in2->data;
                    for (int g = 0; g < groups; g++) {
                        int ci_start = g * cig, co_start = g * cog;
                        for (int n = 0; n < NB; n++)
                            for (int ci = 0; ci < cig; ci++)
                                for (int h = 0; h < H; h++)
                                    for (int w_idx = 0; w_idx < W; w_idx++) {
                                        float sum = 0.f;
                                        for (int co = 0; co < cog; co++)
                                            for (int kh = 0; kh < kH; kh++)
                                                for (int kw = 0; kw < kW; kw++) {
                                                    int oh = (h + pad_h - kh * dil_h);
                                                    int ow = (w_idx + pad_w - kw * dil_w);
                                                    if (oh < 0 || oh % str_h || ow < 0 || ow % str_w) continue;
                                                    oh /= str_h; ow /= str_w;
                                                    if (oh >= oH || ow >= oW) continue;
                                                    sum += out_grad[((n * C_out + co_start + co) * oH + oh) * oW + ow]
                                                         * w_data[(((co_start + co) * cig + ci) * kH + kh) * kW + kw];
                                                }
                                        g1_data[((n * C_in + ci_start + ci) * H + h) * W + w_idx] += sum;
                                    }
                    }
                }
            }
            if (in2->requires_grad && in1->data) {
                Tensor* g2 = ensure_grad(in2);
                if (g2 && g2->data) {
                    float* g2_data = (float*)g2->data;
                    float* in_data = (float*)in1->data;
                    for (int g = 0; g < groups; g++) {
                        int ci_start = g * cig, co_start = g * cog;
                        for (int co = 0; co < cog; co++)
                            for (int ci = 0; ci < cig; ci++)
                                for (int kh = 0; kh < kH; kh++)
                                    for (int kw = 0; kw < kW; kw++) {
                                        float sum = 0.f;
                                        for (int n = 0; n < NB; n++)
                                            for (int oh = 0; oh < oH; oh++)
                                                for (int ow = 0; ow < oW; ow++) {
                                                    int ih = oh * str_h - pad_h + kh * dil_h;
                                                    int iw = ow * str_w - pad_w + kw * dil_w;
                                                    if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                                                    sum += out_grad[((n * C_out + co_start + co) * oH + oh) * oW + ow]
                                                         * in_data[((n * C_in + ci_start + ci) * H + ih) * W + iw];
                                                }
                                        g2_data[(((co_start + co) * cig + ci) * kH + kh) * kW + kw] += sum;
                                    }
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
                    } else if (input_t->ndim == 2 && indices_t->ndim == 1 && dim == 0) {
                        /* Embedding-style: input [V, D], indices [N], out [N, D] */
                        size_t V = (size_t)input_t->shape[0];
                        size_t D = (size_t)input_t->shape[1];
                        size_t N = (size_t)indices_t->numel;
                        for (size_t i = 0; i < N; i++) {
                            int row = (int)idx_data[i];
                            if (row >= 0 && row < (int)V) {
                                for (size_t d = 0; d < D; d++) {
                                    g1_data[row * D + d] += out_grad[i * D + d];
                                }
                            }
                        }
                    } else {
                        /* Generic N-dim gather backward: scatter grad_output into grad_input.
                         * out[i0,..,i_{dim-1}, j, i_{dim+1},...] = input[i0,..., idx[i0,...,j,...], ...]
                         * So grad_input[..., idx[i...], ...] += grad_out[i...] */
                        if (input_t->ndim >= 1 && indices_t->ndim >= 1) {
                            size_t idx_numel = indices_t->numel;
                            if (dim < 0 || dim >= input_t->ndim) dim = input_t->ndim - 1;
                            /* Compute strides for input and indices */
                            size_t in_stride_dim = 1;
                            for (int sd = dim + 1; sd < input_t->ndim; sd++)
                                in_stride_dim *= (size_t)input_t->shape[sd];
                            size_t idx_stride_dim = 1;
                            for (int sd = dim + 1; sd < indices_t->ndim; sd++)
                                idx_stride_dim *= (size_t)indices_t->shape[sd];
                            size_t outer = 1;
                            for (int sd = 0; sd < dim && sd < indices_t->ndim; sd++)
                                outer *= (size_t)indices_t->shape[sd];
                            size_t inner = idx_stride_dim;
                            int idx_dim_size = (dim < indices_t->ndim) ? indices_t->shape[dim] : 1;
                            size_t flat = 0;
                            for (size_t o = 0; o < outer; o++) {
                                for (int dj = 0; dj < idx_dim_size; dj++) {
                                    for (size_t iv = 0; iv < inner; iv++, flat++) {
                                        if (flat >= idx_numel) goto gather_ndim_done;
                                        int idx_val = (int)idx_data[flat];
                                        if (idx_val < 0 || idx_val >= input_t->shape[dim]) continue;
                                        size_t in_flat = o * (size_t)input_t->shape[dim] * in_stride_dim
                                                       + (size_t)idx_val * in_stride_dim + iv;
                                        if (in_flat < input_t->numel)
                                            g1_data[in_flat] += out_grad[flat];
                                    }
                                }
                            }
                            gather_ndim_done:;
                        }
                    }
                }
            }
        }
        break;
    }

    /* ── Unary math ops ─────────────────────────────────────── */

    case UOP_COS:
        // d(cos(x))/dx = -sin(x)
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data, *x = (float*)in1->data;
                for (size_t i = 0; i < out_numel; i++)
                    g1d[i % in1->numel] += out_grad[i] * (-sinf(x[i % in1->numel]));
            }
        }
        break;

    case UOP_TAN:
        // d(tan(x))/dx = sec²(x) = 1/cos²(x)
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data, *x = (float*)in1->data;
                for (size_t i = 0; i < out_numel; i++) {
                    float c = cosf(x[i % in1->numel]);
                    g1d[i % in1->numel] += out_grad[i] / (c * c + 1e-12f);
                }
            }
        }
        break;

    case UOP_ABS:
        // d(|x|)/dx = sign(x)  (0 at x=0)
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data, *x = (float*)in1->data;
                for (size_t i = 0; i < out_numel; i++) {
                    float v = x[i % in1->numel];
                    g1d[i % in1->numel] += out_grad[i] * (v > 0.f ? 1.f : v < 0.f ? -1.f : 0.f);
                }
            }
        }
        break;

    case UOP_CLAMP: {
        // 1 inside [min,max], 0 outside
        ClampParams* cp = (ClampParams*)node->params;
        float lo = cp ? cp->min_val : -1e38f, hi = cp ? cp->max_val : 1e38f;
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data, *x = (float*)in1->data;
                for (size_t i = 0; i < out_numel; i++) {
                    float v = x[i % in1->numel];
                    g1d[i % in1->numel] += out_grad[i] * ((v > lo && v < hi) ? 1.f : 0.f);
                }
            }
        }
        break;
    }

    case UOP_MINIMUM:
        // min(a,b): grad to a where a<=b, grad to b where b<a
        if (in1 && in2 && in1->data && in2->data) {
            float* d1 = (float*)in1->data, *d2 = (float*)in2->data;
            if (in1->requires_grad) {
                Tensor* g1 = ensure_grad(in1);
                if (g1 && g1->data) {
                    float* g1d = (float*)g1->data;
                    for (size_t i = 0; i < out_numel; i++)
                        if (d1[i % in1->numel] <= d2[i % in2->numel])
                            g1d[i % in1->numel] += out_grad[i];
                }
            }
            if (in2->requires_grad) {
                Tensor* g2 = ensure_grad(in2);
                if (g2 && g2->data) {
                    float* g2d = (float*)g2->data;
                    for (size_t i = 0; i < out_numel; i++)
                        if (d2[i % in2->numel] < d1[i % in1->numel])
                            g2d[i % in2->numel] += out_grad[i];
                }
            }
        }
        break;

    case UOP_RSQRT:
        // d/dx[x^(-1/2)] = -0.5 * x^(-3/2) = -0.5 * out^3
        if (in1 && in1->requires_grad && out->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data, *od = (float*)out->data;
                for (size_t i = 0; i < out_numel; i++) {
                    float r = od[i];
                    g1d[i % in1->numel] += out_grad[i] * (-0.5f * r * r * r);
                }
            }
        }
        break;

    case UOP_LOG2:
        // d(log2(x))/dx = 1 / (x * ln(2))
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data, *x = (float*)in1->data;
                const float ln2 = 0.6931471805599453f;
                for (size_t i = 0; i < out_numel; i++)
                    g1d[i % in1->numel] += out_grad[i] / ((x[i % in1->numel] + 1e-8f) * ln2);
            }
        }
        break;

    case UOP_LOG10:
        // d(log10(x))/dx = 1 / (x * ln(10))
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data, *x = (float*)in1->data;
                const float ln10 = 2.302585092994046f;
                for (size_t i = 0; i < out_numel; i++)
                    g1d[i % in1->numel] += out_grad[i] / ((x[i % in1->numel] + 1e-8f) * ln10);
            }
        }
        break;

    case UOP_EXP2:
        // d(2^x)/dx = 2^x * ln(2)
        if (in1 && in1->requires_grad && out->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data, *od = (float*)out->data;
                const float ln2 = 0.6931471805599453f;
                for (size_t i = 0; i < out_numel; i++)
                    g1d[i % in1->numel] += out_grad[i] * od[i] * ln2;
            }
        }
        break;

    case UOP_ASIN:
        // d(asin(x))/dx = 1/sqrt(1-x²)
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data, *x = (float*)in1->data;
                for (size_t i = 0; i < out_numel; i++) {
                    float v = x[i % in1->numel];
                    g1d[i % in1->numel] += out_grad[i] / sqrtf(fmaxf(1.f - v*v, 1e-12f));
                }
            }
        }
        break;

    case UOP_ACOS:
        // d(acos(x))/dx = -1/sqrt(1-x²)
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data, *x = (float*)in1->data;
                for (size_t i = 0; i < out_numel; i++) {
                    float v = x[i % in1->numel];
                    g1d[i % in1->numel] -= out_grad[i] / sqrtf(fmaxf(1.f - v*v, 1e-12f));
                }
            }
        }
        break;

    case UOP_ATAN:
        // d(atan(x))/dx = 1/(1+x²)
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data, *x = (float*)in1->data;
                for (size_t i = 0; i < out_numel; i++) {
                    float v = x[i % in1->numel];
                    g1d[i % in1->numel] += out_grad[i] / (1.f + v*v);
                }
            }
        }
        break;

    case UOP_SINH:
        // d(sinh(x))/dx = cosh(x)
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data, *x = (float*)in1->data;
                for (size_t i = 0; i < out_numel; i++)
                    g1d[i % in1->numel] += out_grad[i] * coshf(x[i % in1->numel]);
            }
        }
        break;

    case UOP_COSH:
        // d(cosh(x))/dx = sinh(x)
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data, *x = (float*)in1->data;
                for (size_t i = 0; i < out_numel; i++)
                    g1d[i % in1->numel] += out_grad[i] * sinhf(x[i % in1->numel]);
            }
        }
        break;

    case UOP_ASINH:
        // d(asinh(x))/dx = 1/sqrt(x²+1)
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data, *x = (float*)in1->data;
                for (size_t i = 0; i < out_numel; i++) {
                    float v = x[i % in1->numel];
                    g1d[i % in1->numel] += out_grad[i] / sqrtf(v*v + 1.f);
                }
            }
        }
        break;

    case UOP_ACOSH:
        // d(acosh(x))/dx = 1/sqrt(x²-1)
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data, *x = (float*)in1->data;
                for (size_t i = 0; i < out_numel; i++) {
                    float v = x[i % in1->numel];
                    g1d[i % in1->numel] += out_grad[i] / sqrtf(fmaxf(v*v - 1.f, 1e-12f));
                }
            }
        }
        break;

    case UOP_ATANH:
        // d(atanh(x))/dx = 1/(1-x²)
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data, *x = (float*)in1->data;
                for (size_t i = 0; i < out_numel; i++) {
                    float v = x[i % in1->numel];
                    g1d[i % in1->numel] += out_grad[i] / fmaxf(1.f - v*v, 1e-12f);
                }
            }
        }
        break;

    case UOP_ERF:
        // d(erf(x))/dx = 2/sqrt(π) * exp(-x²)
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data, *x = (float*)in1->data;
                const float two_over_sqrtpi = 1.1283791670955126f;
                for (size_t i = 0; i < out_numel; i++) {
                    float v = x[i % in1->numel];
                    g1d[i % in1->numel] += out_grad[i] * two_over_sqrtpi * expf(-v*v);
                }
            }
        }
        break;

    case UOP_ERFC:
        // d(erfc(x))/dx = -2/sqrt(π) * exp(-x²)
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data, *x = (float*)in1->data;
                const float two_over_sqrtpi = 1.1283791670955126f;
                for (size_t i = 0; i < out_numel; i++) {
                    float v = x[i % in1->numel];
                    g1d[i % in1->numel] -= out_grad[i] * two_over_sqrtpi * expf(-v*v);
                }
            }
        }
        break;

    case UOP_LOGADDEXP:
        // d/da = exp(a)/(exp(a)+exp(b)), d/db = exp(b)/(exp(a)+exp(b))
        if (in1 && in2 && in1->data && in2->data) {
            float* d1 = (float*)in1->data, *d2 = (float*)in2->data;
            if (in1->requires_grad) {
                Tensor* g1 = ensure_grad(in1);
                if (g1 && g1->data) {
                    float* g1d = (float*)g1->data;
                    for (size_t i = 0; i < out_numel; i++) {
                        float a = d1[i%in1->numel], b = d2[i%in2->numel];
                        float ea = expf(a - fmaxf(a,b)), eb = expf(b - fmaxf(a,b));
                        g1d[i%in1->numel] += out_grad[i] * ea / (ea + eb + 1e-12f);
                    }
                }
            }
            if (in2->requires_grad) {
                Tensor* g2 = ensure_grad(in2);
                if (g2 && g2->data) {
                    float* g2d = (float*)g2->data;
                    for (size_t i = 0; i < out_numel; i++) {
                        float a = d1[i%in1->numel], b = d2[i%in2->numel];
                        float ea = expf(a - fmaxf(a,b)), eb = expf(b - fmaxf(a,b));
                        g2d[i%in2->numel] += out_grad[i] * eb / (ea + eb + 1e-12f);
                    }
                }
            }
        }
        break;

    /* ── View / shape ops (identity backward) ────────────────── */

    case UOP_RESHAPE:
    case UOP_FLATTEN:
    case UOP_UNFLATTEN:
    case UOP_STRIDE:
        // Same element count, different layout: pass gradient straight through.
        if (in1 && in1->requires_grad) {
            accumulate_grad(in1, out_grad, out_numel);
        }
        break;

    /* ── Slice: scatter gradient into the original region ────── */

    case UOP_SLICE: {
        if (in1 && in1->requires_grad) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data;
                SliceParams* sp = (SliceParams*)node->params;
                if (!sp || in1->ndim == 0) {
                    accumulate_grad(in1, out_grad, out_numel);
                } else if (in1->ndim == 1) {
                    int st = sp->start ? sp->start[0] : 0;
                    int step = (sp->step && sp->step[0] != 0) ? sp->step[0] : 1;
                    for (size_t i = 0; i < out_numel; i++)
                        g1d[st + (int)i * step] += out_grad[i];
                } else if (in1->ndim == 2) {
                    int r0 = sp->start ? sp->start[0] : 0;
                    int c0 = sp->start ? sp->start[1] : 0;
                    int rs = (sp->step && sp->step[0]) ? sp->step[0] : 1;
                    int cs = (sp->step && sp->step[1]) ? sp->step[1] : 1;
                    int out_cols = out->shape[1];
                    int in_cols  = in1->shape[1];
                    for (int r = 0; r < out->shape[0]; r++)
                        for (int c = 0; c < out_cols; c++)
                            g1d[(r0 + r*rs) * in_cols + c0 + c*cs] +=
                                out_grad[r * out_cols + c];
                } else {
                    accumulate_grad(in1, out_grad, out_numel);
                }
            }
        }
        break;
    }

    /* ── Pad (constant): crop the padded gradient back ─────── */

    case UOP_PAD: {
        if (in1 && in1->requires_grad) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data;
                PadParams* pp = (PadParams*)node->params;
                if (!pp || pp->mode != PAD_CONSTANT) {
                    // Reflect / replicate: full accumulate as fallback
                    accumulate_grad(in1, out_grad, out_numel);
                } else if (in1->ndim == 1) {
                    int pb = pp->pad_widths[0];
                    int len = (int)in1->numel;
                    for (int i = 0; i < len; i++)
                        g1d[i] += out_grad[pb + i];
                } else if (in1->ndim == 2) {
                    int pr = pp->pad_widths[0], pc = pp->pad_widths[2];
                    int in_r = in1->shape[0], in_c = in1->shape[1];
                    int out_c = out->shape[1];
                    for (int r = 0; r < in_r; r++)
                        for (int c = 0; c < in_c; c++)
                            g1d[r * in_c + c] +=
                                out_grad[(r + pr) * out_c + (c + pc)];
                } else {
                    accumulate_grad(in1, out_grad, out_numel);
                }
            }
        }
        break;
    }

    /* ── Cat: split gradient along cat dim back to each input ── */

    case UOP_CAT: {
        CatParams* cp = (CatParams*)node->params;
        int cat_dim = cp ? cp->dim : 0;

        if (in1 && in1->ndim == 1) {
            // 1-D: simple offset split
            size_t offset = 0;
            for (int t = 0; t < node->num_inputs; t++) {
                Tensor* inp = node->inputs[t];
                if (inp && inp->requires_grad) {
                    Tensor* gi = ensure_grad(inp);
                    if (gi && gi->data) {
                        float* gid = (float*)gi->data;
                        for (size_t k = 0; k < inp->numel; k++)
                            gid[k] += out_grad[offset + k];
                    }
                }
                if (inp) offset += inp->numel;
            }
        } else if (out->ndim == 2) {
            if (cat_dim == 0) {
                size_t offset = 0;
                for (int t = 0; t < node->num_inputs; t++) {
                    Tensor* inp = node->inputs[t];
                    if (inp && inp->requires_grad) {
                        Tensor* gi = ensure_grad(inp);
                        if (gi && gi->data) {
                            float* gid = (float*)gi->data;
                            for (size_t k = 0; k < inp->numel; k++)
                                gid[k] += out_grad[offset + k];
                        }
                    }
                    if (inp) offset += inp->numel;
                }
            } else { // cat_dim == 1
                int out_cols = out->shape[1], rows = out->shape[0];
                int col_offset = 0;
                for (int t = 0; t < node->num_inputs; t++) {
                    Tensor* inp = node->inputs[t];
                    if (inp && inp->requires_grad) {
                        Tensor* gi = ensure_grad(inp);
                        if (gi && gi->data) {
                            float* gid = (float*)gi->data;
                            int tcols = inp->shape[1];
                            for (int r = 0; r < rows; r++)
                                for (int c = 0; c < tcols; c++)
                                    gid[r*tcols + c] +=
                                        out_grad[r*out_cols + col_offset + c];
                        }
                    }
                    if (inp && inp->ndim >= 2) col_offset += inp->shape[1];
                }
            }
        }
        break;
    }

    /* ── Stack: un-stack gradient along stack dim ───────────── */

    case UOP_STACK: {
        StackParams* sp = (StackParams*)node->params;
        int stack_dim = sp ? sp->dim : 0;
        int num_t = node->num_inputs;

        if (stack_dim == 0) {
            size_t per = out_numel / (size_t)(num_t > 0 ? num_t : 1);
            for (int t = 0; t < num_t; t++) {
                Tensor* inp = node->inputs[t];
                if (inp && inp->requires_grad) {
                    Tensor* gi = ensure_grad(inp);
                    if (gi && gi->data) {
                        float* gid = (float*)gi->data;
                        for (size_t k = 0; k < per; k++)
                            gid[k] += out_grad[(size_t)t * per + k];
                    }
                }
            }
        } else if (out->ndim == 2) {
            // stack along dim 1: output [N, T], each input [N]
            int rows = out->shape[0];
            for (int t = 0; t < num_t; t++) {
                Tensor* inp = node->inputs[t];
                if (inp && inp->requires_grad) {
                    Tensor* gi = ensure_grad(inp);
                    if (gi && gi->data) {
                        float* gid = (float*)gi->data;
                        for (int r = 0; r < rows; r++)
                            gid[r] += out_grad[r * num_t + t];
                    }
                }
            }
        }
        break;
    }

    /* ── TRIU / TRIL: apply same mask to gradient ─────────── */

    case UOP_TRIU: {
        if (in1 && in1->requires_grad) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data;
                TriParams* tp = (TriParams*)node->params;
                int diag = tp ? tp->diagonal : 0;
                int rows = in1->shape[in1->ndim-2], cols = in1->shape[in1->ndim-1];
                size_t batch = out_numel / (size_t)(rows * cols);
                for (size_t b = 0; b < batch; b++)
                    for (int r = 0; r < rows; r++)
                        for (int c = 0; c < cols; c++) {
                            size_t idx = b*(size_t)(rows*cols) + (size_t)(r*cols+c);
                            if (c >= r + diag) g1d[idx] += out_grad[idx];
                        }
            }
        }
        break;
    }

    case UOP_TRIL: {
        if (in1 && in1->requires_grad) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data;
                TriParams* tp = (TriParams*)node->params;
                int diag = tp ? tp->diagonal : 0;
                int rows = in1->shape[in1->ndim-2], cols = in1->shape[in1->ndim-1];
                size_t batch = out_numel / (size_t)(rows * cols);
                for (size_t b = 0; b < batch; b++)
                    for (int r = 0; r < rows; r++)
                        for (int c = 0; c < cols; c++) {
                            size_t idx = b*(size_t)(rows*cols) + (size_t)(r*cols+c);
                            if (c <= r + diag) g1d[idx] += out_grad[idx];
                        }
            }
        }
        break;
    }

    /* ── ROLL: reverse-roll the gradient ─────────────────── */

    case UOP_ROLL: {
        if (in1 && in1->requires_grad) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data;
                RollParams* rp = (RollParams*)node->params;
                int shift = rp ? rp->shift : 0;
                size_t n = in1->numel;
                // Reverse roll: shift by -shift
                int s = (int)((n + (size_t)((-shift) % (int)n)) % n);
                for (size_t i = 0; i < out_numel; i++)
                    g1d[(i + (size_t)s) % n] += out_grad[i];
            }
        }
        break;
    }

    /* ── TILE: sum tiled copies back ──────────────────────── */

    case UOP_TILE: {
        if (in1 && in1->requires_grad) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data;
                size_t n = in1->numel;
                for (size_t i = 0; i < out_numel; i++)
                    g1d[i % n] += out_grad[i];
            }
        }
        break;
    }

    /* ── REPEAT_INTERLEAVE: sum repeated elements ─────────── */

    case UOP_REPEAT_INTERLEAVE: {
        if (in1 && in1->requires_grad) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data;
                RepeatInterleaveParams* rp = (RepeatInterleaveParams*)node->params;
                int reps = rp ? rp->repeats : 1;
                if (reps < 1) reps = 1;
                for (size_t i = 0; i < out_numel; i++)
                    g1d[(i / (size_t)reps) % in1->numel] += out_grad[i];
            }
        }
        break;
    }

    /* ── CUMSUM: reverse cumsum of gradient ───────────────── */

    case UOP_CUMSUM: {
        if (in1 && in1->requires_grad) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data;
                CumsumParams* cp = (CumsumParams*)node->params;
                int cdim = cp ? cp->dim : 0;
                if (cdim < 0) cdim += in1->ndim;

                /* Reverse cumsum along cdim: g_in[i] = sum(g_out[i..end]) along that dim.
                 * Use outer × dim × inner layout for arbitrary ndim. */
                int ndim = in1->ndim;
                size_t outer = 1;
                for (int d = 0; d < cdim; d++) outer *= (size_t)in1->shape[d];
                size_t dim_size = (size_t)in1->shape[cdim];
                size_t inner = 1;
                for (int d = cdim + 1; d < ndim; d++) inner *= (size_t)in1->shape[d];

                for (size_t o = 0; o < outer; o++) {
                    for (size_t iv = 0; iv < inner; iv++) {
                        float acc = 0.f;
                        for (int i = (int)dim_size - 1; i >= 0; i--) {
                            size_t idx = (o * dim_size + (size_t)i) * inner + iv;
                            acc += out_grad[idx];
                            g1d[idx] += acc;
                        }
                    }
                }
            }
        }
        break;
    }

    /* ── MAX_REDUCE: gradient to argmax position ─────────── */

    case UOP_MAX_REDUCE: {
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data, *x = (float*)in1->data;
                ReduceParams* rp = (ReduceParams*)node->params;

                if (!rp || rp->num_dims == 0) {
                    /* Global max */
                    size_t am = 0;
                    for (size_t i = 1; i < in1->numel; i++)
                        if (x[i] > x[am]) am = i;
                    g1d[am] += out_grad[0];
                } else {
                    /* Dim-specific: outer × reduce × inner layout */
                    int rd = rp->dims[0]; if (rd < 0) rd += in1->ndim;
                    size_t outer = 1;
                    for (int d = 0; d < rd; d++) outer *= (size_t)in1->shape[d];
                    size_t reduce_size = (size_t)in1->shape[rd];
                    size_t inner = 1;
                    for (int d = rd + 1; d < in1->ndim; d++) inner *= (size_t)in1->shape[d];

                    for (size_t o = 0; o < outer; o++) {
                        for (size_t iv = 0; iv < inner; iv++) {
                            /* Find argmax along reduce dim */
                            size_t am = 0;
                            float best = x[o * reduce_size * inner + iv];
                            for (size_t r = 1; r < reduce_size; r++) {
                                float v = x[(o * reduce_size + r) * inner + iv];
                                if (v > best) { best = v; am = r; }
                            }
                            size_t in_idx  = (o * reduce_size + am) * inner + iv;
                            size_t out_idx = o * inner + iv;
                            g1d[in_idx] += out_grad[out_idx];
                        }
                    }
                }
            }
        }
        break;
    }

    /* ── MIN_REDUCE: gradient to argmin position ─────────── */

    case UOP_MIN_REDUCE: {
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data, *x = (float*)in1->data;
                ReduceParams* rp = (ReduceParams*)node->params;

                if (!rp || rp->num_dims == 0) {
                    size_t am = 0;
                    for (size_t i = 1; i < in1->numel; i++)
                        if (x[i] < x[am]) am = i;
                    g1d[am] += out_grad[0];
                } else {
                    int rd = rp->dims[0]; if (rd < 0) rd += in1->ndim;
                    size_t outer = 1;
                    for (int d = 0; d < rd; d++) outer *= (size_t)in1->shape[d];
                    size_t reduce_size = (size_t)in1->shape[rd];
                    size_t inner = 1;
                    for (int d = rd + 1; d < in1->ndim; d++) inner *= (size_t)in1->shape[d];

                    for (size_t o = 0; o < outer; o++) {
                        for (size_t iv = 0; iv < inner; iv++) {
                            size_t am = 0;
                            float best = x[o * reduce_size * inner + iv];
                            for (size_t r = 1; r < reduce_size; r++) {
                                float v = x[(o * reduce_size + r) * inner + iv];
                                if (v < best) { best = v; am = r; }
                            }
                            size_t in_idx  = (o * reduce_size + am) * inner + iv;
                            size_t out_idx = o * inner + iv;
                            g1d[in_idx] += out_grad[out_idx];
                        }
                    }
                }
            }
        }
        break;
    }

    /* ── LOGSUMEXP: gradient is softmax(x) ─────────────── */

    case UOP_LOGSUMEXP: {
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data, *x = (float*)in1->data;
                // Global logsumexp: softmax weights
                float mx = x[0];
                for (size_t i = 1; i < in1->numel; i++) if (x[i] > mx) mx = x[i];
                float sum_exp = 0.f;
                for (size_t i = 0; i < in1->numel; i++) sum_exp += expf(x[i] - mx);
                for (size_t i = 0; i < in1->numel; i++)
                    g1d[i] += out_grad[0] * expf(x[i] - mx) / (sum_exp + 1e-12f);
            }
        }
        break;
    }

    /* ── PROD: product rule ─────────────────────────────── */

    case UOP_PROD: {
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data, *x = (float*)in1->data;
                size_t n = in1->numel;
                // Compute prefix and suffix products to avoid division by zero
                float* prefix = (float*)calloc(n + 1, sizeof(float));
                float* suffix = (float*)calloc(n + 1, sizeof(float));
                if (prefix && suffix) {
                    prefix[0] = 1.f;
                    for (size_t i = 0; i < n; i++) prefix[i+1] = prefix[i] * x[i];
                    suffix[n] = 1.f;
                    for (int i = (int)n - 1; i >= 0; i--) suffix[i] = suffix[i+1] * x[i];
                    for (size_t i = 0; i < n; i++)
                        g1d[i] += out_grad[0] * prefix[i] * suffix[i+1];
                }
                free(prefix); free(suffix);
            }
        }
        break;
    }

    /* ── TRACE: gradient is scalar * identity ─────────── */

    case UOP_TRACE: {
        if (in1 && in1->requires_grad && in1->ndim == 2) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data;
                int n = in1->shape[0] < in1->shape[1] ? in1->shape[0] : in1->shape[1];
                for (int i = 0; i < n; i++)
                    g1d[i * in1->shape[1] + i] += out_grad[0];
            }
        }
        break;
    }

    /* ── DIAG: backward swaps 1D↔2D role ─────────────── */

    case UOP_DIAG: {
        if (in1 && in1->requires_grad) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data;
                DiagParams* dp = (DiagParams*)node->params;
                int diag = dp ? dp->offset : 0;
                if (in1->ndim == 1) {
                    // Forward: 1D→2D. Backward: extract diagonal of out_grad.
                    int n = out->shape[0];
                    for (int i = 0; i < (int)in1->numel; i++) {
                        int r = (diag >= 0) ? i : i - diag;
                        int c = (diag >= 0) ? i + diag : i;
                        if (r >= 0 && r < n && c >= 0 && c < n)
                            g1d[i] += out_grad[r * n + c];
                    }
                } else if (in1->ndim == 2) {
                    // Forward: 2D→1D. Backward: scatter into diagonal.
                    int rows = in1->shape[0], cols = in1->shape[1];
                    for (size_t k = 0; k < out_numel; k++) {
                        int r = (diag >= 0) ? (int)k : (int)k - diag;
                        int c = (diag >= 0) ? (int)k + diag : (int)k;
                        if (r >= 0 && r < rows && c >= 0 && c < cols)
                            g1d[r * cols + c] += out_grad[k];
                    }
                }
            }
        }
        break;
    }

    /* ── VAR / STD: variance / std-dev backward ─────── */

    case UOP_VAR: {
        if (in1 && in1->requires_grad && in1->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data, *x = (float*)in1->data;
                size_t n = in1->numel;
                float mean = 0.f;
                for (size_t i = 0; i < n; i++) mean += x[i];
                mean /= (float)n;
                // biased: d var / dx_i = 2*(x_i - mean) / n
                float scale = 2.f / (float)n;
                for (size_t i = 0; i < n; i++)
                    g1d[i] += out_grad[0] * scale * (x[i] - mean);
            }
        }
        break;
    }

    case UOP_STD: {
        if (in1 && in1->requires_grad && in1->data && out->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data, *x = (float*)in1->data;
                float std_val = ((float*)out->data)[0];
                size_t n = in1->numel;
                float mean = 0.f;
                for (size_t i = 0; i < n; i++) mean += x[i];
                mean /= (float)n;
                float denom = (float)n * fmaxf(std_val, 1e-8f);
                for (size_t i = 0; i < n; i++)
                    g1d[i] += out_grad[0] * (x[i] - mean) / denom;
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

    /* ── AVGPOOL2D: distribute grad uniformly over window ───── */

    case UOP_AVGPOOL2D: {
        if (!in1 || !in1->requires_grad || !in1->data || !node->params)
            break;
        Pool2DParams* p = (Pool2DParams*)node->params;
        Tensor* g1 = ensure_grad(in1);
        if (!g1 || !g1->data)
            break;
        float* g1d = (float*)g1->data;
        int batch = in1->shape[0], channels = in1->shape[1];
        int in_h = in1->shape[2], in_w = in1->shape[3];
        int out_h = out->shape[2], out_w = out->shape[3];
        int kh = p->kernel_size[0], kw = p->kernel_size[1];
        int sh = p->stride[0] > 0 ? p->stride[0] : kh;
        int sw = p->stride[1] > 0 ? p->stride[1] : kw;
        int ph = p->padding[0], pw = p->padding[1];
        int dh = p->dilation[0] > 0 ? p->dilation[0] : 1;
        int dw = p->dilation[1] > 0 ? p->dilation[1] : 1;

        for (int n = 0; n < batch; n++) {
            for (int c = 0; c < channels; c++) {
                for (int oh = 0; oh < out_h; oh++) {
                    for (int ow2 = 0; ow2 < out_w; ow2++) {
                        int count = 0;
                        int hstart = oh * sh - ph;
                        int wstart = ow2 * sw - pw;
                        for (int fh = 0; fh < kh; fh++) {
                            int ih = hstart + fh * dh;
                            for (int fw = 0; fw < kw; fw++) {
                                int iw2 = wstart + fw * dw;
                                if (ih >= 0 && ih < in_h && iw2 >= 0 && iw2 < in_w)
                                    count++;
                                else if (p->count_include_pad)
                                    count++;
                            }
                        }
                        if (count == 0) continue;
                        size_t out_idx = (((size_t)n * channels + c) * out_h + oh) * out_w + ow2;
                        float go = out_grad[out_idx] / (float)count;
                        for (int fh = 0; fh < kh; fh++) {
                            int ih = hstart + fh * dh;
                            for (int fw = 0; fw < kw; fw++) {
                                int iw2 = wstart + fw * dw;
                                if (ih >= 0 && ih < in_h && iw2 >= 0 && iw2 < in_w)
                                    g1d[(((size_t)n * channels + c) * in_h + ih) * in_w + iw2] += go;
                            }
                        }
                    }
                }
            }
        }
        break;
    }

    /* ── MAXPOOL2D: grad to argmax position (recompute from input) ── */

    case UOP_MAXPOOL2D: {
        if (!in1 || !in1->requires_grad || !in1->data || !node->params)
            break;
        Pool2DParams* p = (Pool2DParams*)node->params;
        Tensor* g1 = ensure_grad(in1);
        if (!g1 || !g1->data)
            break;
        float* g1d = (float*)g1->data;
        float* in_data = (float*)in1->data;
        int batch = in1->shape[0], channels = in1->shape[1];
        int in_h = in1->shape[2], in_w = in1->shape[3];
        int out_h = out->shape[2], out_w = out->shape[3];
        int kh = p->kernel_size[0], kw = p->kernel_size[1];
        int sh = p->stride[0] > 0 ? p->stride[0] : kh;
        int sw = p->stride[1] > 0 ? p->stride[1] : kw;
        int ph = p->padding[0], pw = p->padding[1];
        int dh = p->dilation[0] > 0 ? p->dilation[0] : 1;
        int dw = p->dilation[1] > 0 ? p->dilation[1] : 1;

        for (int n = 0; n < batch; n++) {
            for (int c = 0; c < channels; c++) {
                for (int oh = 0; oh < out_h; oh++) {
                    for (int ow2 = 0; ow2 < out_w; ow2++) {
                        int hstart = oh * sh - ph;
                        int wstart = ow2 * sw - pw;
                        /* Recompute argmax from input */
                        float max_val = -INFINITY;
                        int max_ih = -1, max_iw = -1;
                        for (int fh = 0; fh < kh; fh++) {
                            int ih = hstart + fh * dh;
                            for (int fw = 0; fw < kw; fw++) {
                                int iw2 = wstart + fw * dw;
                                if (ih < 0 || ih >= in_h || iw2 < 0 || iw2 >= in_w)
                                    continue;
                                float v = in_data[(((size_t)n * channels + c) * in_h + ih) * in_w + iw2];
                                if (v > max_val) {
                                    max_val = v;
                                    max_ih = ih;
                                    max_iw = iw2;
                                }
                            }
                        }
                        if (max_ih >= 0 && max_iw >= 0) {
                            size_t out_idx = (((size_t)n * channels + c) * out_h + oh) * out_w + ow2;
                            g1d[(((size_t)n * channels + c) * in_h + max_ih) * in_w + max_iw] += out_grad[out_idx];
                        }
                    }
                }
            }
        }
        break;
    }

    /* ── CONV3D: naive 3D convolution backward ────────────────── */

    case UOP_CONV3D: {
        if (!in1 || !in2 || !node->params)
            break;
        if (!in1->data || !in2->data)
            break;
        Conv3DParams* p = (Conv3DParams*)node->params;
        int batch = in1->shape[0], in_ch = in1->shape[1];
        int in_d = in1->shape[2], in_h = in1->shape[3], in_w = in1->shape[4];
        int out_ch = in2->shape[0];
        int kd = in2->shape[2], kh = in2->shape[3], kw2 = in2->shape[4];
        int out_d = out->shape[2], out_h = out->shape[3], out_w = out->shape[4];

        if (in1->requires_grad) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data;
                float* w_data = (float*)in2->data;
                for (int b = 0; b < batch; b++)
                    for (int ic = 0; ic < in_ch; ic++)
                        for (int id2 = 0; id2 < in_d; id2++)
                            for (int ih2 = 0; ih2 < in_h; ih2++)
                                for (int iw3 = 0; iw3 < in_w; iw3++) {
                                    float s = 0.f;
                                    for (int oc = 0; oc < out_ch; oc++)
                                        for (int fkd = 0; fkd < kd; fkd++) {
                                            int od2 = id2 - fkd * p->dilation[0] + p->padding[0];
                                            if (od2 < 0 || od2 % p->stride[0] != 0) continue;
                                            od2 /= p->stride[0];
                                            if (od2 >= out_d) continue;
                                            for (int fkh = 0; fkh < kh; fkh++) {
                                                int oh2 = ih2 - fkh * p->dilation[1] + p->padding[1];
                                                if (oh2 < 0 || oh2 % p->stride[1] != 0) continue;
                                                oh2 /= p->stride[1];
                                                if (oh2 >= out_h) continue;
                                                for (int fkw = 0; fkw < kw2; fkw++) {
                                                    int ow3 = iw3 - fkw * p->dilation[2] + p->padding[2];
                                                    if (ow3 < 0 || ow3 % p->stride[2] != 0) continue;
                                                    ow3 /= p->stride[2];
                                                    if (ow3 >= out_w) continue;
                                                    size_t og_idx = ((((size_t)b * out_ch + oc) * out_d + od2) * out_h + oh2) * out_w + ow3;
                                                    size_t w_idx  = ((((size_t)oc * in_ch + ic) * kd + fkd) * kh + fkh) * kw2 + fkw;
                                                    s += out_grad[og_idx] * w_data[w_idx];
                                                }
                                            }
                                        }
                                    g1d[((((size_t)b * in_ch + ic) * in_d + id2) * in_h + ih2) * in_w + iw3] += s;
                                }
            }
        }

        if (in2->requires_grad) {
            Tensor* g2 = ensure_grad(in2);
            if (g2 && g2->data) {
                float* g2d = (float*)g2->data;
                float* in_data = (float*)in1->data;
                for (int oc = 0; oc < out_ch; oc++)
                    for (int ic = 0; ic < in_ch; ic++)
                        for (int fkd = 0; fkd < kd; fkd++)
                            for (int fkh = 0; fkh < kh; fkh++)
                                for (int fkw = 0; fkw < kw2; fkw++) {
                                    float s = 0.f;
                                    for (int b = 0; b < batch; b++)
                                        for (int od2 = 0; od2 < out_d; od2++)
                                            for (int oh2 = 0; oh2 < out_h; oh2++)
                                                for (int ow3 = 0; ow3 < out_w; ow3++) {
                                                    int id2 = od2 * p->stride[0] - p->padding[0] + fkd * p->dilation[0];
                                                    int ih2 = oh2 * p->stride[1] - p->padding[1] + fkh * p->dilation[1];
                                                    int iw3 = ow3 * p->stride[2] - p->padding[2] + fkw * p->dilation[2];
                                                    if (id2 < 0 || id2 >= in_d || ih2 < 0 || ih2 >= in_h || iw3 < 0 || iw3 >= in_w)
                                                        continue;
                                                    size_t og_idx = ((((size_t)b * out_ch + oc) * out_d + od2) * out_h + oh2) * out_w + ow3;
                                                    size_t in_idx = ((((size_t)b * in_ch + ic) * in_d + id2) * in_h + ih2) * in_w + iw3;
                                                    s += out_grad[og_idx] * in_data[in_idx];
                                                }
                                    g2d[((((size_t)oc * in_ch + ic) * kd + fkd) * kh + fkh) * kw2 + fkw] += s;
                                }
            }
        }

        /* Bias grad: sum over batch, depth, height, width */
        if (node->num_inputs >= 3 && node->inputs[2] && node->inputs[2]->requires_grad) {
            Tensor* bias = node->inputs[2];
            Tensor* gb = ensure_grad(bias);
            if (gb && gb->data) {
                float* gbd = (float*)gb->data;
                for (int oc = 0; oc < out_ch; oc++) {
                    float s = 0.f;
                    for (int b = 0; b < batch; b++)
                        for (int od2 = 0; od2 < out_d; od2++)
                            for (int oh2 = 0; oh2 < out_h; oh2++)
                                for (int ow3 = 0; ow3 < out_w; ow3++)
                                    s += out_grad[((((size_t)b * out_ch + oc) * out_d + od2) * out_h + oh2) * out_w + ow3];
                    gbd[oc] += s;
                }
            }
        }
        break;
    }

    /* ── CONV_TRANSPOSE2D: backward is a regular convolution ─── */

    case UOP_CONV_TRANSPOSE2D: {
        if (!in1 || !in2 || !node->params)
            break;
        if (!in1->data || !in2->data)
            break;
        ConvTranspose2DParams* p = (ConvTranspose2DParams*)node->params;
        int batch = in1->shape[0], in_ch = in1->shape[1];
        int in_h = in1->shape[2], in_w = in1->shape[3];
        int out_ch = in2->shape[1];
        int kh = in2->shape[2], kw = in2->shape[3];
        int out_h = out->shape[2], out_w = out->shape[3];

        /* grad_input[b, ic, ih, iw] = sum_{oc, kh, kw} out_grad[b, oc, ih*s-p+kh, iw*s-p+kw] * weight[ic, oc, kh, kw] */
        if (in1->requires_grad) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data;
                float* w_data = (float*)in2->data;
                for (int b = 0; b < batch; b++)
                    for (int ic = 0; ic < in_ch; ic++)
                        for (int ih = 0; ih < in_h; ih++)
                            for (int iw2 = 0; iw2 < in_w; iw2++) {
                                float s = 0.f;
                                for (int oc = 0; oc < out_ch; oc++)
                                    for (int fkh = 0; fkh < kh; fkh++) {
                                        int oh = ih * p->stride[0] - p->padding[0] + fkh * p->dilation[0];
                                        if (oh < 0 || oh >= out_h) continue;
                                        for (int fkw = 0; fkw < kw; fkw++) {
                                            int ow2 = iw2 * p->stride[1] - p->padding[1] + fkw * p->dilation[1];
                                            if (ow2 < 0 || ow2 >= out_w) continue;
                                            size_t og_idx = (((size_t)b * out_ch + oc) * out_h + oh) * out_w + ow2;
                                            size_t w_idx  = (((size_t)ic * out_ch + oc) * kh + fkh) * kw + fkw;
                                            s += out_grad[og_idx] * w_data[w_idx];
                                        }
                                    }
                                g1d[(((size_t)b * in_ch + ic) * in_h + ih) * in_w + iw2] += s;
                            }
            }
        }

        /* grad_weight[ic, oc, kh, kw] = sum_{b, ih, iw} input[b, ic, ih, iw] * out_grad[b, oc, ih*s-p+kh, iw*s-p+kw] */
        if (in2->requires_grad) {
            Tensor* g2 = ensure_grad(in2);
            if (g2 && g2->data) {
                float* g2d = (float*)g2->data;
                float* in_data = (float*)in1->data;
                for (int ic = 0; ic < in_ch; ic++)
                    for (int oc = 0; oc < out_ch; oc++)
                        for (int fkh = 0; fkh < kh; fkh++)
                            for (int fkw = 0; fkw < kw; fkw++) {
                                float s = 0.f;
                                for (int b = 0; b < batch; b++)
                                    for (int ih = 0; ih < in_h; ih++) {
                                        int oh = ih * p->stride[0] - p->padding[0] + fkh * p->dilation[0];
                                        if (oh < 0 || oh >= out_h) continue;
                                        for (int iw2 = 0; iw2 < in_w; iw2++) {
                                            int ow2 = iw2 * p->stride[1] - p->padding[1] + fkw * p->dilation[1];
                                            if (ow2 < 0 || ow2 >= out_w) continue;
                                            size_t in_idx = (((size_t)b * in_ch + ic) * in_h + ih) * in_w + iw2;
                                            size_t og_idx = (((size_t)b * out_ch + oc) * out_h + oh) * out_w + ow2;
                                            s += in_data[in_idx] * out_grad[og_idx];
                                        }
                                    }
                                g2d[(((size_t)ic * out_ch + oc) * kh + fkh) * kw + fkw] += s;
                            }
            }
        }

        if (node->num_inputs >= 3 && node->inputs[2] && node->inputs[2]->requires_grad) {
            Tensor* bias = node->inputs[2];
            Tensor* gb = ensure_grad(bias);
            if (gb && gb->data) {
                float* gbd = (float*)gb->data;
                for (int oc = 0; oc < out_ch; oc++) {
                    float s = 0.f;
                    for (int b = 0; b < batch; b++)
                        for (int oh = 0; oh < out_h; oh++)
                            for (int ow2 = 0; ow2 < out_w; ow2++)
                                s += out_grad[(((size_t)b * out_ch + oc) * out_h + oh) * out_w + ow2];
                    gbd[oc] += s;
                }
            }
        }
        break;
    }

    /* ── CONV_TRANSPOSE3D: backward ─────────────────────────── */

    case UOP_CONV_TRANSPOSE3D: {
        if (!in1 || !in2 || !node->params)
            break;
        if (!in1->data || !in2->data)
            break;
        ConvTranspose3DParams* p = (ConvTranspose3DParams*)node->params;
        int batch = in1->shape[0], in_ch = in1->shape[1];
        int in_d = in1->shape[2], in_h = in1->shape[3], in_w = in1->shape[4];
        int out_ch = in2->shape[1];
        int kd = in2->shape[2], kh = in2->shape[3], kw2 = in2->shape[4];
        int out_d = out->shape[2], out_h = out->shape[3], out_w = out->shape[4];

        if (in1->requires_grad) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data;
                float* w_data = (float*)in2->data;
                for (int b = 0; b < batch; b++)
                    for (int ic = 0; ic < in_ch; ic++)
                        for (int id2 = 0; id2 < in_d; id2++)
                            for (int ih2 = 0; ih2 < in_h; ih2++)
                                for (int iw3 = 0; iw3 < in_w; iw3++) {
                                    float s = 0.f;
                                    for (int oc = 0; oc < out_ch; oc++)
                                        for (int fkd = 0; fkd < kd; fkd++) {
                                            int od2 = id2 * p->stride[0] - p->padding[0] + fkd * p->dilation[0];
                                            if (od2 < 0 || od2 >= out_d) continue;
                                            for (int fkh = 0; fkh < kh; fkh++) {
                                                int oh2 = ih2 * p->stride[1] - p->padding[1] + fkh * p->dilation[1];
                                                if (oh2 < 0 || oh2 >= out_h) continue;
                                                for (int fkw = 0; fkw < kw2; fkw++) {
                                                    int ow3 = iw3 * p->stride[2] - p->padding[2] + fkw * p->dilation[2];
                                                    if (ow3 < 0 || ow3 >= out_w) continue;
                                                    size_t og_idx = ((((size_t)b * out_ch + oc) * out_d + od2) * out_h + oh2) * out_w + ow3;
                                                    size_t w_idx  = ((((size_t)ic * out_ch + oc) * kd + fkd) * kh + fkh) * kw2 + fkw;
                                                    s += out_grad[og_idx] * w_data[w_idx];
                                                }
                                            }
                                        }
                                    g1d[((((size_t)b * in_ch + ic) * in_d + id2) * in_h + ih2) * in_w + iw3] += s;
                                }
            }
        }

        if (in2->requires_grad) {
            Tensor* g2 = ensure_grad(in2);
            if (g2 && g2->data) {
                float* g2d = (float*)g2->data;
                float* in_data = (float*)in1->data;
                for (int ic = 0; ic < in_ch; ic++)
                    for (int oc = 0; oc < out_ch; oc++)
                        for (int fkd = 0; fkd < kd; fkd++)
                            for (int fkh = 0; fkh < kh; fkh++)
                                for (int fkw = 0; fkw < kw2; fkw++) {
                                    float s = 0.f;
                                    for (int b = 0; b < batch; b++)
                                        for (int id2 = 0; id2 < in_d; id2++) {
                                            int od2 = id2 * p->stride[0] - p->padding[0] + fkd * p->dilation[0];
                                            if (od2 < 0 || od2 >= out_d) continue;
                                            for (int ih2 = 0; ih2 < in_h; ih2++) {
                                                int oh2 = ih2 * p->stride[1] - p->padding[1] + fkh * p->dilation[1];
                                                if (oh2 < 0 || oh2 >= out_h) continue;
                                                for (int iw3 = 0; iw3 < in_w; iw3++) {
                                                    int ow3 = iw3 * p->stride[2] - p->padding[2] + fkw * p->dilation[2];
                                                    if (ow3 < 0 || ow3 >= out_w) continue;
                                                    size_t in_idx = ((((size_t)b * in_ch + ic) * in_d + id2) * in_h + ih2) * in_w + iw3;
                                                    size_t og_idx = ((((size_t)b * out_ch + oc) * out_d + od2) * out_h + oh2) * out_w + ow3;
                                                    s += in_data[in_idx] * out_grad[og_idx];
                                                }
                                            }
                                        }
                                    g2d[((((size_t)ic * out_ch + oc) * kd + fkd) * kh + fkh) * kw2 + fkw] += s;
                                }
            }
        }

        if (node->num_inputs >= 3 && node->inputs[2] && node->inputs[2]->requires_grad) {
            Tensor* bias = node->inputs[2];
            Tensor* gb = ensure_grad(bias);
            if (gb && gb->data) {
                float* gbd = (float*)gb->data;
                int spatial = out_d * out_h * out_w;
                for (int oc = 0; oc < out_ch; oc++) {
                    float s = 0.f;
                    for (int b = 0; b < batch; b++) {
                        size_t base = ((size_t)b * out_ch + oc) * spatial;
                        for (int i = 0; i < spatial; i++)
                            s += out_grad[base + i];
                    }
                    gbd[oc] += s;
                }
            }
        }
        break;
    }

    /* ── SCATTER: grad_src = gather(grad_out, idx); grad_input passes through ── */

    case UOP_SCATTER: {
        /* Forward: out = input.copy(); out[dim, idx[i]] = src[i]
         * in[0]=input, in[1]=index, in[2]=src */
        if (node->num_inputs < 3) break;
        Tensor* input_t = node->inputs[0];
        Tensor* idx_t   = node->inputs[1];
        Tensor* src_t   = node->inputs[2];
        if (!idx_t || !idx_t->data) break;
        float* idx_data = (float*)idx_t->data;
        ScatterParams* sp = (ScatterParams*)node->params;
        int sdim = sp ? sp->dim : 0;

        /* grad for src: grad_src[i] = grad_out at the position we wrote to */
        if (src_t && src_t->requires_grad) {
            Tensor* gs = ensure_grad(src_t);
            if (gs && gs->data) {
                float* gsd = (float*)gs->data;
                if (out->ndim == 1) {
                    for (size_t i = 0; i < idx_t->numel; i++) {
                        int idx = (int)idx_data[i];
                        if (idx >= 0 && idx < (int)out->numel)
                            gsd[i] += out_grad[idx];
                    }
                } else if (out->ndim == 2) {
                    int cols = out->shape[1];
                    for (size_t i = 0; i < idx_t->numel; i++) {
                        int r = (int)i / idx_t->shape[1], c = (int)i % idx_t->shape[1];
                        int idx = (int)idx_data[i];
                        if (sdim == 0 && idx >= 0 && idx < out->shape[0])
                            gsd[i] += out_grad[(size_t)idx * cols + c];
                        else if (sdim == 1 && idx >= 0 && idx < cols)
                            gsd[i] += out_grad[(size_t)r * cols + idx];
                    }
                }
            }
        }

        /* grad for input: out_grad passes through except at scattered positions
         * (those positions' input contribution was overwritten by src) */
        if (input_t && input_t->requires_grad) {
            Tensor* gi = ensure_grad(input_t);
            if (gi && gi->data) {
                float* gid = (float*)gi->data;
                /* Start with full passthrough */
                for (size_t i = 0; i < out->numel; i++)
                    gid[i % input_t->numel] += out_grad[i];
                /* Subtract back the gradient at scattered positions (those were replaced) */
                if (out->ndim == 1) {
                    for (size_t i = 0; i < idx_t->numel; i++) {
                        int idx = (int)idx_data[i];
                        if (idx >= 0 && idx < (int)out->numel)
                            gid[idx] -= out_grad[idx];
                    }
                } else if (out->ndim == 2) {
                    int cols = out->shape[1];
                    for (size_t i = 0; i < idx_t->numel; i++) {
                        int r = (int)i / idx_t->shape[1], c = (int)i % idx_t->shape[1];
                        int idx = (int)idx_data[i];
                        if (sdim == 0 && idx >= 0 && idx < out->shape[0])
                            gid[(size_t)idx * cols + c] -= out_grad[(size_t)idx * cols + c];
                        else if (sdim == 1 && idx >= 0 && idx < cols)
                            gid[(size_t)r * cols + idx] -= out_grad[(size_t)r * cols + idx];
                    }
                }
            }
        }
        break;
    }

    /* ── UNFOLD: fold (transpose of unfold) — add overlapping windows ─── */

    case UOP_UNFOLD: {
        if (!in1 || !in1->requires_grad || !node->params)
            break;
        UnfoldParams* up = (UnfoldParams*)node->params;
        int ks = up->kernel_size;
        int stride = up->stride;
        Tensor* g1 = ensure_grad(in1);
        if (!g1 || !g1->data)
            break;
        float* g1d = (float*)g1->data;
        int ndim_in = in1->ndim;
        int last_dim = in1->shape[ndim_in - 1];
        int num_windows = (last_dim - ks) / stride + 1;

        if (ndim_in == 1) {
            for (int w = 0; w < num_windows; w++)
                for (int k = 0; k < ks; k++)
                    g1d[w * stride + k] += out_grad[w * ks + k];
        } else if (ndim_in == 2) {
            int N = in1->shape[0];
            for (int n = 0; n < N; n++)
                for (int w = 0; w < num_windows; w++)
                    for (int k = 0; k < ks; k++)
                        g1d[n * last_dim + w * stride + k] += out_grad[(n * num_windows + w) * ks + k];
        } else {
            size_t batch_size = 1;
            for (int d = 0; d < ndim_in - 1; d++)
                batch_size *= (size_t)in1->shape[d];
            for (size_t b = 0; b < batch_size; b++)
                for (int w = 0; w < num_windows; w++)
                    for (int k = 0; k < ks; k++)
                        g1d[b * (size_t)last_dim + w * stride + k] += out_grad[(b * (size_t)num_windows + w) * ks + k];
        }
        break;
    }

    /* ── CUMPROD: exclusive cumprod suffix sum trick ─────────── */

    case UOP_CUMPROD: {
        if (!in1 || !in1->requires_grad || !in1->data || !out->data)
            break;
        Tensor* g1 = ensure_grad(in1);
        if (!g1 || !g1->data)
            break;
        float* g1d = (float*)g1->data;
        float* x = (float*)in1->data;
        float* p_data = (float*)out->data; /* cumprod output */
        CumsumParams* cp = (CumsumParams*)node->params;
        int dim = cp ? cp->dim : 0;
        if (dim < 0) dim += in1->ndim;

        if (in1->ndim == 1) {
            int n = (int)in1->numel;
            /* grad_x[k] = suffix_sum_{i>=k}(grad_out[i] * p_data[i]) / x[k]
             * Use suffix accumulation; handle zeros via O(n^2) fallback. */
            for (int k = 0; k < n; k++) {
                if (fabsf(x[k]) > 1e-12f) {
                    float sfx = 0.f;
                    for (int i = k; i < n; i++)
                        sfx += out_grad[i] * p_data[i];
                    g1d[k] += sfx / x[k];
                } else {
                    /* Zero element: use prefix/suffix products to compute contribution */
                    float prefix = 1.f;
                    for (int j = 0; j < k; j++) prefix *= x[j];
                    float suffix = 1.f;
                    for (int i = k; i < n; i++) {
                        g1d[k] += out_grad[i] * prefix * suffix;
                        suffix *= (i + 1 < n) ? x[i + 1] : 1.f;
                    }
                }
            }
        } else if (in1->ndim == 2) {
            int rows = in1->shape[0], cols = in1->shape[1];
            if (dim == 1) {
                for (int r = 0; r < rows; r++) {
                    for (int k = 0; k < cols; k++) {
                        float xk = x[r * cols + k];
                        if (fabsf(xk) > 1e-12f) {
                            float sfx = 0.f;
                            for (int i = k; i < cols; i++)
                                sfx += out_grad[r * cols + i] * p_data[r * cols + i];
                            g1d[r * cols + k] += sfx / xk;
                        } else {
                            float prefix = 1.f;
                            for (int j = 0; j < k; j++) prefix *= x[r * cols + j];
                            float suffix = 1.f;
                            for (int i = k; i < cols; i++) {
                                g1d[r * cols + k] += out_grad[r * cols + i] * prefix * suffix;
                                if (i + 1 < cols) suffix *= x[r * cols + i + 1];
                            }
                        }
                    }
                }
            } else {
                for (int c = 0; c < cols; c++) {
                    for (int k = 0; k < rows; k++) {
                        float xk = x[k * cols + c];
                        if (fabsf(xk) > 1e-12f) {
                            float sfx = 0.f;
                            for (int i = k; i < rows; i++)
                                sfx += out_grad[i * cols + c] * p_data[i * cols + c];
                            g1d[k * cols + c] += sfx / xk;
                        } else {
                            float prefix = 1.f;
                            for (int j = 0; j < k; j++) prefix *= x[j * cols + c];
                            float suffix = 1.f;
                            for (int i = k; i < rows; i++) {
                                g1d[k * cols + c] += out_grad[i * cols + c] * prefix * suffix;
                                if (i + 1 < rows) suffix *= x[(i + 1) * cols + c];
                            }
                        }
                    }
                }
            }
        }
        break;
    }

    /* ── CUMMAX / CUMMIN: grad routes to first-occurrence argmax/min ── */

    case UOP_CUMMAX: {
        if (!in1 || !in1->requires_grad || !in1->data || !out->data)
            break;
        Tensor* g1 = ensure_grad(in1);
        if (!g1 || !g1->data)
            break;
        float* g1d = (float*)g1->data;
        float* x = (float*)in1->data;
        float* cm = (float*)out->data;
        CumsumParams* cp = (CumsumParams*)node->params;
        int cdim = cp ? cp->dim : 0;

        if (in1->ndim == 1) {
            int n = (int)in1->numel;
            /* For each output j, grad flows to the first i where x[i]==cm[j] and i<=j */
            for (int j = n - 1; j >= 0; j--) {
                int src = 0;
                for (int i = 0; i <= j; i++)
                    if (x[i] == cm[j]) { src = i; break; }
                g1d[src] += out_grad[j];
            }
        } else if (in1->ndim == 2) {
            int rows = in1->shape[0], cols = in1->shape[1];
            if (cdim == 0) {
                for (int c = 0; c < cols; c++)
                    for (int j = rows - 1; j >= 0; j--) {
                        int src = 0;
                        for (int i = 0; i <= j; i++)
                            if (x[i * cols + c] == cm[j * cols + c]) { src = i; break; }
                        g1d[src * cols + c] += out_grad[j * cols + c];
                    }
            } else {
                for (int r = 0; r < rows; r++)
                    for (int j = cols - 1; j >= 0; j--) {
                        int src = 0;
                        for (int i = 0; i <= j; i++)
                            if (x[r * cols + i] == cm[r * cols + j]) { src = i; break; }
                        g1d[r * cols + src] += out_grad[r * cols + j];
                    }
            }
        }
        break;
    }

    case UOP_CUMMIN: {
        if (!in1 || !in1->requires_grad || !in1->data || !out->data)
            break;
        Tensor* g1 = ensure_grad(in1);
        if (!g1 || !g1->data)
            break;
        float* g1d = (float*)g1->data;
        float* x = (float*)in1->data;
        float* cm = (float*)out->data;
        CumsumParams* cp = (CumsumParams*)node->params;
        int cdim = cp ? cp->dim : 0;

        if (in1->ndim == 1) {
            int n = (int)in1->numel;
            for (int j = n - 1; j >= 0; j--) {
                int src = 0;
                for (int i = 0; i <= j; i++)
                    if (x[i] == cm[j]) { src = i; break; }
                g1d[src] += out_grad[j];
            }
        } else if (in1->ndim == 2) {
            int rows = in1->shape[0], cols = in1->shape[1];
            if (cdim == 0) {
                for (int c = 0; c < cols; c++)
                    for (int j = rows - 1; j >= 0; j--) {
                        int src = 0;
                        for (int i = 0; i <= j; i++)
                            if (x[i * cols + c] == cm[j * cols + c]) { src = i; break; }
                        g1d[src * cols + c] += out_grad[j * cols + c];
                    }
            } else {
                for (int r = 0; r < rows; r++)
                    for (int j = cols - 1; j >= 0; j--) {
                        int src = 0;
                        for (int i = 0; i <= j; i++)
                            if (x[r * cols + i] == cm[r * cols + j]) { src = i; break; }
                        g1d[r * cols + src] += out_grad[r * cols + j];
                    }
            }
        }
        break;
    }

    /* ── LOGCUMSUMEXP: grad_x[k] = exp(x[k]) * suffix_sum(grad_out * exp(-out)) ── */

    case UOP_LOGCUMSUMEXP: {
        if (!in1 || !in1->requires_grad || !in1->data || !out->data)
            break;
        Tensor* g1 = ensure_grad(in1);
        if (!g1 || !g1->data)
            break;
        float* g1d = (float*)g1->data;
        float* x = (float*)in1->data;
        float* lse = (float*)out->data;
        CumsumParams* cp = (CumsumParams*)node->params;
        int dim = cp ? cp->dim : 0;

        if (in1->ndim == 1) {
            int n = (int)in1->numel;
            /* grad_x[k] = exp(x[k]) * sum_{i>=k} (out_grad[i] * exp(-lse[i])) */
            float sfx = 0.f;
            for (int i = n - 1; i >= 0; i--) {
                sfx += out_grad[i] * expf(-lse[i]);
                g1d[i] += expf(x[i]) * sfx;
            }
        } else if (in1->ndim == 2) {
            int rows = in1->shape[0], cols = in1->shape[1];
            if (dim == 1) {
                for (int r = 0; r < rows; r++) {
                    float sfx = 0.f;
                    for (int i = cols - 1; i >= 0; i--) {
                        sfx += out_grad[r * cols + i] * expf(-lse[r * cols + i]);
                        g1d[r * cols + i] += expf(x[r * cols + i]) * sfx;
                    }
                }
            } else {
                for (int c = 0; c < cols; c++) {
                    float sfx = 0.f;
                    for (int i = rows - 1; i >= 0; i--) {
                        sfx += out_grad[i * cols + c] * expf(-lse[i * cols + c]);
                        g1d[i * cols + c] += expf(x[i * cols + c]) * sfx;
                    }
                }
            }
        }
        break;
    }

    case UOP_SIGN:
        // sign(x) is piecewise constant — gradient is 0 everywhere (STE: pass zero)
        break;

    case UOP_COPYSIGN:
        // copysign(x,y) = |x| * sign(y)
        // d/dx = sign(y)*sign(x) = sign(x*y != 0 ? x*y : 0), i.e. +1 same sign, -1 opposite
        // d/dy = 0 (non-differentiable; conventional zero gradient)
        if (in1 && in1->requires_grad && in1->data && in2 && in2->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1d = (float*)g1->data;
                float* x   = (float*)in1->data;
                float* y   = (float*)in2->data;
                for (size_t i = 0; i < out_numel; i++) {
                    float xi = x[i % in1->numel];
                    float yi = y[i % in2->numel];
                    float s  = (yi > 0.f ? 1.f : yi < 0.f ? -1.f : 0.f)
                             * (xi > 0.f ? 1.f : xi < 0.f ? -1.f : 0.f);
                    g1d[i % in1->numel] += out_grad[i] * s;
                }
            }
        }
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
        /* Backward DCE: skip nodes where no input requires a gradient.
         * cml_ir_build_backward sets requires_grad via a forward scan. */
        if (!nodes[i]->requires_grad)
            continue;
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

    /* Ensure requires_grad / needs_input_grad flags are populated for
     * backward DCE.  cml_ir_build_backward is idempotent (forward scan). */
    cml_ir_build_backward(ir, node);

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
