/**
 * @file backward.c
 * @brief Backward pass execution with CPU fallback
 *
 * This file implements backward pass for gradient computation
 * using a CPU interpreter.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "tensor/tensor.h"
#include "core/logging.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "ops/uops.h"

int cml_ir_build_backward(CMLGraph_t ir, struct IRNode* output_node) {
    if (!ir || !output_node) {
        LOG_ERROR("Invalid arguments to cml_ir_build_backward");
        return -1;
    }

    LOG_DEBUG("Building backward graph for IR");

    // Mark all nodes that require gradients
    struct IRNode* node = ir->head;
    while (node) {
        if (node->requires_grad) {
            LOG_DEBUG("Node %s requires gradient",
                      node->output_name ? node->output_name : "unnamed");
        }
        node = node->next;
    }

    // Mark the output node as used (needed for dead code elimination)
    output_node->is_used = true;

    // Backward pass uses CPU interpreter (the symbolic differentiation logic
    // already creates standard UOps — the LLVM JIT handles those during
    // forward execution of the backward IR graph)

    LOG_DEBUG("Backward graph structure prepared (CPU fallback will handle gradients)");
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

    // Handle broadcasting: accumulate with modulo indexing
    for (size_t i = 0; i < numel; i++) {
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

    // Get input tensors
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
                for (size_t i = 0; i < out_numel; i++) {
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
                for (size_t i = 0; i < out_numel; i++) {
                    size_t i1 = i % in1->numel;
                    size_t i2 = i % in2->numel;
                    g1_data[i1] += out_grad[i] * in2_data[i2];
                }
            }
        }
        if (in2 && in2->requires_grad && in1 && in1->data) {
            Tensor* g2 = ensure_grad(in2);
            if (g2 && g2->data) {
                float* g2_data  = (float*)g2->data;
                float* in1_data = (float*)in1->data;
                for (size_t i = 0; i < out_numel; i++) {
                    size_t i1 = i % in1->numel;
                    size_t i2 = i % in2->numel;
                    g2_data[i2] += out_grad[i] * in1_data[i1];
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
                for (size_t i = 0; i < in1->numel; i++) {
                    g1_data[i] += out_grad[0] * scale;
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
                    for (size_t i = 0; i < out_numel; i++) {
                        size_t i1 = i % in1->numel;
                        size_t i2 = i % in2->numel;
                        if (in1_data[i1] >= in2_data[i2]) {
                            g1_data[i1] += out_grad[i];
                        }
                    }
                }
            }
            if (in2->requires_grad) {
                Tensor* g2 = ensure_grad(in2);
                if (g2 && g2->data) {
                    float* g2_data = (float*)g2->data;
                    for (size_t i = 0; i < out_numel; i++) {
                        size_t i1 = i % in1->numel;
                        size_t i2 = i % in2->numel;
                        if (in2_data[i2] > in1_data[i1]) {
                            g2_data[i2] += out_grad[i];
                        }
                    }
                }
            }
        }
        break;

    case UOP_MATMUL: {
        // d(A @ B)/dA = grad @ B^T
        // d(A @ B)/dB = A^T @ grad
        if (!in1 || !in2 || in1->ndim < 2 || in2->ndim < 2)
            break;

        int M = in1->shape[in1->ndim - 2];
        int K = in1->shape[in1->ndim - 1];
        int N = in2->shape[in2->ndim - 1];

        // Gradient w.r.t. first input: grad @ B^T
        if (in1->requires_grad && in2->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data  = (float*)g1->data;
                float* in2_data = (float*)in2->data;
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

        // Gradient w.r.t. second input: A^T @ grad
        if (in2->requires_grad && in1->data) {
            Tensor* g2 = ensure_grad(in2);
            if (g2 && g2->data) {
                float* g2_data  = (float*)g2->data;
                float* in1_data = (float*)in1->data;
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
        // Backward for 2D convolution: input gradient via "full" correlation
        // grad_input = conv2d_full(grad_output, rot180(weight))
        // grad_weight = conv2d(input, grad_output)
        if (!in1 || !in2)
            break;

        // in1 = input [N, C_in, H, W], in2 = weight [C_out, C_in, kH, kW]
        int ndim_in = in1->ndim;
        int ndim_w  = in2->ndim;
        if (ndim_in != 4 || ndim_w != 4)
            break;

        int N = in1->shape[0], C_in = in1->shape[1];
        int H = in1->shape[2], W = in1->shape[3];
        int C_out = in2->shape[0], kH = in2->shape[2], kW = in2->shape[3];
        int oH = out->shape[2], oW = out->shape[3];

        // Gradient w.r.t. input
        if (in1->requires_grad && in2->data) {
            Tensor* g1 = ensure_grad(in1);
            if (g1 && g1->data) {
                float* g1_data = (float*)g1->data;
                float* w_data  = (float*)in2->data;
                // Full convolution: pad grad_output by (kH-1, kW-1), convolve with rot180(weight)
                for (int n = 0; n < N; n++) {
                    for (int ci = 0; ci < C_in; ci++) {
                        for (int h = 0; h < H; h++) {
                            for (int w_idx = 0; w_idx < W; w_idx++) {
                                float sum = 0.0f;
                                for (int co = 0; co < C_out; co++) {
                                    for (int kh = 0; kh < kH; kh++) {
                                        for (int kw = 0; kw < kW; kw++) {
                                            int oh = h - kh;
                                            int ow = w_idx - kw;
                                            if (oh >= 0 && oh < oH && ow >= 0 && ow < oW) {
                                                float g =
                                                    out_grad[((n * C_out + co) * oH + oh) * oW +
                                                             ow];
                                                float wt =
                                                    w_data[((co * C_in + ci) * kH + kh) * kW + kw];
                                                sum += g * wt;
                                            }
                                        }
                                    }
                                }
                                g1_data[((n * C_in + ci) * H + h) * W + w_idx] += sum;
                            }
                        }
                    }
                }
            }
        }

        // Gradient w.r.t. weight
        if (in2->requires_grad && in1->data) {
            Tensor* g2 = ensure_grad(in2);
            if (g2 && g2->data) {
                float* g2_data = (float*)g2->data;
                float* in_data = (float*)in1->data;
                for (int co = 0; co < C_out; co++) {
                    for (int ci = 0; ci < C_in; ci++) {
                        for (int kh = 0; kh < kH; kh++) {
                            for (int kw = 0; kw < kW; kw++) {
                                float sum = 0.0f;
                                for (int n = 0; n < N; n++) {
                                    for (int oh = 0; oh < oH; oh++) {
                                        for (int ow = 0; ow < oW; ow++) {
                                            float g =
                                                out_grad[((n * C_out + co) * oH + oh) * oW + ow];
                                            float inp =
                                                in_data[((n * C_in + ci) * H + (oh + kh)) * W +
                                                        (ow + kw)];
                                            sum += g * inp;
                                        }
                                    }
                                }
                                g2_data[((co * C_in + ci) * kH + kh) * kW + kw] += sum;
                            }
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

    LOG_DEBUG("Executing backward pass using CPU interpreter");

    // Count nodes to build reverse traversal order
    int node_count   = 0;
    struct IRNode* n = ir->head;
    while (n) {
        node_count++;
        n = n->next;
    }

    if (node_count == 0)
        return 0;

    // Build array of nodes for reverse traversal
    struct IRNode** nodes = malloc(node_count * sizeof(struct IRNode*));
    if (!nodes) {
        LOG_ERROR("Failed to allocate node array for backward pass");
        return -1;
    }

    n = ir->head;
    for (int i = 0; i < node_count; i++) {
        nodes[i] = n;
        n        = n->next;
    }

    // Initialize loss gradient to 1
    struct IRNode* loss_node = ir->tail;
    if (loss_node && loss_node->output && loss_node->output->requires_grad) {
        Tensor* loss = loss_node->output;
        if (!loss->grad) {
            TensorConfig config = {.dtype      = loss->dtype,
                                   .device     = loss->device,
                                   .has_dtype  = true,
                                   .has_device = true};
            loss->grad          = tensor_zeros(loss->shape, loss->ndim, &config);
        }
        if (loss->grad && loss->grad->data && loss->numel == 1) {
            *(float*)loss->grad->data = 1.0f;
        }
    }

    // Traverse in reverse order (from loss to inputs)
    for (int i = node_count - 1; i >= 0; i--) {
        cpu_backward_node(nodes[i]);
    }

    free(nodes);

    LOG_DEBUG("CPU backward pass completed");
    return 0;
}

int cml_ir_execute_backward(CMLGraph_t ir) {
    if (!ir) {
        LOG_ERROR("NULL IR passed to cml_ir_execute_backward");
        return -1;
    }

    LOG_DEBUG("Executing backward pass");

    // Start from the tail node (loss) and set gradient to 1 if needed
    struct IRNode* node = ir->tail;
    if (!node) {
        LOG_ERROR("No tail node in IR graph");
        return -1;
    }

    // Initialize gradient for the output node (loss)
    if (node->output && node->output->requires_grad) {
        // Allocate grad tensor if not already present
        if (!node->output->grad) {
            // Create a proper gradient tensor (same shape as output)
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
        // Set gradient to 1 for scalar loss
        if (node->output->numel == 1) {
            if (node->output->dtype == DTYPE_FLOAT32) {
                *(float*)node->output->grad->data = 1.0f;
            }
        }
    }

    // Backward pass uses CPU interpreter (each backward op is a standard UOp
    // that the LLVM JIT handles during forward execution of backward graph)
    return cpu_execute_backward(ir);
}
