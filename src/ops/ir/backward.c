/**
 * @file backward.c
 * @brief Backward pass execution with CPU fallback
 *
 * This file implements backward pass for gradient computation.
 * When MLIR is available, uses MLIR automatic differentiation.
 * Otherwise, uses a CPU interpreter for gradient computation.
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

int cml_ir_build_backward(CMLIR_t ir, struct IRNode* output_node) {
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

    // The backward graph will be built by MLIR's automatic differentiation
    // We just need to mark the output node
    output_node->is_used = true;

    LOG_DEBUG("Backward graph structure prepared (MLIR will handle AD)");
    return 0;
}

// Helper to allocate gradient tensor if needed
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

// Accumulate gradient into tensor
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

// CPU backward pass for a single node
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

    default:
        // Unsupported op - gradients not computed
        LOG_DEBUG("CPU backward: no gradient rule for op type %d", node->type);
        break;
    }

    return 0;
}

// Execute backward pass using CPU interpreter
static int cpu_execute_backward(CMLIR_t ir) {
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

int cml_ir_execute_backward(CMLIR_t ir) {
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

    // Use CPU fallback for backward pass
    // (MLIR backward is not fully implemented yet)
    return cpu_execute_backward(ir);
}
