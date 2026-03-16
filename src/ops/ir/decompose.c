/**
 * @file decompose.c
 * @brief IR Decomposition Pass — rewrites composite ops into primitive ops
 *
 * Reduces composite ops to ~28 primitives so backends only need to handle
 * a minimal set. Runs before optimization so fusion can optimize primitive chains.
 */

#include "ops/ir/decompose.h"
#include "ops/ir/internal.h"
#include "ops/uops.h"
#include "tensor/tensor.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdatomic.h>

// Counter for generating unique intermediate names
static atomic_int g_decompose_counter = 0;

// Forward declaration
static char* decompose_unique_name(void);
static struct IRNode* create_primitive_node(CMLGraph_t ir, UOpType type,
                                            Tensor** inputs, int num_inputs,
                                            void* params, int* out_shape,
                                            int out_ndim);
static struct IRNode* insert_fill_node(CMLGraph_t ir, int* shape, int ndim, float value);
static void replace_node_with_chain(CMLGraph_t ir, struct IRNode* original,
                                    struct IRNode* chain_head, struct IRNode* chain_tail);

// Helpers

static char* decompose_unique_name(void) {
    int id = atomic_fetch_add(&g_decompose_counter, 1);
    char* name = malloc(32);
    if (name) {
        snprintf(name, 32, "_d%d", id);
    }
    return name;
}

/**
 * Create a new primitive IR node (not yet linked into the graph).
 * Creates an intermediate output tensor with the given shape.
 */
static struct IRNode* create_primitive_node(CMLGraph_t ir, UOpType type,
                                            Tensor** inputs, int num_inputs,
                                            void* params, int* out_shape,
                                            int out_ndim) {
    struct IRNode* node = calloc(1, sizeof(struct IRNode));
    if (!node) return NULL;

    node->type = type;
    node->num_inputs = num_inputs;
    node->params = params;

    // Copy input names
    if (num_inputs > 0 && inputs) {
        node->input_names = malloc((size_t)num_inputs * sizeof(char*));
        node->inputs = malloc((size_t)num_inputs * sizeof(Tensor*));
        if (!node->input_names || !node->inputs) {
            free(node->input_names);
            free(node->inputs);
            free(node);
            return NULL;
        }
        for (int i = 0; i < num_inputs; i++) {
            node->inputs[i] = inputs[i];
            // Get name from tensor's IR node if available
            if (inputs[i] && inputs[i]->ir_node &&
                ((struct IRNode*)inputs[i]->ir_node)->output_name) {
                node->input_names[i] = strdup(((struct IRNode*)inputs[i]->ir_node)->output_name);
            } else {
                // Generate a placeholder name
                node->input_names[i] = decompose_unique_name();
            }
        }
    } else {
        node->input_names = NULL;
        node->inputs = NULL;
    }

    // Generate output name
    node->output_name = decompose_unique_name();
    node->next = NULL;

    // Set output shape
    if (out_shape && out_ndim > 0) {
        node->output_shape = malloc((size_t)out_ndim * sizeof(int));
        if (node->output_shape) {
            memcpy(node->output_shape, out_shape, (size_t)out_ndim * sizeof(int));
        }
        node->output_ndim = out_ndim;
    }

    // Autograd: inherit requires_grad from inputs
    node->requires_grad = false;
    memset(node->needs_input_grad, 0, sizeof(node->needs_input_grad));
    for (int i = 0; i < num_inputs && i < 8; i++) {
        if (inputs && inputs[i] && inputs[i]->requires_grad) {
            node->requires_grad = true;
            node->needs_input_grad[i] = true;
        }
    }

    // Create intermediate output tensor
    Tensor* out_tensor = calloc(1, sizeof(Tensor));
    if (!out_tensor) {
        // Cleanup
        for (int i = 0; i < num_inputs; i++) free(node->input_names[i]);
        free(node->input_names);
        free(node->inputs);
        free(node->output_name);
        free(node->output_shape);
        free(node);
        return NULL;
    }

    out_tensor->ir_node = node;
    out_tensor->ir_context = ir;
    node->output = out_tensor;

    if (out_shape && out_ndim > 0) {
        out_tensor->shape = malloc((size_t)out_ndim * sizeof(int));
        if (out_tensor->shape) {
            memcpy(out_tensor->shape, out_shape, (size_t)out_ndim * sizeof(int));
        }
        out_tensor->ndim = out_ndim;
        size_t numel = 1;
        for (int i = 0; i < out_ndim; i++) numel *= (size_t)out_shape[i];
        out_tensor->numel = numel;
    }

    // Inherit dtype/device from first input
    if (num_inputs > 0 && inputs && inputs[0]) {
        out_tensor->dtype = inputs[0]->dtype;
        out_tensor->device = inputs[0]->device;
    } else {
        out_tensor->dtype = DTYPE_FLOAT32;
        out_tensor->device = DEVICE_CPU;
    }

    out_tensor->is_executed = false;
    out_tensor->data = NULL;
    out_tensor->owns_data = true;
    out_tensor->requires_grad = node->requires_grad;
    out_tensor->grad = NULL;
    out_tensor->ref_count = 1;
    out_tensor->base = NULL;

    // Compute strides
    if (out_ndim > 0 && out_tensor->shape) {
        out_tensor->strides = malloc((size_t)out_ndim * sizeof(size_t));
        if (out_tensor->strides) {
            size_t stride = 1;
            for (int i = out_ndim - 1; i >= 0; i--) {
                out_tensor->strides[i] = stride;
                stride *= (size_t)out_tensor->shape[i];
            }
        }
    }
    out_tensor->storage_offset = 0;
    out_tensor->is_contiguous = true;
    out_tensor->buffer_handle = NULL;
    out_tensor->user_data = NULL;

    return node;
}

/**
 * Create a FILL node for a constant scalar broadcast to given shape.
 */
static struct IRNode* insert_fill_node(CMLGraph_t ir, int* shape, int ndim, float value) {
    FillParams* params = malloc(sizeof(FillParams));
    if (!params) return NULL;
    params->value = value;
    params->ndim = ndim;
    params->shape = malloc((size_t)ndim * sizeof(int));
    if (!params->shape) { free(params); return NULL; }
    memcpy(params->shape, shape, (size_t)ndim * sizeof(int));

    struct IRNode* node = create_primitive_node(ir, UOP_FILL, NULL, 0, params, shape, ndim);
    return node;
}

/**
 * Replace `original` node with a chain of primitive nodes [chain_head..chain_tail].
 *
 * The chain_tail's output tensor replaces the original node's output tensor,
 * so all downstream references remain valid.
 *
 * The chain is inserted before `original` in the linked list, then `original`
 * is removed.
 */
static void replace_node_with_chain(CMLGraph_t ir, struct IRNode* original,
                                    struct IRNode* chain_head, struct IRNode* chain_tail) {
    if (!ir || !original || !chain_head || !chain_tail) return;

    // The chain_tail's output should take over the original's output tensor
    // so that downstream nodes referencing original->output still work.
    Tensor* orig_output = original->output;
    Tensor* chain_output = chain_tail->output;

    if (orig_output && chain_output) {
        // Copy the final result's IR references to the original output tensor
        orig_output->ir_node = chain_tail;

        // Free the chain_tail's intermediate output tensor (no longer needed)
        if (chain_output->shape) free(chain_output->shape);
        if (chain_output->strides) free(chain_output->strides);
        free(chain_output);

        // Point chain_tail at the original output tensor
        chain_tail->output = orig_output;

        // Update the chain_tail's output name to match original
        if (chain_tail->output_name) free(chain_tail->output_name);
        chain_tail->output_name = strdup(original->output_name ? original->output_name : "_decomp");
    }

    // Find the node before `original` in the linked list
    struct IRNode* prev = NULL;
    struct IRNode* cur = ir->head;
    while (cur && cur != original) {
        prev = cur;
        cur = cur->next;
    }

    // Count chain length to update node_count
    int chain_len = 0;
    for (struct IRNode* n = chain_head; n; n = n->next) {
        chain_len++;
        if (n == chain_tail) break;
    }

    // Link chain into the list
    chain_tail->next = original->next;

    if (prev) {
        prev->next = chain_head;
    } else {
        ir->head = chain_head;
    }

    // Update tail if needed
    if (ir->tail == original) {
        ir->tail = chain_tail;
    }

    // Update node_count: replaced 1 node with chain_len nodes
    ir->node_count += (chain_len - 1);

    // Free the original node (but NOT its output tensor — we kept it)
    original->output = NULL; // Prevent double-free
    if (original->input_names) {
        for (int i = 0; i < original->num_inputs; i++) {
            free(original->input_names[i]);
        }
        free(original->input_names);
    }
    free(original->inputs);
    free(original->output_name);
    free(original->output_shape);
    // Don't free params — some nodes share params or they were shallow-copied
    // The original node owned them, but since we're decomposing, the params
    // are no longer needed. Free them here for composite ops.
    if (original->params) {
        // For safety, free known param types
        switch (original->type) {
        case UOP_CLAMP: free(original->params); break;
        default: break; // Most composite ops have no params
        }
    }
    free(original);
}

/**
 * Link chain_head..chain_tail into a singly-linked chain.
 * Returns the tail.
 */
static struct IRNode* chain_append(struct IRNode** head, struct IRNode** tail,
                                    struct IRNode* node) {
    if (!node) return *tail;
    if (!*head) {
        *head = node;
        *tail = node;
    } else {
        (*tail)->next = node;
        *tail = node;
    }
    node->next = NULL;
    return node;
}

// Decomposition Rules

// SIGMOID: recip(1 + exp(-x))
static int decompose_sigmoid(CMLGraph_t ir, struct IRNode* node) {
    Tensor* x = node->inputs[0];
    int* shape = x->shape;
    int ndim = x->ndim;

    struct IRNode *head = NULL, *tail = NULL;

    // neg_x = neg(x)
    struct IRNode* neg_node = create_primitive_node(ir, UOP_NEG, &x, 1, NULL, shape, ndim);
    if (!neg_node) return -1;
    chain_append(&head, &tail, neg_node);

    // exp_neg = exp(neg_x)
    Tensor* neg_x = neg_node->output;
    struct IRNode* exp_node = create_primitive_node(ir, UOP_EXP, &neg_x, 1, NULL, shape, ndim);
    if (!exp_node) return -1;
    chain_append(&head, &tail, exp_node);

    // one = fill(1.0)
    struct IRNode* one_node = insert_fill_node(ir, shape, ndim, 1.0f);
    if (!one_node) return -1;
    chain_append(&head, &tail, one_node);

    // sum_ = one + exp_neg
    Tensor* one = one_node->output;
    Tensor* exp_neg = exp_node->output;
    Tensor* add_inputs[] = {one, exp_neg};
    struct IRNode* add_node = create_primitive_node(ir, UOP_ADD, add_inputs, 2, NULL, shape, ndim);
    if (!add_node) return -1;
    chain_append(&head, &tail, add_node);

    // result = recip(sum_)
    Tensor* sum_ = add_node->output;
    struct IRNode* recip_node = create_primitive_node(ir, UOP_RECIP, &sum_, 1, NULL, shape, ndim);
    if (!recip_node) return -1;
    chain_append(&head, &tail, recip_node);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// TANH: 2 * sigmoid(2x) - 1
static int decompose_tanh(CMLGraph_t ir, struct IRNode* node) {
    Tensor* x = node->inputs[0];
    int* shape = x->shape;
    int ndim = x->ndim;

    struct IRNode *head = NULL, *tail = NULL;

    // two = fill(2.0)
    struct IRNode* two_node = insert_fill_node(ir, shape, ndim, 2.0f);
    if (!two_node) return -1;
    chain_append(&head, &tail, two_node);

    // two_x = 2 * x
    Tensor* two = two_node->output;
    Tensor* mul_inputs[] = {two, x};
    struct IRNode* mul_node = create_primitive_node(ir, UOP_MUL, mul_inputs, 2, NULL, shape, ndim);
    if (!mul_node) return -1;
    chain_append(&head, &tail, mul_node);

    // sigmoid(2x) = recip(1 + exp(-2x))
    Tensor* two_x = mul_node->output;
    struct IRNode* neg_node = create_primitive_node(ir, UOP_NEG, &two_x, 1, NULL, shape, ndim);
    if (!neg_node) return -1;
    chain_append(&head, &tail, neg_node);

    Tensor* neg_2x = neg_node->output;
    struct IRNode* exp_node = create_primitive_node(ir, UOP_EXP, &neg_2x, 1, NULL, shape, ndim);
    if (!exp_node) return -1;
    chain_append(&head, &tail, exp_node);

    struct IRNode* one_node = insert_fill_node(ir, shape, ndim, 1.0f);
    if (!one_node) return -1;
    chain_append(&head, &tail, one_node);

    Tensor* add_inputs[] = {one_node->output, exp_node->output};
    struct IRNode* add_node = create_primitive_node(ir, UOP_ADD, add_inputs, 2, NULL, shape, ndim);
    if (!add_node) return -1;
    chain_append(&head, &tail, add_node);

    struct IRNode* sig_node = create_primitive_node(ir, UOP_RECIP, &add_node->output, 1, NULL, shape, ndim);
    if (!sig_node) return -1;
    chain_append(&head, &tail, sig_node);

    // 2 * sigmoid(2x)
    struct IRNode* two2_node = insert_fill_node(ir, shape, ndim, 2.0f);
    if (!two2_node) return -1;
    chain_append(&head, &tail, two2_node);

    Tensor* mul2_inputs[] = {two2_node->output, sig_node->output};
    struct IRNode* mul2_node = create_primitive_node(ir, UOP_MUL, mul2_inputs, 2, NULL, shape, ndim);
    if (!mul2_node) return -1;
    chain_append(&head, &tail, mul2_node);

    // 2 * sigmoid(2x) - 1
    struct IRNode* one2_node = insert_fill_node(ir, shape, ndim, 1.0f);
    if (!one2_node) return -1;
    chain_append(&head, &tail, one2_node);

    Tensor* sub_inputs[] = {mul2_node->output, one2_node->output};
    struct IRNode* sub_node = create_primitive_node(ir, UOP_SUB, sub_inputs, 2, NULL, shape, ndim);
    if (!sub_node) return -1;
    chain_append(&head, &tail, sub_node);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// ABS: where(x < 0, -x, x)
static int decompose_abs(CMLGraph_t ir, struct IRNode* node) {
    Tensor* x = node->inputs[0];
    int* shape = x->shape;
    int ndim = x->ndim;

    struct IRNode *head = NULL, *tail = NULL;

    // zero = fill(0)
    struct IRNode* zero_node = insert_fill_node(ir, shape, ndim, 0.0f);
    if (!zero_node) return -1;
    chain_append(&head, &tail, zero_node);

    // cond = x < 0
    Tensor* cmplt_inputs[] = {x, zero_node->output};
    struct IRNode* cmplt_node = create_primitive_node(ir, UOP_CMPLT, cmplt_inputs, 2, NULL, shape, ndim);
    if (!cmplt_node) return -1;
    chain_append(&head, &tail, cmplt_node);

    // neg_x = -x
    struct IRNode* neg_node = create_primitive_node(ir, UOP_NEG, &x, 1, NULL, shape, ndim);
    if (!neg_node) return -1;
    chain_append(&head, &tail, neg_node);

    // result = where(cond, neg_x, x)
    Tensor* where_inputs[] = {cmplt_node->output, neg_node->output, x};
    struct IRNode* where_node = create_primitive_node(ir, UOP_WHERE, where_inputs, 3, NULL, shape, ndim);
    if (!where_node) return -1;
    chain_append(&head, &tail, where_node);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// SQUARE: x * x
static int decompose_square(CMLGraph_t ir, struct IRNode* node) {
    Tensor* x = node->inputs[0];
    int* shape = x->shape;
    int ndim = x->ndim;

    struct IRNode *head = NULL, *tail = NULL;

    Tensor* mul_inputs[] = {x, x};
    struct IRNode* mul_node = create_primitive_node(ir, UOP_MUL, mul_inputs, 2, NULL, shape, ndim);
    if (!mul_node) return -1;
    chain_append(&head, &tail, mul_node);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// RSQRT: recip(sqrt(x))
static int decompose_rsqrt(CMLGraph_t ir, struct IRNode* node) {
    Tensor* x = node->inputs[0];
    int* shape = x->shape;
    int ndim = x->ndim;

    struct IRNode *head = NULL, *tail = NULL;

    struct IRNode* sqrt_node = create_primitive_node(ir, UOP_SQRT, &x, 1, NULL, shape, ndim);
    if (!sqrt_node) return -1;
    chain_append(&head, &tail, sqrt_node);

    struct IRNode* recip_node = create_primitive_node(ir, UOP_RECIP, &sqrt_node->output, 1, NULL, shape, ndim);
    if (!recip_node) return -1;
    chain_append(&head, &tail, recip_node);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// COS: sin(x + pi/2)
static int decompose_cos(CMLGraph_t ir, struct IRNode* node) {
    Tensor* x = node->inputs[0];
    int* shape = x->shape;
    int ndim = x->ndim;

    struct IRNode *head = NULL, *tail = NULL;

    struct IRNode* halfpi_node = insert_fill_node(ir, shape, ndim, (float)(M_PI / 2.0));
    if (!halfpi_node) return -1;
    chain_append(&head, &tail, halfpi_node);

    Tensor* add_inputs[] = {x, halfpi_node->output};
    struct IRNode* add_node = create_primitive_node(ir, UOP_ADD, add_inputs, 2, NULL, shape, ndim);
    if (!add_node) return -1;
    chain_append(&head, &tail, add_node);

    struct IRNode* sin_node = create_primitive_node(ir, UOP_SIN, &add_node->output, 1, NULL, shape, ndim);
    if (!sin_node) return -1;
    chain_append(&head, &tail, sin_node);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// TAN: sin(x) / sin(x + pi/2)
static int decompose_tan(CMLGraph_t ir, struct IRNode* node) {
    Tensor* x = node->inputs[0];
    int* shape = x->shape;
    int ndim = x->ndim;

    struct IRNode *head = NULL, *tail = NULL;

    // sin(x)
    struct IRNode* sin_node = create_primitive_node(ir, UOP_SIN, &x, 1, NULL, shape, ndim);
    if (!sin_node) return -1;
    chain_append(&head, &tail, sin_node);

    // cos(x) = sin(x + pi/2)
    struct IRNode* halfpi_node = insert_fill_node(ir, shape, ndim, (float)(M_PI / 2.0));
    if (!halfpi_node) return -1;
    chain_append(&head, &tail, halfpi_node);

    Tensor* add_inputs[] = {x, halfpi_node->output};
    struct IRNode* add_node = create_primitive_node(ir, UOP_ADD, add_inputs, 2, NULL, shape, ndim);
    if (!add_node) return -1;
    chain_append(&head, &tail, add_node);

    struct IRNode* cos_node = create_primitive_node(ir, UOP_SIN, &add_node->output, 1, NULL, shape, ndim);
    if (!cos_node) return -1;
    chain_append(&head, &tail, cos_node);

    // sin(x) / cos(x)
    Tensor* div_inputs[] = {sin_node->output, cos_node->output};
    struct IRNode* div_node = create_primitive_node(ir, UOP_DIV, div_inputs, 2, NULL, shape, ndim);
    if (!div_node) return -1;
    chain_append(&head, &tail, div_node);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// LOG2: log(x) / log(2)
static int decompose_log2(CMLGraph_t ir, struct IRNode* node) {
    Tensor* x = node->inputs[0];
    int* shape = x->shape;
    int ndim = x->ndim;

    struct IRNode *head = NULL, *tail = NULL;

    struct IRNode* log_node = create_primitive_node(ir, UOP_LOG, &x, 1, NULL, shape, ndim);
    if (!log_node) return -1;
    chain_append(&head, &tail, log_node);

    // 1/log(2) as constant multiplier is more efficient
    struct IRNode* inv_ln2_node = insert_fill_node(ir, shape, ndim, (float)(1.0 / log(2.0)));
    if (!inv_ln2_node) return -1;
    chain_append(&head, &tail, inv_ln2_node);

    Tensor* mul_inputs[] = {log_node->output, inv_ln2_node->output};
    struct IRNode* mul_node = create_primitive_node(ir, UOP_MUL, mul_inputs, 2, NULL, shape, ndim);
    if (!mul_node) return -1;
    chain_append(&head, &tail, mul_node);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// EXP2: exp(x * log(2))
static int decompose_exp2(CMLGraph_t ir, struct IRNode* node) {
    Tensor* x = node->inputs[0];
    int* shape = x->shape;
    int ndim = x->ndim;

    struct IRNode *head = NULL, *tail = NULL;

    struct IRNode* ln2_node = insert_fill_node(ir, shape, ndim, (float)log(2.0));
    if (!ln2_node) return -1;
    chain_append(&head, &tail, ln2_node);

    Tensor* mul_inputs[] = {x, ln2_node->output};
    struct IRNode* mul_node = create_primitive_node(ir, UOP_MUL, mul_inputs, 2, NULL, shape, ndim);
    if (!mul_node) return -1;
    chain_append(&head, &tail, mul_node);

    struct IRNode* exp_node = create_primitive_node(ir, UOP_EXP, &mul_node->output, 1, NULL, shape, ndim);
    if (!exp_node) return -1;
    chain_append(&head, &tail, exp_node);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// SIGN: where(x > 0, 1, where(x < 0, -1, 0))
static int decompose_sign(CMLGraph_t ir, struct IRNode* node) {
    Tensor* x = node->inputs[0];
    int* shape = x->shape;
    int ndim = x->ndim;

    struct IRNode *head = NULL, *tail = NULL;

    struct IRNode* zero_node = insert_fill_node(ir, shape, ndim, 0.0f);
    if (!zero_node) return -1;
    chain_append(&head, &tail, zero_node);

    // x < 0
    Tensor* cmplt_inputs[] = {x, zero_node->output};
    struct IRNode* cmplt_neg = create_primitive_node(ir, UOP_CMPLT, cmplt_inputs, 2, NULL, shape, ndim);
    if (!cmplt_neg) return -1;
    chain_append(&head, &tail, cmplt_neg);

    // 0 < x  (i.e., x > 0)
    Tensor* cmpgt_inputs[] = {zero_node->output, x};
    struct IRNode* cmplt_pos = create_primitive_node(ir, UOP_CMPLT, cmpgt_inputs, 2, NULL, shape, ndim);
    if (!cmplt_pos) return -1;
    chain_append(&head, &tail, cmplt_pos);

    struct IRNode* neg1_node = insert_fill_node(ir, shape, ndim, -1.0f);
    if (!neg1_node) return -1;
    chain_append(&head, &tail, neg1_node);

    struct IRNode* zero2_node = insert_fill_node(ir, shape, ndim, 0.0f);
    if (!zero2_node) return -1;
    chain_append(&head, &tail, zero2_node);

    // inner_where = where(x < 0, -1, 0)
    Tensor* inner_inputs[] = {cmplt_neg->output, neg1_node->output, zero2_node->output};
    struct IRNode* inner_where = create_primitive_node(ir, UOP_WHERE, inner_inputs, 3, NULL, shape, ndim);
    if (!inner_where) return -1;
    chain_append(&head, &tail, inner_where);

    struct IRNode* one_node = insert_fill_node(ir, shape, ndim, 1.0f);
    if (!one_node) return -1;
    chain_append(&head, &tail, one_node);

    // result = where(x > 0, 1, inner_where)
    Tensor* outer_inputs[] = {cmplt_pos->output, one_node->output, inner_where->output};
    struct IRNode* outer_where = create_primitive_node(ir, UOP_WHERE, outer_inputs, 3, NULL, shape, ndim);
    if (!outer_where) return -1;
    chain_append(&head, &tail, outer_where);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// Phase B — Comparisons

// CMPEQ: where(a < b, 0, where(b < a, 0, 1))
static int decompose_cmpeq(CMLGraph_t ir, struct IRNode* node) {
    Tensor* a = node->inputs[0];
    Tensor* b = node->inputs[1];
    int* shape = node->output ? node->output->shape : node->output_shape;
    int ndim = node->output ? node->output->ndim : node->output_ndim;

    struct IRNode *head = NULL, *tail = NULL;

    // a < b
    Tensor* ab_inputs[] = {a, b};
    struct IRNode* lt_ab = create_primitive_node(ir, UOP_CMPLT, ab_inputs, 2, NULL, shape, ndim);
    if (!lt_ab) return -1;
    chain_append(&head, &tail, lt_ab);

    // b < a
    Tensor* ba_inputs[] = {b, a};
    struct IRNode* lt_ba = create_primitive_node(ir, UOP_CMPLT, ba_inputs, 2, NULL, shape, ndim);
    if (!lt_ba) return -1;
    chain_append(&head, &tail, lt_ba);

    struct IRNode* zero_node = insert_fill_node(ir, shape, ndim, 0.0f);
    if (!zero_node) return -1;
    chain_append(&head, &tail, zero_node);

    struct IRNode* one_node = insert_fill_node(ir, shape, ndim, 1.0f);
    if (!one_node) return -1;
    chain_append(&head, &tail, one_node);

    // inner = where(b < a, 0, 1)
    Tensor* inner_inputs[] = {lt_ba->output, zero_node->output, one_node->output};
    struct IRNode* inner = create_primitive_node(ir, UOP_WHERE, inner_inputs, 3, NULL, shape, ndim);
    if (!inner) return -1;
    chain_append(&head, &tail, inner);

    struct IRNode* zero2_node = insert_fill_node(ir, shape, ndim, 0.0f);
    if (!zero2_node) return -1;
    chain_append(&head, &tail, zero2_node);

    // result = where(a < b, 0, inner)
    Tensor* outer_inputs[] = {lt_ab->output, zero2_node->output, inner->output};
    struct IRNode* outer = create_primitive_node(ir, UOP_WHERE, outer_inputs, 3, NULL, shape, ndim);
    if (!outer) return -1;
    chain_append(&head, &tail, outer);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// CMPNE: 1 - cmpeq(a, b)  => decompose to primitives directly
static int decompose_cmpne(CMLGraph_t ir, struct IRNode* node) {
    Tensor* a = node->inputs[0];
    Tensor* b = node->inputs[1];
    int* shape = node->output ? node->output->shape : node->output_shape;
    int ndim = node->output ? node->output->ndim : node->output_ndim;

    struct IRNode *head = NULL, *tail = NULL;

    // a < b
    Tensor* ab_inputs[] = {a, b};
    struct IRNode* lt_ab = create_primitive_node(ir, UOP_CMPLT, ab_inputs, 2, NULL, shape, ndim);
    if (!lt_ab) return -1;
    chain_append(&head, &tail, lt_ab);

    // b < a
    Tensor* ba_inputs[] = {b, a};
    struct IRNode* lt_ba = create_primitive_node(ir, UOP_CMPLT, ba_inputs, 2, NULL, shape, ndim);
    if (!lt_ba) return -1;
    chain_append(&head, &tail, lt_ba);

    struct IRNode* zero_node = insert_fill_node(ir, shape, ndim, 0.0f);
    if (!zero_node) return -1;
    chain_append(&head, &tail, zero_node);

    struct IRNode* one_node = insert_fill_node(ir, shape, ndim, 1.0f);
    if (!one_node) return -1;
    chain_append(&head, &tail, one_node);

    // inner = where(b < a, 0, 1)  -- this is cmpeq
    Tensor* inner_inputs[] = {lt_ba->output, zero_node->output, one_node->output};
    struct IRNode* inner = create_primitive_node(ir, UOP_WHERE, inner_inputs, 3, NULL, shape, ndim);
    if (!inner) return -1;
    chain_append(&head, &tail, inner);

    struct IRNode* zero2_node = insert_fill_node(ir, shape, ndim, 0.0f);
    if (!zero2_node) return -1;
    chain_append(&head, &tail, zero2_node);

    // cmpeq = where(a < b, 0, inner)
    Tensor* cmpeq_inputs[] = {lt_ab->output, zero2_node->output, inner->output};
    struct IRNode* cmpeq = create_primitive_node(ir, UOP_WHERE, cmpeq_inputs, 3, NULL, shape, ndim);
    if (!cmpeq) return -1;
    chain_append(&head, &tail, cmpeq);

    // result = 1 - cmpeq
    struct IRNode* one2_node = insert_fill_node(ir, shape, ndim, 1.0f);
    if (!one2_node) return -1;
    chain_append(&head, &tail, one2_node);

    Tensor* sub_inputs[] = {one2_node->output, cmpeq->output};
    struct IRNode* sub_node = create_primitive_node(ir, UOP_SUB, sub_inputs, 2, NULL, shape, ndim);
    if (!sub_node) return -1;
    chain_append(&head, &tail, sub_node);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// CMPLE: 1 - cmplt(b, a)
static int decompose_cmple(CMLGraph_t ir, struct IRNode* node) {
    Tensor* a = node->inputs[0];
    Tensor* b = node->inputs[1];
    int* shape = node->output ? node->output->shape : node->output_shape;
    int ndim = node->output ? node->output->ndim : node->output_ndim;

    struct IRNode *head = NULL, *tail = NULL;

    // b < a
    Tensor* ba_inputs[] = {b, a};
    struct IRNode* lt_ba = create_primitive_node(ir, UOP_CMPLT, ba_inputs, 2, NULL, shape, ndim);
    if (!lt_ba) return -1;
    chain_append(&head, &tail, lt_ba);

    // 1 - (b < a)
    struct IRNode* one_node = insert_fill_node(ir, shape, ndim, 1.0f);
    if (!one_node) return -1;
    chain_append(&head, &tail, one_node);

    Tensor* sub_inputs[] = {one_node->output, lt_ba->output};
    struct IRNode* sub_node = create_primitive_node(ir, UOP_SUB, sub_inputs, 2, NULL, shape, ndim);
    if (!sub_node) return -1;
    chain_append(&head, &tail, sub_node);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// CMPGT: cmplt(b, a)
static int decompose_cmpgt(CMLGraph_t ir, struct IRNode* node) {
    Tensor* a = node->inputs[0];
    Tensor* b = node->inputs[1];
    int* shape = node->output ? node->output->shape : node->output_shape;
    int ndim = node->output ? node->output->ndim : node->output_ndim;

    struct IRNode *head = NULL, *tail = NULL;

    Tensor* ba_inputs[] = {b, a};
    struct IRNode* lt_ba = create_primitive_node(ir, UOP_CMPLT, ba_inputs, 2, NULL, shape, ndim);
    if (!lt_ba) return -1;
    chain_append(&head, &tail, lt_ba);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// CMPGE: 1 - cmplt(a, b)
static int decompose_cmpge(CMLGraph_t ir, struct IRNode* node) {
    Tensor* a = node->inputs[0];
    Tensor* b = node->inputs[1];
    int* shape = node->output ? node->output->shape : node->output_shape;
    int ndim = node->output ? node->output->ndim : node->output_ndim;

    struct IRNode *head = NULL, *tail = NULL;

    Tensor* ab_inputs[] = {a, b};
    struct IRNode* lt_ab = create_primitive_node(ir, UOP_CMPLT, ab_inputs, 2, NULL, shape, ndim);
    if (!lt_ab) return -1;
    chain_append(&head, &tail, lt_ab);

    struct IRNode* one_node = insert_fill_node(ir, shape, ndim, 1.0f);
    if (!one_node) return -1;
    chain_append(&head, &tail, one_node);

    Tensor* sub_inputs[] = {one_node->output, lt_ab->output};
    struct IRNode* sub_node = create_primitive_node(ir, UOP_SUB, sub_inputs, 2, NULL, shape, ndim);
    if (!sub_node) return -1;
    chain_append(&head, &tail, sub_node);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// MINIMUM: where(a < b, a, b)
static int decompose_minimum(CMLGraph_t ir, struct IRNode* node) {
    Tensor* a = node->inputs[0];
    Tensor* b = node->inputs[1];
    int* shape = node->output ? node->output->shape : node->output_shape;
    int ndim = node->output ? node->output->ndim : node->output_ndim;

    struct IRNode *head = NULL, *tail = NULL;

    Tensor* cmplt_inputs[] = {a, b};
    struct IRNode* cmplt_node = create_primitive_node(ir, UOP_CMPLT, cmplt_inputs, 2, NULL, shape, ndim);
    if (!cmplt_node) return -1;
    chain_append(&head, &tail, cmplt_node);

    Tensor* where_inputs[] = {cmplt_node->output, a, b};
    struct IRNode* where_node = create_primitive_node(ir, UOP_WHERE, where_inputs, 3, NULL, shape, ndim);
    if (!where_node) return -1;
    chain_append(&head, &tail, where_node);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// Phase C — Math

// LOG10: log(x) * (1/log(10))
static int decompose_log10(CMLGraph_t ir, struct IRNode* node) {
    Tensor* x = node->inputs[0];
    int* shape = x->shape;
    int ndim = x->ndim;

    struct IRNode *head = NULL, *tail = NULL;

    struct IRNode* log_node = create_primitive_node(ir, UOP_LOG, &x, 1, NULL, shape, ndim);
    if (!log_node) return -1;
    chain_append(&head, &tail, log_node);

    struct IRNode* inv_ln10 = insert_fill_node(ir, shape, ndim, (float)(1.0 / log(10.0)));
    if (!inv_ln10) return -1;
    chain_append(&head, &tail, inv_ln10);

    Tensor* mul_inputs[] = {log_node->output, inv_ln10->output};
    struct IRNode* mul_node = create_primitive_node(ir, UOP_MUL, mul_inputs, 2, NULL, shape, ndim);
    if (!mul_node) return -1;
    chain_append(&head, &tail, mul_node);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// LOGADDEXP: max(a,b) + log(1 + exp(-|a-b|))
static int decompose_logaddexp(CMLGraph_t ir, struct IRNode* node) {
    Tensor* a = node->inputs[0];
    Tensor* b = node->inputs[1];
    int* shape = node->output ? node->output->shape : node->output_shape;
    int ndim = node->output ? node->output->ndim : node->output_ndim;

    struct IRNode *head = NULL, *tail = NULL;

    // max(a, b)
    Tensor* max_inputs[] = {a, b};
    struct IRNode* max_node = create_primitive_node(ir, UOP_MAX, max_inputs, 2, NULL, shape, ndim);
    if (!max_node) return -1;
    chain_append(&head, &tail, max_node);

    // a - b
    Tensor* sub_inputs[] = {a, b};
    struct IRNode* sub_node = create_primitive_node(ir, UOP_SUB, sub_inputs, 2, NULL, shape, ndim);
    if (!sub_node) return -1;
    chain_append(&head, &tail, sub_node);

    // |a - b| via where(diff < 0, -diff, diff)
    struct IRNode* zero_node = insert_fill_node(ir, shape, ndim, 0.0f);
    if (!zero_node) return -1;
    chain_append(&head, &tail, zero_node);

    Tensor* cmplt_inputs[] = {sub_node->output, zero_node->output};
    struct IRNode* cmplt_node = create_primitive_node(ir, UOP_CMPLT, cmplt_inputs, 2, NULL, shape, ndim);
    if (!cmplt_node) return -1;
    chain_append(&head, &tail, cmplt_node);

    struct IRNode* neg_diff = create_primitive_node(ir, UOP_NEG, &sub_node->output, 1, NULL, shape, ndim);
    if (!neg_diff) return -1;
    chain_append(&head, &tail, neg_diff);

    Tensor* abs_inputs[] = {cmplt_node->output, neg_diff->output, sub_node->output};
    struct IRNode* abs_node = create_primitive_node(ir, UOP_WHERE, abs_inputs, 3, NULL, shape, ndim);
    if (!abs_node) return -1;
    chain_append(&head, &tail, abs_node);

    // -|a-b|
    struct IRNode* neg_abs = create_primitive_node(ir, UOP_NEG, &abs_node->output, 1, NULL, shape, ndim);
    if (!neg_abs) return -1;
    chain_append(&head, &tail, neg_abs);

    // exp(-|a-b|)
    struct IRNode* exp_node = create_primitive_node(ir, UOP_EXP, &neg_abs->output, 1, NULL, shape, ndim);
    if (!exp_node) return -1;
    chain_append(&head, &tail, exp_node);

    // 1 + exp(-|a-b|)
    struct IRNode* one_node = insert_fill_node(ir, shape, ndim, 1.0f);
    if (!one_node) return -1;
    chain_append(&head, &tail, one_node);

    Tensor* add1_inputs[] = {one_node->output, exp_node->output};
    struct IRNode* add1_node = create_primitive_node(ir, UOP_ADD, add1_inputs, 2, NULL, shape, ndim);
    if (!add1_node) return -1;
    chain_append(&head, &tail, add1_node);

    // log(1 + exp(-|a-b|))
    struct IRNode* log_node = create_primitive_node(ir, UOP_LOG, &add1_node->output, 1, NULL, shape, ndim);
    if (!log_node) return -1;
    chain_append(&head, &tail, log_node);

    // max(a,b) + log(...)
    Tensor* add2_inputs[] = {max_node->output, log_node->output};
    struct IRNode* add2_node = create_primitive_node(ir, UOP_ADD, add2_inputs, 2, NULL, shape, ndim);
    if (!add2_node) return -1;
    chain_append(&head, &tail, add2_node);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// MOD: a - floor(a/b) * b
static int decompose_mod(CMLGraph_t ir, struct IRNode* node) {
    Tensor* a = node->inputs[0];
    Tensor* b = node->inputs[1];
    int* shape = node->output ? node->output->shape : node->output_shape;
    int ndim = node->output ? node->output->ndim : node->output_ndim;

    struct IRNode *head = NULL, *tail = NULL;

    // a / b
    Tensor* div_inputs[] = {a, b};
    struct IRNode* div_node = create_primitive_node(ir, UOP_DIV, div_inputs, 2, NULL, shape, ndim);
    if (!div_node) return -1;
    chain_append(&head, &tail, div_node);

    // floor(a / b)
    struct IRNode* floor_node = create_primitive_node(ir, UOP_FLOOR, &div_node->output, 1, NULL, shape, ndim);
    if (!floor_node) return -1;
    chain_append(&head, &tail, floor_node);

    // floor(a/b) * b
    Tensor* mul_inputs[] = {floor_node->output, b};
    struct IRNode* mul_node = create_primitive_node(ir, UOP_MUL, mul_inputs, 2, NULL, shape, ndim);
    if (!mul_node) return -1;
    chain_append(&head, &tail, mul_node);

    // a - floor(a/b) * b
    Tensor* sub_inputs[] = {a, mul_node->output};
    struct IRNode* sub_node = create_primitive_node(ir, UOP_SUB, sub_inputs, 2, NULL, shape, ndim);
    if (!sub_node) return -1;
    chain_append(&head, &tail, sub_node);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// IDIV: floor(a / b)
static int decompose_idiv(CMLGraph_t ir, struct IRNode* node) {
    Tensor* a = node->inputs[0];
    Tensor* b = node->inputs[1];
    int* shape = node->output ? node->output->shape : node->output_shape;
    int ndim = node->output ? node->output->ndim : node->output_ndim;

    struct IRNode *head = NULL, *tail = NULL;

    Tensor* div_inputs[] = {a, b};
    struct IRNode* div_node = create_primitive_node(ir, UOP_DIV, div_inputs, 2, NULL, shape, ndim);
    if (!div_node) return -1;
    chain_append(&head, &tail, div_node);

    struct IRNode* floor_node = create_primitive_node(ir, UOP_FLOOR, &div_node->output, 1, NULL, shape, ndim);
    if (!floor_node) return -1;
    chain_append(&head, &tail, floor_node);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// COPYSIGN: abs(a) * sign(b) => where(a<0,-a,a) * where(b>0,1,where(b<0,-1,0))
// Simplified: where(b >= 0, abs(a), -abs(a))
static int decompose_copysign(CMLGraph_t ir, struct IRNode* node) {
    Tensor* a = node->inputs[0];
    Tensor* b = node->inputs[1];
    int* shape = node->output ? node->output->shape : node->output_shape;
    int ndim = node->output ? node->output->ndim : node->output_ndim;

    struct IRNode *head = NULL, *tail = NULL;

    // abs(a): where(a < 0, -a, a)
    struct IRNode* zero_node = insert_fill_node(ir, shape, ndim, 0.0f);
    if (!zero_node) return -1;
    chain_append(&head, &tail, zero_node);

    Tensor* cmplt_a[] = {a, zero_node->output};
    struct IRNode* lt_a = create_primitive_node(ir, UOP_CMPLT, cmplt_a, 2, NULL, shape, ndim);
    if (!lt_a) return -1;
    chain_append(&head, &tail, lt_a);

    struct IRNode* neg_a = create_primitive_node(ir, UOP_NEG, &a, 1, NULL, shape, ndim);
    if (!neg_a) return -1;
    chain_append(&head, &tail, neg_a);

    Tensor* abs_inputs[] = {lt_a->output, neg_a->output, a};
    struct IRNode* abs_a = create_primitive_node(ir, UOP_WHERE, abs_inputs, 3, NULL, shape, ndim);
    if (!abs_a) return -1;
    chain_append(&head, &tail, abs_a);

    // neg_abs = -abs(a)
    struct IRNode* neg_abs = create_primitive_node(ir, UOP_NEG, &abs_a->output, 1, NULL, shape, ndim);
    if (!neg_abs) return -1;
    chain_append(&head, &tail, neg_abs);

    // b < 0
    struct IRNode* zero2 = insert_fill_node(ir, shape, ndim, 0.0f);
    if (!zero2) return -1;
    chain_append(&head, &tail, zero2);

    Tensor* cmplt_b[] = {b, zero2->output};
    struct IRNode* lt_b = create_primitive_node(ir, UOP_CMPLT, cmplt_b, 2, NULL, shape, ndim);
    if (!lt_b) return -1;
    chain_append(&head, &tail, lt_b);

    // result = where(b < 0, -abs(a), abs(a))
    Tensor* where_inputs[] = {lt_b->output, neg_abs->output, abs_a->output};
    struct IRNode* result = create_primitive_node(ir, UOP_WHERE, where_inputs, 3, NULL, shape, ndim);
    if (!result) return -1;
    chain_append(&head, &tail, result);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// Phase D — Reductions

// MEAN: sum(x) / n
static int decompose_mean(CMLGraph_t ir, struct IRNode* node) {
    Tensor* x = node->inputs[0];
    int* out_shape = node->output ? node->output->shape : node->output_shape;
    int out_ndim = node->output ? node->output->ndim : node->output_ndim;
    if (!out_shape) return -1;

    struct IRNode *head = NULL, *tail = NULL;

    // Deep copy reduce params for the new SUM node
    ReduceParams* orig_params = (ReduceParams*)node->params;
    ReduceParams* sum_params = malloc(sizeof(ReduceParams));
    if (!sum_params) return -1;
    if (orig_params) {
        sum_params->keepdim = orig_params->keepdim;
        sum_params->num_dims = orig_params->num_dims;
        if (orig_params->dims && orig_params->num_dims > 0) {
            sum_params->dims = malloc((size_t)orig_params->num_dims * sizeof(int));
            if (!sum_params->dims) { free(sum_params); return -1; }
            memcpy(sum_params->dims, orig_params->dims, (size_t)orig_params->num_dims * sizeof(int));
        } else {
            sum_params->dims = NULL;
        }
    } else {
        sum_params->dims = NULL;
        sum_params->num_dims = 0;
        sum_params->keepdim = false;
    }

    // sum(x)
    struct IRNode* sum_node = create_primitive_node(ir, UOP_SUM, &x, 1, sum_params, out_shape, out_ndim);
    if (!sum_node) { free(sum_params->dims); free(sum_params); return -1; }
    chain_append(&head, &tail, sum_node);

    // Compute n = number of elements being reduced
    float n = 1.0f;
    if (orig_params && orig_params->dims && orig_params->num_dims > 0) {
        for (int i = 0; i < orig_params->num_dims; i++) {
            int dim = orig_params->dims[i];
            if (dim < 0) dim += x->ndim;
            if (dim >= 0 && dim < x->ndim) {
                n *= (float)x->shape[dim];
            }
        }
    } else {
        // Reduce all dims
        n = (float)x->numel;
    }

    // 1/n constant
    struct IRNode* inv_n = insert_fill_node(ir, out_shape, out_ndim, 1.0f / n);
    if (!inv_n) return -1;
    chain_append(&head, &tail, inv_n);

    // sum / n
    Tensor* mul_inputs[] = {sum_node->output, inv_n->output};
    struct IRNode* mul_node = create_primitive_node(ir, UOP_MUL, mul_inputs, 2, NULL, out_shape, out_ndim);
    if (!mul_node) return -1;
    chain_append(&head, &tail, mul_node);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// MIN_REDUCE: neg(max_reduce(neg(x)))
static int decompose_min_reduce(CMLGraph_t ir, struct IRNode* node) {
    Tensor* x = node->inputs[0];
    int* out_shape = node->output ? node->output->shape : node->output_shape;
    int out_ndim = node->output ? node->output->ndim : node->output_ndim;
    if (!out_shape) return -1;

    struct IRNode *head = NULL, *tail = NULL;

    // neg(x)
    struct IRNode* neg_node = create_primitive_node(ir, UOP_NEG, &x, 1, NULL, x->shape, x->ndim);
    if (!neg_node) return -1;
    chain_append(&head, &tail, neg_node);

    // Deep copy reduce params
    ReduceParams* orig_params = (ReduceParams*)node->params;
    ReduceParams* max_params = malloc(sizeof(ReduceParams));
    if (!max_params) return -1;
    if (orig_params) {
        max_params->keepdim = orig_params->keepdim;
        max_params->num_dims = orig_params->num_dims;
        if (orig_params->dims && orig_params->num_dims > 0) {
            max_params->dims = malloc((size_t)orig_params->num_dims * sizeof(int));
            if (!max_params->dims) { free(max_params); return -1; }
            memcpy(max_params->dims, orig_params->dims, (size_t)orig_params->num_dims * sizeof(int));
        } else {
            max_params->dims = NULL;
        }
    } else {
        max_params->dims = NULL;
        max_params->num_dims = 0;
        max_params->keepdim = false;
    }

    // max_reduce(neg(x))
    struct IRNode* max_node = create_primitive_node(ir, UOP_MAX_REDUCE, &neg_node->output, 1,
                                                     max_params, out_shape, out_ndim);
    if (!max_node) { free(max_params->dims); free(max_params); return -1; }
    chain_append(&head, &tail, max_node);

    // neg(max_reduce(neg(x)))
    struct IRNode* neg2_node = create_primitive_node(ir, UOP_NEG, &max_node->output, 1, NULL, out_shape, out_ndim);
    if (!neg2_node) return -1;
    chain_append(&head, &tail, neg2_node);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// Phase E — Logical

// LOGICAL_NOT: where(x, 0, 1) — but x is float, so where(x != 0, 0, 1)
// Simplified: we treat nonzero as true. where(x < 0 OR 0 < x, 0, 1)
// Even simpler: use CMPEQ with 0, which gives 1 where x==0 and 0 where x!=0
// But CMPEQ itself gets decomposed. So: where(x < 0, 0, where(0 < x, 0, 1))
static int decompose_logical_not(CMLGraph_t ir, struct IRNode* node) {
    Tensor* x = node->inputs[0];
    int* shape = x->shape;
    int ndim = x->ndim;

    struct IRNode *head = NULL, *tail = NULL;

    struct IRNode* zero_node = insert_fill_node(ir, shape, ndim, 0.0f);
    if (!zero_node) return -1;
    chain_append(&head, &tail, zero_node);

    // x < 0
    Tensor* cmplt1_inputs[] = {x, zero_node->output};
    struct IRNode* lt_neg = create_primitive_node(ir, UOP_CMPLT, cmplt1_inputs, 2, NULL, shape, ndim);
    if (!lt_neg) return -1;
    chain_append(&head, &tail, lt_neg);

    // 0 < x
    Tensor* cmplt2_inputs[] = {zero_node->output, x};
    struct IRNode* lt_pos = create_primitive_node(ir, UOP_CMPLT, cmplt2_inputs, 2, NULL, shape, ndim);
    if (!lt_pos) return -1;
    chain_append(&head, &tail, lt_pos);

    struct IRNode* zero2 = insert_fill_node(ir, shape, ndim, 0.0f);
    if (!zero2) return -1;
    chain_append(&head, &tail, zero2);

    struct IRNode* one_node = insert_fill_node(ir, shape, ndim, 1.0f);
    if (!one_node) return -1;
    chain_append(&head, &tail, one_node);

    // inner = where(0 < x, 0, 1)
    Tensor* inner_inputs[] = {lt_pos->output, zero2->output, one_node->output};
    struct IRNode* inner = create_primitive_node(ir, UOP_WHERE, inner_inputs, 3, NULL, shape, ndim);
    if (!inner) return -1;
    chain_append(&head, &tail, inner);

    struct IRNode* zero3 = insert_fill_node(ir, shape, ndim, 0.0f);
    if (!zero3) return -1;
    chain_append(&head, &tail, zero3);

    // result = where(x < 0, 0, inner)
    Tensor* outer_inputs[] = {lt_neg->output, zero3->output, inner->output};
    struct IRNode* outer = create_primitive_node(ir, UOP_WHERE, outer_inputs, 3, NULL, shape, ndim);
    if (!outer) return -1;
    chain_append(&head, &tail, outer);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// LOGICAL_AND: where(a, where(b, 1, 0), 0)
// Since a/b are float, nonzero = true. Use cmpne with 0.
// Simplified: where(a != 0, where(b != 0, 1, 0), 0)
// But cmpne gets decomposed. Use: treat a directly as condition for WHERE
// (WHERE checks cond != 0), so: where(a, where(b, 1, 0), 0) works directly
static int decompose_logical_and(CMLGraph_t ir, struct IRNode* node) {
    Tensor* a = node->inputs[0];
    Tensor* b = node->inputs[1];
    int* shape = node->output ? node->output->shape : node->output_shape;
    int ndim = node->output ? node->output->ndim : node->output_ndim;

    struct IRNode *head = NULL, *tail = NULL;

    struct IRNode* one_node = insert_fill_node(ir, shape, ndim, 1.0f);
    if (!one_node) return -1;
    chain_append(&head, &tail, one_node);

    struct IRNode* zero_node = insert_fill_node(ir, shape, ndim, 0.0f);
    if (!zero_node) return -1;
    chain_append(&head, &tail, zero_node);

    // inner = where(b, 1, 0)
    Tensor* inner_inputs[] = {b, one_node->output, zero_node->output};
    struct IRNode* inner = create_primitive_node(ir, UOP_WHERE, inner_inputs, 3, NULL, shape, ndim);
    if (!inner) return -1;
    chain_append(&head, &tail, inner);

    struct IRNode* zero2 = insert_fill_node(ir, shape, ndim, 0.0f);
    if (!zero2) return -1;
    chain_append(&head, &tail, zero2);

    // result = where(a, inner, 0)
    Tensor* outer_inputs[] = {a, inner->output, zero2->output};
    struct IRNode* outer = create_primitive_node(ir, UOP_WHERE, outer_inputs, 3, NULL, shape, ndim);
    if (!outer) return -1;
    chain_append(&head, &tail, outer);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// LOGICAL_OR: where(a, 1, where(b, 1, 0))
static int decompose_logical_or(CMLGraph_t ir, struct IRNode* node) {
    Tensor* a = node->inputs[0];
    Tensor* b = node->inputs[1];
    int* shape = node->output ? node->output->shape : node->output_shape;
    int ndim = node->output ? node->output->ndim : node->output_ndim;

    struct IRNode *head = NULL, *tail = NULL;

    struct IRNode* one_node = insert_fill_node(ir, shape, ndim, 1.0f);
    if (!one_node) return -1;
    chain_append(&head, &tail, one_node);

    struct IRNode* zero_node = insert_fill_node(ir, shape, ndim, 0.0f);
    if (!zero_node) return -1;
    chain_append(&head, &tail, zero_node);

    // inner = where(b, 1, 0)
    Tensor* inner_inputs[] = {b, one_node->output, zero_node->output};
    struct IRNode* inner = create_primitive_node(ir, UOP_WHERE, inner_inputs, 3, NULL, shape, ndim);
    if (!inner) return -1;
    chain_append(&head, &tail, inner);

    struct IRNode* one2 = insert_fill_node(ir, shape, ndim, 1.0f);
    if (!one2) return -1;
    chain_append(&head, &tail, one2);

    // result = where(a, 1, inner)
    Tensor* outer_inputs[] = {a, one2->output, inner->output};
    struct IRNode* outer = create_primitive_node(ir, UOP_WHERE, outer_inputs, 3, NULL, shape, ndim);
    if (!outer) return -1;
    chain_append(&head, &tail, outer);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// CLAMP: where(x < min, min, where(x > max, max, x))
//      = where(x < min, min, where(max < x, max, x))
static int decompose_clamp(CMLGraph_t ir, struct IRNode* node) {
    Tensor* x = node->inputs[0];
    int* shape = x->shape;
    int ndim = x->ndim;

    ClampParams* params = (ClampParams*)node->params;
    float min_val = params ? params->min_val : -INFINITY;
    float max_val = params ? params->max_val : INFINITY;

    struct IRNode *head = NULL, *tail = NULL;

    struct IRNode* max_const = insert_fill_node(ir, shape, ndim, max_val);
    if (!max_const) return -1;
    chain_append(&head, &tail, max_const);

    // max < x  (i.e., x > max)
    Tensor* cmpgt_inputs[] = {max_const->output, x};
    struct IRNode* gt_max = create_primitive_node(ir, UOP_CMPLT, cmpgt_inputs, 2, NULL, shape, ndim);
    if (!gt_max) return -1;
    chain_append(&head, &tail, gt_max);

    // inner = where(x > max, max, x)
    Tensor* inner_inputs[] = {gt_max->output, max_const->output, x};
    struct IRNode* inner = create_primitive_node(ir, UOP_WHERE, inner_inputs, 3, NULL, shape, ndim);
    if (!inner) return -1;
    chain_append(&head, &tail, inner);

    struct IRNode* min_const = insert_fill_node(ir, shape, ndim, min_val);
    if (!min_const) return -1;
    chain_append(&head, &tail, min_const);

    // x < min
    Tensor* cmplt_inputs[] = {x, min_const->output};
    struct IRNode* lt_min = create_primitive_node(ir, UOP_CMPLT, cmplt_inputs, 2, NULL, shape, ndim);
    if (!lt_min) return -1;
    chain_append(&head, &tail, lt_min);

    // result = where(x < min, min, inner)
    Tensor* outer_inputs[] = {lt_min->output, min_const->output, inner->output};
    struct IRNode* outer = create_primitive_node(ir, UOP_WHERE, outer_inputs, 3, NULL, shape, ndim);
    if (!outer) return -1;
    chain_append(&head, &tail, outer);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// SINH: (exp(x) - exp(-x)) / 2
static int decompose_sinh(CMLGraph_t ir, struct IRNode* node) {
    Tensor* x = node->inputs[0];
    int* shape = x->shape;
    int ndim = x->ndim;

    struct IRNode *head = NULL, *tail = NULL;

    struct IRNode* exp_x = create_primitive_node(ir, UOP_EXP, &x, 1, NULL, shape, ndim);
    if (!exp_x) return -1;
    chain_append(&head, &tail, exp_x);

    struct IRNode* neg_x = create_primitive_node(ir, UOP_NEG, &x, 1, NULL, shape, ndim);
    if (!neg_x) return -1;
    chain_append(&head, &tail, neg_x);

    struct IRNode* exp_neg_x = create_primitive_node(ir, UOP_EXP, &neg_x->output, 1, NULL, shape, ndim);
    if (!exp_neg_x) return -1;
    chain_append(&head, &tail, exp_neg_x);

    Tensor* sub_inputs[] = {exp_x->output, exp_neg_x->output};
    struct IRNode* sub_node = create_primitive_node(ir, UOP_SUB, sub_inputs, 2, NULL, shape, ndim);
    if (!sub_node) return -1;
    chain_append(&head, &tail, sub_node);

    struct IRNode* half_node = insert_fill_node(ir, shape, ndim, 0.5f);
    if (!half_node) return -1;
    chain_append(&head, &tail, half_node);

    Tensor* mul_inputs[] = {sub_node->output, half_node->output};
    struct IRNode* mul_node = create_primitive_node(ir, UOP_MUL, mul_inputs, 2, NULL, shape, ndim);
    if (!mul_node) return -1;
    chain_append(&head, &tail, mul_node);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// COSH: (exp(x) + exp(-x)) / 2
static int decompose_cosh(CMLGraph_t ir, struct IRNode* node) {
    Tensor* x = node->inputs[0];
    int* shape = x->shape;
    int ndim = x->ndim;

    struct IRNode *head = NULL, *tail = NULL;

    struct IRNode* exp_x = create_primitive_node(ir, UOP_EXP, &x, 1, NULL, shape, ndim);
    if (!exp_x) return -1;
    chain_append(&head, &tail, exp_x);

    struct IRNode* neg_x = create_primitive_node(ir, UOP_NEG, &x, 1, NULL, shape, ndim);
    if (!neg_x) return -1;
    chain_append(&head, &tail, neg_x);

    struct IRNode* exp_neg_x = create_primitive_node(ir, UOP_EXP, &neg_x->output, 1, NULL, shape, ndim);
    if (!exp_neg_x) return -1;
    chain_append(&head, &tail, exp_neg_x);

    Tensor* add_inputs[] = {exp_x->output, exp_neg_x->output};
    struct IRNode* add_node = create_primitive_node(ir, UOP_ADD, add_inputs, 2, NULL, shape, ndim);
    if (!add_node) return -1;
    chain_append(&head, &tail, add_node);

    struct IRNode* half_node = insert_fill_node(ir, shape, ndim, 0.5f);
    if (!half_node) return -1;
    chain_append(&head, &tail, half_node);

    Tensor* mul_inputs[] = {add_node->output, half_node->output};
    struct IRNode* mul_node = create_primitive_node(ir, UOP_MUL, mul_inputs, 2, NULL, shape, ndim);
    if (!mul_node) return -1;
    chain_append(&head, &tail, mul_node);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// ATANH: 0.5 * log((1+x)/(1-x))
static int decompose_atanh(CMLGraph_t ir, struct IRNode* node) {
    Tensor* x = node->inputs[0];
    int* shape = x->shape;
    int ndim = x->ndim;

    struct IRNode *head = NULL, *tail = NULL;

    struct IRNode* one_node = insert_fill_node(ir, shape, ndim, 1.0f);
    if (!one_node) return -1;
    chain_append(&head, &tail, one_node);

    // 1 + x
    Tensor* add_inputs[] = {one_node->output, x};
    struct IRNode* add_node = create_primitive_node(ir, UOP_ADD, add_inputs, 2, NULL, shape, ndim);
    if (!add_node) return -1;
    chain_append(&head, &tail, add_node);

    struct IRNode* one2 = insert_fill_node(ir, shape, ndim, 1.0f);
    if (!one2) return -1;
    chain_append(&head, &tail, one2);

    // 1 - x
    Tensor* sub_inputs[] = {one2->output, x};
    struct IRNode* sub_node = create_primitive_node(ir, UOP_SUB, sub_inputs, 2, NULL, shape, ndim);
    if (!sub_node) return -1;
    chain_append(&head, &tail, sub_node);

    // (1+x) / (1-x)
    Tensor* div_inputs[] = {add_node->output, sub_node->output};
    struct IRNode* div_node = create_primitive_node(ir, UOP_DIV, div_inputs, 2, NULL, shape, ndim);
    if (!div_node) return -1;
    chain_append(&head, &tail, div_node);

    // log((1+x)/(1-x))
    struct IRNode* log_node = create_primitive_node(ir, UOP_LOG, &div_node->output, 1, NULL, shape, ndim);
    if (!log_node) return -1;
    chain_append(&head, &tail, log_node);

    // 0.5 * log(...)
    struct IRNode* half = insert_fill_node(ir, shape, ndim, 0.5f);
    if (!half) return -1;
    chain_append(&head, &tail, half);

    Tensor* mul_inputs[] = {half->output, log_node->output};
    struct IRNode* mul_node = create_primitive_node(ir, UOP_MUL, mul_inputs, 2, NULL, shape, ndim);
    if (!mul_node) return -1;
    chain_append(&head, &tail, mul_node);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// SOFTPLUS: log(1 + exp(x))
static int decompose_softplus(CMLGraph_t ir, struct IRNode* node) {
    Tensor* x = node->inputs[0];
    int* shape = x->shape;
    int ndim = x->ndim;

    struct IRNode *head = NULL, *tail = NULL;

    struct IRNode* exp_node = create_primitive_node(ir, UOP_EXP, &x, 1, NULL, shape, ndim);
    if (!exp_node) return -1;
    chain_append(&head, &tail, exp_node);

    struct IRNode* one_node = insert_fill_node(ir, shape, ndim, 1.0f);
    if (!one_node) return -1;
    chain_append(&head, &tail, one_node);

    Tensor* add_inputs[] = {one_node->output, exp_node->output};
    struct IRNode* add_node = create_primitive_node(ir, UOP_ADD, add_inputs, 2, NULL, shape, ndim);
    if (!add_node) return -1;
    chain_append(&head, &tail, add_node);

    struct IRNode* log_node = create_primitive_node(ir, UOP_LOG, &add_node->output, 1, NULL, shape, ndim);
    if (!log_node) return -1;
    chain_append(&head, &tail, log_node);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// LOGSIGMOID: log(sigmoid(x)) = -softplus(-x) = -log(1 + exp(-x))
static int decompose_logsigmoid(CMLGraph_t ir, struct IRNode* node) {
    Tensor* x = node->inputs[0];
    int* shape = x->shape;
    int ndim = x->ndim;

    struct IRNode *head = NULL, *tail = NULL;

    struct IRNode* neg_x = create_primitive_node(ir, UOP_NEG, &x, 1, NULL, shape, ndim);
    if (!neg_x) return -1;
    chain_append(&head, &tail, neg_x);

    struct IRNode* exp_node = create_primitive_node(ir, UOP_EXP, &neg_x->output, 1, NULL, shape, ndim);
    if (!exp_node) return -1;
    chain_append(&head, &tail, exp_node);

    struct IRNode* one_node = insert_fill_node(ir, shape, ndim, 1.0f);
    if (!one_node) return -1;
    chain_append(&head, &tail, one_node);

    Tensor* add_inputs[] = {one_node->output, exp_node->output};
    struct IRNode* add_node = create_primitive_node(ir, UOP_ADD, add_inputs, 2, NULL, shape, ndim);
    if (!add_node) return -1;
    chain_append(&head, &tail, add_node);

    struct IRNode* log_node = create_primitive_node(ir, UOP_LOG, &add_node->output, 1, NULL, shape, ndim);
    if (!log_node) return -1;
    chain_append(&head, &tail, log_node);

    struct IRNode* neg_result = create_primitive_node(ir, UOP_NEG, &log_node->output, 1, NULL, shape, ndim);
    if (!neg_result) return -1;
    chain_append(&head, &tail, neg_result);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// ERFC: 1 - erf(x)
static int decompose_erfc(CMLGraph_t ir, struct IRNode* node) {
    Tensor* x = node->inputs[0];
    int* shape = x->shape;
    int ndim = x->ndim;

    struct IRNode *head = NULL, *tail = NULL;

    struct IRNode* erf_node = create_primitive_node(ir, UOP_ERF, &x, 1, NULL, shape, ndim);
    if (!erf_node) return -1;
    chain_append(&head, &tail, erf_node);

    struct IRNode* one_node = insert_fill_node(ir, shape, ndim, 1.0f);
    if (!one_node) return -1;
    chain_append(&head, &tail, one_node);

    Tensor* sub_inputs[] = {one_node->output, erf_node->output};
    struct IRNode* sub_node = create_primitive_node(ir, UOP_SUB, sub_inputs, 2, NULL, shape, ndim);
    if (!sub_node) return -1;
    chain_append(&head, &tail, sub_node);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// Main Decomposition Pass

int cml_ir_decompose(CMLGraph_t ir) {
    if (!ir) return -1;
    if (ir->is_decomposed) return 0;

    LOG_DEBUG("Running IR decomposition pass");

    struct IRNode* node = ir->head;

    while (node) {
        struct IRNode* next = node->next; // Save next before potential replacement
        int result = 0;
        bool was_decomposed = false;

        switch (node->type) {
        // Phase A — Activations
        case UOP_SIGMOID:
            result = decompose_sigmoid(ir, node);
            was_decomposed = true;
            break;
        case UOP_TANH:
            result = decompose_tanh(ir, node);
            break;
        case UOP_SQUARE:
            result = decompose_square(ir, node);
            break;
        case UOP_RSQRT:
            result = decompose_rsqrt(ir, node);
            break;
        case UOP_ABS:
            result = decompose_abs(ir, node);
            break;

        // Phase B — Comparisons
        case UOP_CMPEQ:
            result = decompose_cmpeq(ir, node);
            break;
        case UOP_CMPNE:
            result = decompose_cmpne(ir, node);
            break;
        case UOP_CMPLE:
            result = decompose_cmple(ir, node);
            break;
        case UOP_CMPGT:
            result = decompose_cmpgt(ir, node);
            break;
        case UOP_CMPGE:
            result = decompose_cmpge(ir, node);
            break;
        case UOP_MINIMUM:
            result = decompose_minimum(ir, node);
            break;

        // Phase C — Math
        case UOP_COS:
            result = decompose_cos(ir, node);
            break;
        case UOP_TAN:
            result = decompose_tan(ir, node);
            break;
        case UOP_LOG2:
            result = decompose_log2(ir, node);
            break;
        case UOP_EXP2:
            result = decompose_exp2(ir, node);
            break;
        case UOP_LOG10:
            result = decompose_log10(ir, node);
            break;
        case UOP_LOGADDEXP:
            result = decompose_logaddexp(ir, node);
            break;
        case UOP_MOD:
            result = decompose_mod(ir, node);
            break;
        case UOP_IDIV:
            result = decompose_idiv(ir, node);
            break;
        case UOP_COPYSIGN:
            result = decompose_copysign(ir, node);
            break;
        case UOP_SIGN:
            result = decompose_sign(ir, node);
            break;
        case UOP_SINH:
            result = decompose_sinh(ir, node);
            break;
        case UOP_COSH:
            result = decompose_cosh(ir, node);
            break;
        case UOP_ATANH:
            result = decompose_atanh(ir, node);
            break;

        // Phase D — Reductions
        case UOP_MEAN:
            result = decompose_mean(ir, node);
            break;
        case UOP_MIN_REDUCE:
            result = decompose_min_reduce(ir, node);
            break;

        // Phase E — Logical & misc
        case UOP_LOGICAL_NOT:
            result = decompose_logical_not(ir, node);
            break;
        case UOP_LOGICAL_AND:
            result = decompose_logical_and(ir, node);
            break;
        case UOP_LOGICAL_OR:
            result = decompose_logical_or(ir, node);
            break;
        case UOP_CLAMP:
            result = decompose_clamp(ir, node);
            break;
        case UOP_SOFTPLUS:
            result = decompose_softplus(ir, node);
            break;
        case UOP_LOGSIGMOID:
            result = decompose_logsigmoid(ir, node);
            break;
        case UOP_ERFC:
            result = decompose_erfc(ir, node);
            break;

        default:
            // Not a composite op — keep as-is
            break;
        }

        if (result != 0 && was_decomposed) {
            LOG_WARNING("Failed to decompose op type %d", node->type);
        }
        (void)was_decomposed;
        (void)result;

        // After decomposition, node is freed and replaced.
        // The chain's tail->next was set to 'next' in replace_node_with_chain.
        node = next;
    }

    ir->is_decomposed = true;
    LOG_DEBUG("IR decomposition pass complete");
    return 0;
}
