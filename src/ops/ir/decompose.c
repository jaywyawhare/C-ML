/* Rewrites composite ops into ~28 primitives so backends handle a minimal set. */

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
#include "alloc/cml_allocator.h"

static atomic_int g_decompose_counter = 0;

static char* decompose_unique_name(void);
static struct IRNode* create_primitive_node(CMLGraph_t ir, UOpType type,
                                            Tensor** inputs, int num_inputs,
                                            void* params, int* out_shape,
                                            int out_ndim);
static struct IRNode* insert_fill_node(CMLGraph_t ir, int* shape, int ndim, float value);
static void replace_node_with_chain(CMLGraph_t ir, struct IRNode* original,
                                    struct IRNode* chain_head, struct IRNode* chain_tail);

static char* decompose_unique_name(void) {
    int id = atomic_fetch_add(&g_decompose_counter, 1);
    char* name = cml_malloc(32);
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
    struct IRNode* node = cml_calloc(1, sizeof(struct IRNode));
    if (!node) return NULL;

    node->type = type;
    node->num_inputs = num_inputs;
    node->params = params;

    if (num_inputs > 0 && inputs) {
        node->input_names = cml_malloc((size_t)num_inputs * sizeof(char*));
        node->inputs = cml_malloc((size_t)num_inputs * sizeof(Tensor*));
        if (!node->input_names || !node->inputs) {
            cml_free(node->input_names);
            cml_free(node->inputs);
            cml_free(node);
            return NULL;
        }
        for (int i = 0; i < num_inputs; i++) {
            node->inputs[i] = inputs[i];
            if (inputs[i] && inputs[i]->ir_node &&
                ((struct IRNode*)inputs[i]->ir_node)->output_name) {
                node->input_names[i] = cml_strdup(((struct IRNode*)inputs[i]->ir_node)->output_name);
            } else {
                node->input_names[i] = decompose_unique_name();
            }
        }
    } else {
        node->input_names = NULL;
        node->inputs = NULL;
    }

    node->output_name = decompose_unique_name();
    node->next = NULL;

    if (out_shape && out_ndim > 0) {
        node->output_shape = cml_malloc((size_t)out_ndim * sizeof(int));
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

    Tensor* out_tensor = cml_calloc(1, sizeof(Tensor));
    if (!out_tensor) {
        for (int i = 0; i < num_inputs; i++) cml_free(node->input_names[i]);
        cml_free(node->input_names);
        cml_free(node->inputs);
        cml_free(node->output_name);
        cml_free(node->output_shape);
        cml_free(node);
        return NULL;
    }

    out_tensor->ir_node = node;
    out_tensor->ir_context = ir;
    node->output = out_tensor;

    if (out_shape && out_ndim > 0) {
        out_tensor->shape = cml_malloc((size_t)out_ndim * sizeof(int));
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

    if (out_ndim > 0 && out_tensor->shape) {
        out_tensor->strides = cml_malloc((size_t)out_ndim * sizeof(size_t));
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
    FillParams* params = cml_malloc(sizeof(FillParams));
    if (!params) return NULL;
    params->value = value;
    params->ndim = ndim;
    params->shape = cml_malloc((size_t)ndim * sizeof(int));
    if (!params->shape) { cml_free(params); return NULL; }
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
        if (chain_output->shape) cml_free(chain_output->shape);
        if (chain_output->strides) cml_free(chain_output->strides);
        cml_free(chain_output);

        // Point chain_tail at the original output tensor
        chain_tail->output = orig_output;

        if (chain_tail->output_name) cml_free(chain_tail->output_name);
        chain_tail->output_name = cml_strdup(original->output_name ? original->output_name : "_decomp");
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

    if (ir->tail == original) {
        ir->tail = chain_tail;
    }

    ir->node_count += (chain_len - 1);

    // Free the original node (but NOT its output tensor — we kept it)
    original->output = NULL; // Prevent double-free
    if (original->input_names) {
        for (int i = 0; i < original->num_inputs; i++) {
            cml_free(original->input_names[i]);
        }
        cml_free(original->input_names);
    }
    cml_free(original->inputs);
    cml_free(original->output_name);
    cml_free(original->output_shape);
    cml_ir_free_node_params(original);
    cml_free(original);
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

// RELU: max(x, 0)
static int decompose_relu(CMLGraph_t ir, struct IRNode* node) {
    Tensor* x = node->inputs[0];
    int* shape = x->shape;
    int ndim = x->ndim;

    struct IRNode *head = NULL, *tail = NULL;

    struct IRNode* zero_node = insert_fill_node(ir, shape, ndim, 0.0f);
    if (!zero_node) return -1;
    chain_append(&head, &tail, zero_node);

    Tensor* max_inputs[] = {x, zero_node->output};
    struct IRNode* max_node = create_primitive_node(ir, UOP_MAX, max_inputs, 2, NULL, shape, ndim);
    if (!max_node) return -1;
    chain_append(&head, &tail, max_node);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// SILU / SWISH: x * sigmoid(x) = x * recip(1 + exp(-x))
static int decompose_silu(CMLGraph_t ir, struct IRNode* node) {
    Tensor* x = node->inputs[0];
    int* shape = x->shape;
    int ndim = x->ndim;
    struct IRNode *head = NULL, *tail = NULL;

    struct IRNode* neg = create_primitive_node(ir, UOP_NEG, &x, 1, NULL, shape, ndim);
    if (!neg) return -1;
    chain_append(&head, &tail, neg);

    struct IRNode* e = create_primitive_node(ir, UOP_EXP, &neg->output, 1, NULL, shape, ndim);
    if (!e) return -1;
    chain_append(&head, &tail, e);

    struct IRNode* one = insert_fill_node(ir, shape, ndim, 1.0f);
    if (!one) return -1;
    chain_append(&head, &tail, one);

    Tensor* add_in[] = {one->output, e->output};
    struct IRNode* denom = create_primitive_node(ir, UOP_ADD, add_in, 2, NULL, shape, ndim);
    if (!denom) return -1;
    chain_append(&head, &tail, denom);

    struct IRNode* sig = create_primitive_node(ir, UOP_RECIP, &denom->output, 1, NULL, shape, ndim);
    if (!sig) return -1;
    chain_append(&head, &tail, sig);

    Tensor* mul_in[] = {x, sig->output};
    struct IRNode* result = create_primitive_node(ir, UOP_MUL, mul_in, 2, NULL, shape, ndim);
    if (!result) return -1;
    chain_append(&head, &tail, result);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// RELU6: min(max(x, 0), 6) = where(max(x,0) < 6, max(x,0), 6)
static int decompose_relu6(CMLGraph_t ir, struct IRNode* node) {
    Tensor* x = node->inputs[0];
    int* shape = x->shape;
    int ndim = x->ndim;
    struct IRNode *head = NULL, *tail = NULL;

    struct IRNode* zero = insert_fill_node(ir, shape, ndim, 0.0f);
    if (!zero) return -1;
    chain_append(&head, &tail, zero);

    Tensor* max_in[] = {x, zero->output};
    struct IRNode* relu = create_primitive_node(ir, UOP_MAX, max_in, 2, NULL, shape, ndim);
    if (!relu) return -1;
    chain_append(&head, &tail, relu);

    struct IRNode* six = insert_fill_node(ir, shape, ndim, 6.0f);
    if (!six) return -1;
    chain_append(&head, &tail, six);

    Tensor* cmp_in[] = {relu->output, six->output};
    struct IRNode* cond = create_primitive_node(ir, UOP_CMPLT, cmp_in, 2, NULL, shape, ndim);
    if (!cond) return -1;
    chain_append(&head, &tail, cond);

    Tensor* where_in[] = {cond->output, relu->output, six->output};
    struct IRNode* result = create_primitive_node(ir, UOP_WHERE, where_in, 3, NULL, shape, ndim);
    if (!result) return -1;
    chain_append(&head, &tail, result);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// HARDSWISH: x * clamp(x+3, 0, 6) / 6
static int decompose_hardswish(CMLGraph_t ir, struct IRNode* node) {
    Tensor* x = node->inputs[0];
    int* s = x->shape; int nd = x->ndim;
    struct IRNode *h = NULL, *t = NULL;

    struct IRNode* three = insert_fill_node(ir, s, nd, 3.0f); if (!three) return -1; chain_append(&h, &t, three);
    Tensor* a1[] = {x, three->output};
    struct IRNode* xp3 = create_primitive_node(ir, UOP_ADD, a1, 2, NULL, s, nd); if (!xp3) return -1; chain_append(&h, &t, xp3);
    struct IRNode* zero = insert_fill_node(ir, s, nd, 0.0f); if (!zero) return -1; chain_append(&h, &t, zero);
    Tensor* m1[] = {xp3->output, zero->output};
    struct IRNode* r = create_primitive_node(ir, UOP_MAX, m1, 2, NULL, s, nd); if (!r) return -1; chain_append(&h, &t, r);
    struct IRNode* six = insert_fill_node(ir, s, nd, 6.0f); if (!six) return -1; chain_append(&h, &t, six);
    Tensor* c1[] = {r->output, six->output};
    struct IRNode* cond = create_primitive_node(ir, UOP_CMPLT, c1, 2, NULL, s, nd); if (!cond) return -1; chain_append(&h, &t, cond);
    Tensor* w1[] = {cond->output, r->output, six->output};
    struct IRNode* clamp = create_primitive_node(ir, UOP_WHERE, w1, 3, NULL, s, nd); if (!clamp) return -1; chain_append(&h, &t, clamp);
    struct IRNode* sixth = insert_fill_node(ir, s, nd, 1.0f / 6.0f); if (!sixth) return -1; chain_append(&h, &t, sixth);
    Tensor* mm[] = {x, clamp->output};
    struct IRNode* xc = create_primitive_node(ir, UOP_MUL, mm, 2, NULL, s, nd); if (!xc) return -1; chain_append(&h, &t, xc);
    Tensor* mm2[] = {xc->output, sixth->output};
    struct IRNode* res = create_primitive_node(ir, UOP_MUL, mm2, 2, NULL, s, nd); if (!res) return -1; chain_append(&h, &t, res);

    replace_node_with_chain(ir, node, h, t);
    return 0;
}

// MISH: x * tanh(softplus(x)); softplus = log(1+exp(x)); tanh(y) = 2/(1+exp(-2y)) - 1
static int decompose_mish(CMLGraph_t ir, struct IRNode* node) {
    Tensor* x = node->inputs[0];
    int* s = x->shape; int nd = x->ndim;
    struct IRNode *h = NULL, *t = NULL;

    struct IRNode* ex = create_primitive_node(ir, UOP_EXP, &x, 1, NULL, s, nd); if (!ex) return -1; chain_append(&h, &t, ex);
    struct IRNode* one = insert_fill_node(ir, s, nd, 1.0f); if (!one) return -1; chain_append(&h, &t, one);
    Tensor* a1[] = {one->output, ex->output};
    struct IRNode* sp1 = create_primitive_node(ir, UOP_ADD, a1, 2, NULL, s, nd); if (!sp1) return -1; chain_append(&h, &t, sp1);
    struct IRNode* sp = create_primitive_node(ir, UOP_LOG, &sp1->output, 1, NULL, s, nd); if (!sp) return -1; chain_append(&h, &t, sp);
    struct IRNode* two = insert_fill_node(ir, s, nd, 2.0f); if (!two) return -1; chain_append(&h, &t, two);
    Tensor* m1[] = {two->output, sp->output};
    struct IRNode* twosp = create_primitive_node(ir, UOP_MUL, m1, 2, NULL, s, nd); if (!twosp) return -1; chain_append(&h, &t, twosp);
    struct IRNode* n = create_primitive_node(ir, UOP_NEG, &twosp->output, 1, NULL, s, nd); if (!n) return -1; chain_append(&h, &t, n);
    struct IRNode* en = create_primitive_node(ir, UOP_EXP, &n->output, 1, NULL, s, nd); if (!en) return -1; chain_append(&h, &t, en);
    Tensor* a2[] = {one->output, en->output};
    struct IRNode* d = create_primitive_node(ir, UOP_ADD, a2, 2, NULL, s, nd); if (!d) return -1; chain_append(&h, &t, d);
    struct IRNode* rc = create_primitive_node(ir, UOP_RECIP, &d->output, 1, NULL, s, nd); if (!rc) return -1; chain_append(&h, &t, rc);
    Tensor* m2[] = {two->output, rc->output};
    struct IRNode* twosig = create_primitive_node(ir, UOP_MUL, m2, 2, NULL, s, nd); if (!twosig) return -1; chain_append(&h, &t, twosig);
    Tensor* sub1[] = {twosig->output, one->output};
    struct IRNode* th = create_primitive_node(ir, UOP_SUB, sub1, 2, NULL, s, nd); if (!th) return -1; chain_append(&h, &t, th);
    Tensor* mm[] = {x, th->output};
    struct IRNode* res = create_primitive_node(ir, UOP_MUL, mm, 2, NULL, s, nd); if (!res) return -1; chain_append(&h, &t, res);

    replace_node_with_chain(ir, node, h, t);
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


// MEAN: sum(x) / n
static int decompose_mean(CMLGraph_t ir, struct IRNode* node) {
    Tensor* x = node->inputs[0];
    int* out_shape = node->output ? node->output->shape : node->output_shape;
    int out_ndim = node->output ? node->output->ndim : node->output_ndim;
    if (!out_shape) return -1;

    struct IRNode *head = NULL, *tail = NULL;

    // Deep copy reduce params for the new SUM node
    ReduceParams* orig_params = (ReduceParams*)node->params;
    ReduceParams* sum_params = cml_malloc(sizeof(ReduceParams));
    if (!sum_params) return -1;
    if (orig_params) {
        sum_params->keepdim = orig_params->keepdim;
        sum_params->num_dims = orig_params->num_dims;
        if (orig_params->dims && orig_params->num_dims > 0) {
            sum_params->dims = cml_malloc((size_t)orig_params->num_dims * sizeof(int));
            if (!sum_params->dims) { cml_free(sum_params); return -1; }
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
    if (!sum_node) { cml_free(sum_params->dims); cml_free(sum_params); return -1; }
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
    ReduceParams* max_params = cml_malloc(sizeof(ReduceParams));
    if (!max_params) return -1;
    if (orig_params) {
        max_params->keepdim = orig_params->keepdim;
        max_params->num_dims = orig_params->num_dims;
        if (orig_params->dims && orig_params->num_dims > 0) {
            max_params->dims = cml_malloc((size_t)orig_params->num_dims * sizeof(int));
            if (!max_params->dims) { cml_free(max_params); return -1; }
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
    if (!max_node) { cml_free(max_params->dims); cml_free(max_params); return -1; }
    chain_append(&head, &tail, max_node);

    // neg(max_reduce(neg(x)))
    struct IRNode* neg2_node = create_primitive_node(ir, UOP_NEG, &max_node->output, 1, NULL, out_shape, out_ndim);
    if (!neg2_node) return -1;
    chain_append(&head, &tail, neg2_node);

    replace_node_with_chain(ir, node, head, tail);
    return 0;
}


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

// malloc a ReduceParams copy from src with a chosen keepdim (node takes ownership)
static ReduceParams* dup_reduce_params(const ReduceParams* src, bool keepdim) {
    ReduceParams* p = cml_malloc(sizeof(ReduceParams));
    if (!p) return NULL;
    p->keepdim = keepdim;
    if (src && src->dims && src->num_dims > 0) {
        p->num_dims = src->num_dims;
        p->dims = cml_malloc((size_t)src->num_dims * sizeof(int));
        if (!p->dims) { cml_free(p); return NULL; }
        memcpy(p->dims, src->dims, (size_t)src->num_dims * sizeof(int));
    } else {
        p->dims = NULL; p->num_dims = 0;
    }
    return p;
}

// Build the biased variance sub-chain: mean((x - mean(x,dims,keepdim))^2, dims).
// Appends nodes to head/tail and returns the final variance node (NULL on error).
static struct IRNode* build_variance(CMLGraph_t ir, struct IRNode* node,
                                     struct IRNode** head, struct IRNode** tail) {
    Tensor* x = node->inputs[0];
    int xnd = x->ndim;
    if (xnd < 1 || xnd > 15) return NULL;
    int* out_shape = node->output ? node->output->shape : node->output_shape;
    int out_ndim = node->output ? node->output->ndim : node->output_ndim;
    if (!out_shape) return NULL;
    ReduceParams* rp = (ReduceParams*)node->params;

    int keep_shape[16], full_shape[16];
    for (int i = 0; i < xnd; i++) { keep_shape[i] = x->shape[i]; full_shape[i] = x->shape[i]; }
    float n = 1.0f;
    if (rp && rp->dims && rp->num_dims > 0) {
        for (int i = 0; i < rp->num_dims; i++) {
            int d = rp->dims[i]; if (d < 0) d += xnd;
            if (d >= 0 && d < xnd) { n *= (float)x->shape[d]; keep_shape[d] = 1; }
        }
    } else {
        for (int i = 0; i < xnd; i++) keep_shape[i] = 1;
        n = (float)x->numel;
    }
    float inv_n = (n > 0) ? 1.0f / n : 0.0f;

    // mean1 = sum(x, dims, keepdim=true) * (1/n)  -> shape keep_shape
    ReduceParams* sp1 = dup_reduce_params(rp, true);
    if (!sp1) return NULL;
    struct IRNode* sum1 = create_primitive_node(ir, UOP_SUM, &x, 1, sp1, keep_shape, xnd);
    if (!sum1) return NULL; chain_append(head, tail, sum1);
    struct IRNode* invn1 = insert_fill_node(ir, keep_shape, xnd, inv_n);
    if (!invn1) return NULL; chain_append(head, tail, invn1);
    Tensor* m1[] = {sum1->output, invn1->output};
    struct IRNode* mean1 = create_primitive_node(ir, UOP_MUL, m1, 2, NULL, keep_shape, xnd);
    if (!mean1) return NULL; chain_append(head, tail, mean1);

    // broadcast mean1 to x's shape (binary ops don't broadcast on their own)
    ExpandParams* ep = cml_malloc(sizeof(ExpandParams));
    if (!ep) return NULL;
    ep->new_ndim = xnd;
    ep->new_shape = cml_malloc((size_t)xnd * sizeof(int));
    if (!ep->new_shape) { cml_free(ep); return NULL; }
    memcpy(ep->new_shape, full_shape, (size_t)xnd * sizeof(int));
    struct IRNode* meanx = create_primitive_node(ir, UOP_EXPAND, &mean1->output, 1, ep, full_shape, xnd);
    if (!meanx) return NULL; chain_append(head, tail, meanx);

    // diff = x - meanx ; sq = diff * diff
    Tensor* din[] = {x, meanx->output};
    struct IRNode* diff = create_primitive_node(ir, UOP_SUB, din, 2, NULL, full_shape, xnd);
    if (!diff) return NULL; chain_append(head, tail, diff);
    Tensor* sqin[] = {diff->output, diff->output};
    struct IRNode* sq = create_primitive_node(ir, UOP_MUL, sqin, 2, NULL, full_shape, xnd);
    if (!sq) return NULL; chain_append(head, tail, sq);

    // var = sum(sq, dims, keepdim=node's) * (1/n)  -> node's output shape
    ReduceParams* sp2 = dup_reduce_params(rp, rp ? rp->keepdim : false);
    if (!sp2) return NULL;
    struct IRNode* sum2 = create_primitive_node(ir, UOP_SUM, &sq->output, 1, sp2, out_shape, out_ndim);
    if (!sum2) return NULL; chain_append(head, tail, sum2);
    struct IRNode* invn2 = insert_fill_node(ir, out_shape, out_ndim, inv_n);
    if (!invn2) return NULL; chain_append(head, tail, invn2);
    Tensor* vin[] = {sum2->output, invn2->output};
    struct IRNode* var = create_primitive_node(ir, UOP_MUL, vin, 2, NULL, out_shape, out_ndim);
    if (!var) return NULL; chain_append(head, tail, var);
    return var;
}

// VAR: mean((x - mean(x))^2)  [biased, matches the executor]
static int decompose_var(CMLGraph_t ir, struct IRNode* node) {
    struct IRNode *head = NULL, *tail = NULL;
    struct IRNode* var = build_variance(ir, node, &head, &tail);
    if (!var) return -1;
    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// STD: sqrt(var)
static int decompose_std(CMLGraph_t ir, struct IRNode* node) {
    struct IRNode *head = NULL, *tail = NULL;
    struct IRNode* var = build_variance(ir, node, &head, &tail);
    if (!var) return -1;
    int* os = node->output ? node->output->shape : node->output_shape;
    int on = node->output ? node->output->ndim : node->output_ndim;
    struct IRNode* std = create_primitive_node(ir, UOP_SQRT, &var->output, 1, NULL, os, on);
    if (!std) return -1;
    chain_append(&head, &tail, std);
    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// ELU: where(0<x, x, alpha*(exp(x)-1)), optionally scaled (SELU).
static int decompose_elu_impl(CMLGraph_t ir, struct IRNode* node, float alpha, float scale) {
    Tensor* x = node->inputs[0];
    int* s = x->shape; int nd = x->ndim;
    struct IRNode *h = NULL, *t = NULL;

    struct IRNode* zero = insert_fill_node(ir, s, nd, 0.0f); if (!zero) return -1; chain_append(&h, &t, zero);
    Tensor* ci[] = {zero->output, x};
    struct IRNode* cond = create_primitive_node(ir, UOP_CMPLT, ci, 2, NULL, s, nd); if (!cond) return -1; chain_append(&h, &t, cond);
    struct IRNode* ex = create_primitive_node(ir, UOP_EXP, &x, 1, NULL, s, nd); if (!ex) return -1; chain_append(&h, &t, ex);
    struct IRNode* one = insert_fill_node(ir, s, nd, 1.0f); if (!one) return -1; chain_append(&h, &t, one);
    Tensor* si[] = {ex->output, one->output};
    struct IRNode* em1 = create_primitive_node(ir, UOP_SUB, si, 2, NULL, s, nd); if (!em1) return -1; chain_append(&h, &t, em1);
    struct IRNode* af = insert_fill_node(ir, s, nd, alpha); if (!af) return -1; chain_append(&h, &t, af);
    Tensor* mi[] = {em1->output, af->output};
    struct IRNode* neg = create_primitive_node(ir, UOP_MUL, mi, 2, NULL, s, nd); if (!neg) return -1; chain_append(&h, &t, neg);
    Tensor* wi[] = {cond->output, x, neg->output};
    struct IRNode* res = create_primitive_node(ir, UOP_WHERE, wi, 3, NULL, s, nd); if (!res) return -1; chain_append(&h, &t, res);

    if (scale != 1.0f) {
        struct IRNode* sc = insert_fill_node(ir, s, nd, scale); if (!sc) return -1; chain_append(&h, &t, sc);
        Tensor* smi[] = {res->output, sc->output};
        struct IRNode* sres = create_primitive_node(ir, UOP_MUL, smi, 2, NULL, s, nd); if (!sres) return -1; chain_append(&h, &t, sres);
        res = sres;
    }
    (void)res;
    replace_node_with_chain(ir, node, h, t);
    return 0;
}

static int decompose_elu(CMLGraph_t ir, struct IRNode* node) {
    ClampParams* cp = (ClampParams*)node->params;
    float alpha = cp ? cp->min_val : 1.0f;
    return decompose_elu_impl(ir, node, alpha, 1.0f);
}

// SELU: scale * elu(x, alpha) with the standard SELU constants.
static int decompose_selu(CMLGraph_t ir, struct IRNode* node) {
    return decompose_elu_impl(ir, node, 1.6732632423543772f, 1.0507009873554805f);
}

// CELU: where(0<x, x, alpha*(exp(x/alpha)-1)).
static int decompose_celu(CMLGraph_t ir, struct IRNode* node) {
    Tensor* x = node->inputs[0];
    int* s = x->shape; int nd = x->ndim;
    ClampParams* cp = (ClampParams*)node->params;
    float alpha = cp ? cp->min_val : 1.0f;
    if (alpha == 0.0f) return 0;  // undefined; leave for executor
    struct IRNode *h = NULL, *t = NULL;

    struct IRNode* zero = insert_fill_node(ir, s, nd, 0.0f); if (!zero) return -1; chain_append(&h, &t, zero);
    Tensor* ci[] = {zero->output, x};
    struct IRNode* cond = create_primitive_node(ir, UOP_CMPLT, ci, 2, NULL, s, nd); if (!cond) return -1; chain_append(&h, &t, cond);
    struct IRNode* inva = insert_fill_node(ir, s, nd, 1.0f / alpha); if (!inva) return -1; chain_append(&h, &t, inva);
    Tensor* xi[] = {x, inva->output};
    struct IRNode* xa = create_primitive_node(ir, UOP_MUL, xi, 2, NULL, s, nd); if (!xa) return -1; chain_append(&h, &t, xa);
    struct IRNode* ex = create_primitive_node(ir, UOP_EXP, &xa->output, 1, NULL, s, nd); if (!ex) return -1; chain_append(&h, &t, ex);
    struct IRNode* one = insert_fill_node(ir, s, nd, 1.0f); if (!one) return -1; chain_append(&h, &t, one);
    Tensor* si[] = {ex->output, one->output};
    struct IRNode* em1 = create_primitive_node(ir, UOP_SUB, si, 2, NULL, s, nd); if (!em1) return -1; chain_append(&h, &t, em1);
    struct IRNode* af = insert_fill_node(ir, s, nd, alpha); if (!af) return -1; chain_append(&h, &t, af);
    Tensor* mi[] = {em1->output, af->output};
    struct IRNode* neg = create_primitive_node(ir, UOP_MUL, mi, 2, NULL, s, nd); if (!neg) return -1; chain_append(&h, &t, neg);
    Tensor* wi[] = {cond->output, x, neg->output};
    struct IRNode* res = create_primitive_node(ir, UOP_WHERE, wi, 3, NULL, s, nd); if (!res) return -1; chain_append(&h, &t, res);
    (void)res;
    replace_node_with_chain(ir, node, h, t);
    return 0;
}

// SOFTSIGN: x / (1 + |x|);  |x| = max(x, -x).
static int decompose_softsign(CMLGraph_t ir, struct IRNode* node) {
    Tensor* x = node->inputs[0];
    int* s = x->shape; int nd = x->ndim;
    struct IRNode *h = NULL, *t = NULL;

    struct IRNode* negx = create_primitive_node(ir, UOP_NEG, &x, 1, NULL, s, nd); if (!negx) return -1; chain_append(&h, &t, negx);
    Tensor* mi[] = {x, negx->output};
    struct IRNode* absx = create_primitive_node(ir, UOP_MAX, mi, 2, NULL, s, nd); if (!absx) return -1; chain_append(&h, &t, absx);
    struct IRNode* one = insert_fill_node(ir, s, nd, 1.0f); if (!one) return -1; chain_append(&h, &t, one);
    Tensor* ai[] = {one->output, absx->output};
    struct IRNode* denom = create_primitive_node(ir, UOP_ADD, ai, 2, NULL, s, nd); if (!denom) return -1; chain_append(&h, &t, denom);
    Tensor* di[] = {x, denom->output};
    struct IRNode* res = create_primitive_node(ir, UOP_DIV, di, 2, NULL, s, nd); if (!res) return -1; chain_append(&h, &t, res);
    (void)res;
    replace_node_with_chain(ir, node, h, t);
    return 0;
}

// NEG: x * (-1).
static int decompose_neg(CMLGraph_t ir, struct IRNode* node) {
    Tensor* x = node->inputs[0];
    int* s = x->shape; int nd = x->ndim;
    struct IRNode *h = NULL, *t = NULL;
    struct IRNode* m1 = insert_fill_node(ir, s, nd, -1.0f); if (!m1) return -1; chain_append(&h, &t, m1);
    Tensor* mi[] = {x, m1->output};
    struct IRNode* r = create_primitive_node(ir, UOP_MUL, mi, 2, NULL, s, nd); if (!r) return -1; chain_append(&h, &t, r);
    replace_node_with_chain(ir, node, h, t);
    return 0;
}

// SUB: a + (b * -1). Same broadcast semantics as the SUB kernel (ADD/MUL share it).
static int decompose_sub(CMLGraph_t ir, struct IRNode* node) {
    if (node->num_inputs < 2) return 0;
    Tensor* a = node->inputs[0];
    Tensor* b = node->inputs[1];
    int* os = node->output ? node->output->shape : node->output_shape;
    int on = node->output ? node->output->ndim : node->output_ndim;
    if (!os) return -1;
    struct IRNode *h = NULL, *t = NULL;
    struct IRNode* m1 = insert_fill_node(ir, b->shape, b->ndim, -1.0f); if (!m1) return -1; chain_append(&h, &t, m1);
    Tensor* ni[] = {b, m1->output};
    struct IRNode* nb = create_primitive_node(ir, UOP_MUL, ni, 2, NULL, b->shape, b->ndim); if (!nb) return -1; chain_append(&h, &t, nb);
    Tensor* ai[] = {a, nb->output};
    struct IRNode* r = create_primitive_node(ir, UOP_ADD, ai, 2, NULL, os, on); if (!r) return -1; chain_append(&h, &t, r);
    replace_node_with_chain(ir, node, h, t);
    return 0;
}

// DIV: a * recip(b).
static int decompose_div(CMLGraph_t ir, struct IRNode* node) {
    if (node->num_inputs < 2) return 0;
    Tensor* a = node->inputs[0];
    Tensor* b = node->inputs[1];
    int* os = node->output ? node->output->shape : node->output_shape;
    int on = node->output ? node->output->ndim : node->output_ndim;
    if (!os) return -1;
    struct IRNode *h = NULL, *t = NULL;
    struct IRNode* rb = create_primitive_node(ir, UOP_RECIP, &b, 1, NULL, b->shape, b->ndim); if (!rb) return -1; chain_append(&h, &t, rb);
    Tensor* mi[] = {a, rb->output};
    struct IRNode* r = create_primitive_node(ir, UOP_MUL, mi, 2, NULL, os, on); if (!r) return -1; chain_append(&h, &t, r);
    replace_node_with_chain(ir, node, h, t);
    return 0;
}

// FLATTEN: a reshape to the node's (already-computed) output shape.
static int decompose_flatten(CMLGraph_t ir, struct IRNode* node) {
    Tensor* x = node->inputs[0];
    int* os = node->output ? node->output->shape : node->output_shape;
    int on = node->output ? node->output->ndim : node->output_ndim;
    if (!os || on <= 0) return -1;
    ReshapeParams* rp = cml_malloc(sizeof(ReshapeParams));
    if (!rp) return -1;
    rp->new_ndim = on;
    rp->new_shape = cml_malloc((size_t)on * sizeof(int));
    if (!rp->new_shape) { cml_free(rp); return -1; }
    memcpy(rp->new_shape, os, (size_t)on * sizeof(int));
    struct IRNode *h = NULL, *t = NULL;
    struct IRNode* r = create_primitive_node(ir, UOP_RESHAPE, &x, 1, rp, os, on);
    if (!r) { cml_free(rp->new_shape); cml_free(rp); return -1; }
    chain_append(&h, &t, r);
    replace_node_with_chain(ir, node, h, t);
    return 0;
}

// ── Movement helpers for structural (pool / conv) decompositions ──────────

// UNFOLD the last axis into [..., num_windows, ks]. Appends to the chain and
// returns the new node (its output tensor carries the unfolded shape).
static struct IRNode* insert_unfold_last(CMLGraph_t ir, Tensor* in, int in_nd,
                                         int ks, int stride,
                                         struct IRNode** head, struct IRNode** tail) {
    int L = in->shape[in_nd - 1];
    int nw = (L - ks) / stride + 1;
    int out_shape[16];
    for (int i = 0; i < in_nd - 1; i++) out_shape[i] = in->shape[i];
    out_shape[in_nd - 1] = nw;
    out_shape[in_nd]     = ks;
    UnfoldParams* up = cml_malloc(sizeof(UnfoldParams));
    if (!up) return NULL;
    up->kernel_size = ks;
    up->stride      = stride;
    struct IRNode* n = create_primitive_node(ir, UOP_UNFOLD, &in, 1, up, out_shape, in_nd + 1);
    if (!n) { cml_free(up); return NULL; }
    chain_append(head, tail, n);
    return n;
}

// PERMUTE `in` (ndim nd) by `perm`; out_shape[i] = in->shape[perm[i]].
static struct IRNode* insert_permute(CMLGraph_t ir, Tensor* in, int nd, const int* perm,
                                     struct IRNode** head, struct IRNode** tail) {
    int out_shape[16];
    for (int i = 0; i < nd; i++) out_shape[i] = in->shape[perm[i]];
    PermuteParams* pp = cml_malloc(sizeof(PermuteParams));
    if (!pp) return NULL;
    pp->num_dims = nd;
    pp->perm = cml_malloc((size_t)nd * sizeof(int));
    if (!pp->perm) { cml_free(pp); return NULL; }
    memcpy(pp->perm, perm, (size_t)nd * sizeof(int));
    struct IRNode* n = create_primitive_node(ir, UOP_PERMUTE, &in, 1, pp, out_shape, nd);
    if (!n) { cml_free(pp->perm); cml_free(pp); return NULL; }
    chain_append(head, tail, n);
    return n;
}

// Reduce (SUM or MAX_REDUCE) the LAST axis of `in` (ndim nd), keepdim=false.
static struct IRNode* insert_reduce_last(CMLGraph_t ir, Tensor* in, int nd, UOpType rtype,
                                         struct IRNode** head, struct IRNode** tail) {
    int out_shape[16];
    for (int i = 0; i < nd - 1; i++) out_shape[i] = in->shape[i];
    ReduceParams* rp = cml_malloc(sizeof(ReduceParams));
    if (!rp) return NULL;
    rp->num_dims = 1;
    rp->keepdim  = false;
    rp->dims = cml_malloc(sizeof(int));
    if (!rp->dims) { cml_free(rp); return NULL; }
    rp->dims[0] = nd - 1;
    struct IRNode* n = create_primitive_node(ir, rtype, &in, 1, rp, out_shape, nd - 1);
    if (!n) { cml_free(rp->dims); cml_free(rp); return NULL; }
    chain_append(head, tail, n);
    return n;
}

// POOL2D → pad? → unfold(W) → permute → unfold(H) → reduce(kh) → permute →
// reduce(kw) → permute → (avg: * 1/(kh*kw)). Uses only movement + reduce
// primitives, so backward is automatic. Returns -1 to leave the node for the
// executor fallback (exotic dilation / ceil_mode / count_exclude_pad cases).
static int decompose_pool2d(CMLGraph_t ir, struct IRNode* node, bool is_max) {
    Tensor* x = node->inputs[0];
    if (!x || x->ndim != 4) return 0;
    Pool2DParams* p = (Pool2DParams*)node->params;
    if (!p) return 0;
    int kh = p->kernel_size[0], kw = p->kernel_size[1];
    int sh = p->stride[0] > 0 ? p->stride[0] : kh;
    int sw = p->stride[1] > 0 ? p->stride[1] : kw;
    int ph = p->padding[0], pw = p->padding[1];
    int dh = p->dilation[0] > 0 ? p->dilation[0] : 1;
    int dw = p->dilation[1] > 0 ? p->dilation[1] : 1;
    // Only the clean case decomposes; anything exotic leaves the node intact
    // (return 0, no warning) so the executor's pool kernel handles it.
    if (dh != 1 || dw != 1 || p->ceil_mode) return 0;
    if (!is_max && (ph > 0 || pw > 0) && !p->count_include_pad) return 0;

    int N = x->shape[0], C = x->shape[1];
    struct IRNode *head = NULL, *tail = NULL;
    Tensor* cur = x;
    int Hp = x->shape[2], Wp = x->shape[3];

    // Optional pad on H, W (zeros for avg / count_include_pad; -inf for max).
    if (ph > 0 || pw > 0) {
        PadParams* pd = cml_malloc(sizeof(PadParams));
        if (!pd) return -1;
        pd->num_dims = 4;
        pd->mode = PAD_CONSTANT;
        pd->value = is_max ? -INFINITY : 0.0f;
        pd->pad_widths = cml_malloc(8 * sizeof(int));
        if (!pd->pad_widths) { cml_free(pd); return -1; }
        int pw_arr[8] = {0,0, 0,0, ph,ph, pw,pw};
        memcpy(pd->pad_widths, pw_arr, 8 * sizeof(int));
        Hp = x->shape[2] + 2 * ph; Wp = x->shape[3] + 2 * pw;
        int ps[4] = {N, C, Hp, Wp};
        struct IRNode* padn = create_primitive_node(ir, UOP_PAD, &cur, 1, pd, ps, 4);
        if (!padn) { cml_free(pd->pad_widths); cml_free(pd); return -1; }
        chain_append(&head, &tail, padn);
        cur = padn->output;
    }

    int OH = (Hp - kh) / sh + 1;
    int OW = (Wp - kw) / sw + 1;
    UOpType rtype = is_max ? UOP_MAX_REDUCE : UOP_SUM;

    // unfold W: [N,C,Hp,Wp] -> [N,C,Hp,OW,kw]
    struct IRNode* uW = insert_unfold_last(ir, cur, 4, kw, sw, &head, &tail);
    if (!uW) return -1;
    // permute H to last: [N,C,Hp,OW,kw] -> [N,C,OW,kw,Hp]
    int pA[5] = {0,1,3,4,2};
    struct IRNode* pmA = insert_permute(ir, uW->output, 5, pA, &head, &tail);
    if (!pmA) return -1;
    // unfold H: [N,C,OW,kw,Hp] -> [N,C,OW,kw,OH,kh]
    struct IRNode* uH = insert_unfold_last(ir, pmA->output, 5, kh, sh, &head, &tail);
    if (!uH) return -1;
    // reduce kh (last): -> [N,C,OW,kw,OH]
    struct IRNode* r1 = insert_reduce_last(ir, uH->output, 6, rtype, &head, &tail);
    if (!r1) return -1;
    // permute kw to last: [N,C,OW,kw,OH] -> [N,C,OW,OH,kw]
    int pB[5] = {0,1,2,4,3};
    struct IRNode* pmB = insert_permute(ir, r1->output, 5, pB, &head, &tail);
    if (!pmB) return -1;
    // reduce kw (last): -> [N,C,OW,OH]
    struct IRNode* r2 = insert_reduce_last(ir, pmB->output, 5, rtype, &head, &tail);
    if (!r2) return -1;
    // permute to [N,C,OH,OW]
    int pC[4] = {0,1,3,2};
    struct IRNode* pmC = insert_permute(ir, r2->output, 4, pC, &head, &tail);
    if (!pmC) return -1;
    struct IRNode* result = pmC;

    // avg: multiply by 1/(kh*kw)
    if (!is_max) {
        int os[4] = {N, C, OH, OW};
        float inv = (kh * kw > 0) ? 1.0f / (float)(kh * kw) : 0.0f;
        struct IRNode* filln = insert_fill_node(ir, os, 4, inv);
        if (!filln) return -1;
        chain_append(&head, &tail, filln);
        Tensor* mi[] = {result->output, filln->output};
        struct IRNode* mul = create_primitive_node(ir, UOP_MUL, mi, 2, NULL, os, 4);
        if (!mul) return -1;
        chain_append(&head, &tail, mul);
        result = mul;
    }

    (void)result;
    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

static int decompose_maxpool2d(CMLGraph_t ir, struct IRNode* node) {
    return decompose_pool2d(ir, node, true);
}
static int decompose_avgpool2d(CMLGraph_t ir, struct IRNode* node) {
    return decompose_pool2d(ir, node, false);
}

// RESHAPE `in` to new_shape (must be same numel; contiguous view).
static struct IRNode* insert_reshape(CMLGraph_t ir, Tensor* in, const int* new_shape, int new_nd,
                                     struct IRNode** head, struct IRNode** tail) {
    ReshapeParams* rp = cml_malloc(sizeof(ReshapeParams));
    if (!rp) return NULL;
    rp->new_ndim = new_nd;
    rp->new_shape = cml_malloc((size_t)new_nd * sizeof(int));
    if (!rp->new_shape) { cml_free(rp); return NULL; }
    memcpy(rp->new_shape, new_shape, (size_t)new_nd * sizeof(int));
    struct IRNode* n = create_primitive_node(ir, UOP_RESHAPE, &in, 1, rp, (int*)new_shape, new_nd);
    if (!n) { cml_free(rp->new_shape); cml_free(rp); return NULL; }
    chain_append(head, tail, n);
    return n;
}

// EXPAND (broadcast) `in` to new_shape.
static struct IRNode* insert_expand(CMLGraph_t ir, Tensor* in, const int* new_shape, int new_nd,
                                    struct IRNode** head, struct IRNode** tail) {
    ExpandParams* ep = cml_malloc(sizeof(ExpandParams));
    if (!ep) return NULL;
    ep->new_ndim = new_nd;
    ep->new_shape = cml_malloc((size_t)new_nd * sizeof(int));
    if (!ep->new_shape) { cml_free(ep); return NULL; }
    memcpy(ep->new_shape, new_shape, (size_t)new_nd * sizeof(int));
    struct IRNode* n = create_primitive_node(ir, UOP_EXPAND, &in, 1, ep, (int*)new_shape, new_nd);
    if (!n) { cml_free(ep->new_shape); cml_free(ep); return NULL; }
    chain_append(head, tail, n);
    return n;
}

// CONV2D → im2col (pad? → unfold(W) → permute → unfold(H) → permute → reshape)
// → matmul with reshaped/transposed weight → reshape → permute → (+bias).
// MATMUL is kept as the one hardware-GEMM primitive. Clean case only
// (dilation=1, groups=1); otherwise leaves the node for the executor.
static int decompose_conv2d(CMLGraph_t ir, struct IRNode* node) {
    Tensor* x = node->inputs[0];
    Tensor* w = node->inputs[1];
    Tensor* bias = (node->num_inputs >= 3) ? node->inputs[2] : NULL;
    if (!x || x->ndim != 4 || !w || w->ndim != 4) return 0;
    Conv2DParams* p = (Conv2DParams*)node->params;
    if (!p) return 0;
    int Cout = w->shape[0], Cin = w->shape[1], kh = w->shape[2], kw = w->shape[3];
    int sh = p->stride   ? p->stride[0]   : 1, sw = p->stride   ? p->stride[1]   : 1;
    int ph = p->padding  ? p->padding[0]  : 0, pw = p->padding  ? p->padding[1]  : 0;
    int dh = p->dilation ? p->dilation[0] : 1, dw = p->dilation ? p->dilation[1] : 1;
    int groups = p->groups > 0 ? p->groups : 1;
    if (dh != 1 || dw != 1 || groups != 1) return 0;   // executor fallback
    if (x->shape[1] != Cin) return 0;

    int N = x->shape[0];
    int Hp = x->shape[2], Wp = x->shape[3];
    struct IRNode *head = NULL, *tail = NULL;
    Tensor* cur = x;

    if (ph > 0 || pw > 0) {
        PadParams* pd = cml_malloc(sizeof(PadParams));
        if (!pd) return -1;
        pd->num_dims = 4; pd->mode = PAD_CONSTANT; pd->value = 0.0f;
        pd->pad_widths = cml_malloc(8 * sizeof(int));
        if (!pd->pad_widths) { cml_free(pd); return -1; }
        int pw_arr[8] = {0,0, 0,0, ph,ph, pw,pw};
        memcpy(pd->pad_widths, pw_arr, 8 * sizeof(int));
        Hp = x->shape[2] + 2 * ph; Wp = x->shape[3] + 2 * pw;
        int ps[4] = {N, Cin, Hp, Wp};
        struct IRNode* padn = create_primitive_node(ir, UOP_PAD, &cur, 1, pd, ps, 4);
        if (!padn) { cml_free(pd->pad_widths); cml_free(pd); return -1; }
        chain_append(&head, &tail, padn);
        cur = padn->output;
    }

    int OH = (Hp - kh) / sh + 1, OW = (Wp - kw) / sw + 1;
    int K = Cin * kh * kw, M = OH * OW;

    // im2col: [N,Cin,Hp,Wp] -> ... -> [N*M, K] with K ordered (Cin,kh,kw)
    struct IRNode* uW = insert_unfold_last(ir, cur, 4, kw, sw, &head, &tail);      // [N,Cin,Hp,OW,kw]
    if (!uW) return -1;
    int pA[5] = {0,1,3,4,2};
    struct IRNode* pmA = insert_permute(ir, uW->output, 5, pA, &head, &tail);      // [N,Cin,OW,kw,Hp]
    if (!pmA) return -1;
    struct IRNode* uH = insert_unfold_last(ir, pmA->output, 5, kh, sh, &head, &tail); // [N,Cin,OW,kw,OH,kh]
    if (!uH) return -1;
    int pB[6] = {0,4,2,1,5,3};
    struct IRNode* pmB = insert_permute(ir, uH->output, 6, pB, &head, &tail);      // [N,OH,OW,Cin,kh,kw]
    if (!pmB) return -1;
    int im_shape[2] = {N * M, K};
    struct IRNode* im = insert_reshape(ir, pmB->output, im_shape, 2, &head, &tail); // [N*M, K]
    if (!im) return -1;

    // weight [Cout,Cin,kh,kw] -> [Cout,K] -> transpose [K,Cout]
    int wr_shape[2] = {Cout, K};
    struct IRNode* wr = insert_reshape(ir, w, wr_shape, 2, &head, &tail);
    if (!wr) return -1;
    int wtp[2] = {1, 0};
    struct IRNode* wt = insert_permute(ir, wr->output, 2, wtp, &head, &tail);      // [K,Cout]
    if (!wt) return -1;

    // matmul [N*M,K] @ [K,Cout] -> [N*M,Cout]
    Tensor* mm_in[] = {im->output, wt->output};
    int mm_shape[2] = {N * M, Cout};
    struct IRNode* mm = create_primitive_node(ir, UOP_MATMUL, mm_in, 2, NULL, mm_shape, 2);
    if (!mm) return -1;
    chain_append(&head, &tail, mm);

    // reshape [N,OH,OW,Cout] then permute -> [N,Cout,OH,OW]
    int r4_shape[4] = {N, OH, OW, Cout};
    struct IRNode* r4 = insert_reshape(ir, mm->output, r4_shape, 4, &head, &tail);
    if (!r4) return -1;
    int pC[4] = {0,3,1,2};
    struct IRNode* pmC = insert_permute(ir, r4->output, 4, pC, &head, &tail);      // [N,Cout,OH,OW]
    if (!pmC) return -1;
    struct IRNode* result = pmC;

    if (bias && bias->ndim == 1 && bias->shape[0] == Cout) {
        int br_shape[4] = {1, Cout, 1, 1};
        struct IRNode* br = insert_reshape(ir, bias, br_shape, 4, &head, &tail);
        if (!br) return -1;
        int be_shape[4] = {N, Cout, OH, OW};
        struct IRNode* be = insert_expand(ir, br->output, be_shape, 4, &head, &tail);
        if (!be) return -1;
        Tensor* ain[] = {result->output, be->output};
        struct IRNode* add = create_primitive_node(ir, UOP_ADD, ain, 2, NULL, be_shape, 4);
        if (!add) return -1;
        chain_append(&head, &tail, add);
        result = add;
    }

    (void)result;
    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// CONV3D → im2col via three unfolds (W,H,D) → matmul → reshape → permute →
// (+bias). Same construction as conv2d with an extra depth axis. Clean case
// only (dilation=1); otherwise leaves the node for the executor.
static int decompose_conv3d(CMLGraph_t ir, struct IRNode* node) {
    Tensor* x = node->inputs[0];
    Tensor* w = node->inputs[1];
    Tensor* bias = (node->num_inputs >= 3) ? node->inputs[2] : NULL;
    if (!x || x->ndim != 5 || !w || w->ndim != 5) return 0;
    Conv3DParams* p = (Conv3DParams*)node->params;
    if (!p) return 0;
    int Cout = w->shape[0], Cin = w->shape[1], kd = w->shape[2], kh = w->shape[3], kw = w->shape[4];
    int sd = p->stride[0] > 0 ? p->stride[0] : 1;
    int sh = p->stride[1] > 0 ? p->stride[1] : 1;
    int sw = p->stride[2] > 0 ? p->stride[2] : 1;
    int pd = p->padding[0], ph = p->padding[1], pw = p->padding[2];
    if (p->dilation[0] != 1 || p->dilation[1] != 1 || p->dilation[2] != 1) return 0;
    if (x->shape[1] != Cin) return 0;

    int N = x->shape[0];
    int Dp = x->shape[2], Hp = x->shape[3], Wp = x->shape[4];
    struct IRNode *head = NULL, *tail = NULL;
    Tensor* cur = x;

    if (pd > 0 || ph > 0 || pw > 0) {
        PadParams* pdp = cml_malloc(sizeof(PadParams));
        if (!pdp) return -1;
        pdp->num_dims = 5; pdp->mode = PAD_CONSTANT; pdp->value = 0.0f;
        pdp->pad_widths = cml_malloc(10 * sizeof(int));
        if (!pdp->pad_widths) { cml_free(pdp); return -1; }
        int pwv[10] = {0,0, 0,0, pd,pd, ph,ph, pw,pw};
        memcpy(pdp->pad_widths, pwv, 10 * sizeof(int));
        Dp = x->shape[2] + 2 * pd; Hp = x->shape[3] + 2 * ph; Wp = x->shape[4] + 2 * pw;
        int ps[5] = {N, Cin, Dp, Hp, Wp};
        struct IRNode* padn = create_primitive_node(ir, UOP_PAD, &cur, 1, pdp, ps, 5);
        if (!padn) { cml_free(pdp->pad_widths); cml_free(pdp); return -1; }
        chain_append(&head, &tail, padn);
        cur = padn->output;
    }

    int OD = (Dp - kd) / sd + 1, OH = (Hp - kh) / sh + 1, OW = (Wp - kw) / sw + 1;
    int K = Cin * kd * kh * kw, M = OD * OH * OW;

    struct IRNode* uW = insert_unfold_last(ir, cur, 5, kw, sw, &head, &tail);       // [N,Cin,Dp,Hp,OW,kw]
    if (!uW) return -1;
    int pA[6] = {0,1,2,4,5,3};
    struct IRNode* pmA = insert_permute(ir, uW->output, 6, pA, &head, &tail);       // [N,Cin,Dp,OW,kw,Hp]
    if (!pmA) return -1;
    struct IRNode* uH = insert_unfold_last(ir, pmA->output, 6, kh, sh, &head, &tail); // [N,Cin,Dp,OW,kw,OH,kh]
    if (!uH) return -1;
    int pB[7] = {0,1,3,4,5,6,2};
    struct IRNode* pmB = insert_permute(ir, uH->output, 7, pB, &head, &tail);       // [N,Cin,OW,kw,OH,kh,Dp]
    if (!pmB) return -1;
    struct IRNode* uD = insert_unfold_last(ir, pmB->output, 7, kd, sd, &head, &tail); // [N,Cin,OW,kw,OH,kh,OD,kd]
    if (!uD) return -1;
    int pC[8] = {0,6,4,2,1,7,5,3};
    struct IRNode* pmC = insert_permute(ir, uD->output, 8, pC, &head, &tail);       // [N,OD,OH,OW,Cin,kd,kh,kw]
    if (!pmC) return -1;
    int im_shape[2] = {N * M, K};
    struct IRNode* im = insert_reshape(ir, pmC->output, im_shape, 2, &head, &tail); // [N*M, K]
    if (!im) return -1;

    int wr_shape[2] = {Cout, K};
    struct IRNode* wr = insert_reshape(ir, w, wr_shape, 2, &head, &tail);
    if (!wr) return -1;
    int wtp[2] = {1, 0};
    struct IRNode* wt = insert_permute(ir, wr->output, 2, wtp, &head, &tail);       // [K,Cout]
    if (!wt) return -1;

    Tensor* mm_in[] = {im->output, wt->output};
    int mm_shape[2] = {N * M, Cout};
    struct IRNode* mm = create_primitive_node(ir, UOP_MATMUL, mm_in, 2, NULL, mm_shape, 2);
    if (!mm) return -1;
    chain_append(&head, &tail, mm);

    int r5_shape[5] = {N, OD, OH, OW, Cout};
    struct IRNode* r5 = insert_reshape(ir, mm->output, r5_shape, 5, &head, &tail);
    if (!r5) return -1;
    int pD[5] = {0,4,1,2,3};
    struct IRNode* pmD = insert_permute(ir, r5->output, 5, pD, &head, &tail);       // [N,Cout,OD,OH,OW]
    if (!pmD) return -1;
    struct IRNode* result = pmD;

    if (bias && bias->ndim == 1 && bias->shape[0] == Cout) {
        int br_shape[5] = {1, Cout, 1, 1, 1};
        struct IRNode* br = insert_reshape(ir, bias, br_shape, 5, &head, &tail);
        if (!br) return -1;
        int be_shape[5] = {N, Cout, OD, OH, OW};
        struct IRNode* be = insert_expand(ir, br->output, be_shape, 5, &head, &tail);
        if (!be) return -1;
        Tensor* ain[] = {result->output, be->output};
        struct IRNode* add = create_primitive_node(ir, UOP_ADD, ain, 2, NULL, be_shape, 5);
        if (!add) return -1;
        chain_append(&head, &tail, add);
        result = add;
    }

    (void)result;
    replace_node_with_chain(ir, node, head, tail);
    return 0;
}

// Main Decomposition Pass

int cml_ir_decompose(CMLGraph_t ir) {
    if (!ir) return -1;
    if (ir->is_decomposed) return 0;

    LOG_DEBUG("Running IR decomposition pass");

    /* Fixpoint: some rules EMIT other composite ops (e.g. VAR→…→SUB, softsign
     * →DIV, sigmoid→NEG). Inserted nodes aren't revisited within one scan, so
     * repeat until a full scan creates no new nodes (g_decompose_counter is
     * bumped once per node created). Bounded to guard against error loops. */
    const int MAX_PASSES = 24;
    for (int pass = 0; pass < MAX_PASSES; pass++) {
    int counter_before = atomic_load(&g_decompose_counter);
    struct IRNode* node = ir->head;

    while (node) {
        struct IRNode* next = node->next;
        int result = 0;
        bool decomposed = false;

        switch (node->type) {
        case UOP_NEG:      result = decompose_neg(ir, node);      decomposed = true; break;
        case UOP_SUB:      result = decompose_sub(ir, node);      decomposed = true; break;
        case UOP_DIV:      result = decompose_div(ir, node);      decomposed = true; break;
        case UOP_FLATTEN:  result = decompose_flatten(ir, node);  decomposed = true; break;
        case UOP_RELU:     result = decompose_relu(ir, node);     decomposed = true; break;
        case UOP_RELU6:    result = decompose_relu6(ir, node);    decomposed = true; break;
        case UOP_SILU:     result = decompose_silu(ir, node);     decomposed = true; break;
        case UOP_HARDSWISH:result = decompose_hardswish(ir, node);decomposed = true; break;
        case UOP_MISH:     result = decompose_mish(ir, node);     decomposed = true; break;
        case UOP_SIGMOID:  result = decompose_sigmoid(ir, node);  decomposed = true; break;
        case UOP_TANH:     result = decompose_tanh(ir, node);     decomposed = true; break;
        case UOP_SQUARE:   result = decompose_square(ir, node);   decomposed = true; break;
        case UOP_RSQRT:    result = decompose_rsqrt(ir, node);    decomposed = true; break;
        case UOP_ABS:      result = decompose_abs(ir, node);      decomposed = true; break;

        case UOP_CMPEQ:    result = decompose_cmpeq(ir, node);    decomposed = true; break;
        case UOP_CMPNE:    result = decompose_cmpne(ir, node);    decomposed = true; break;
        case UOP_CMPLE:    result = decompose_cmple(ir, node);    decomposed = true; break;
        case UOP_CMPGT:    result = decompose_cmpgt(ir, node);    decomposed = true; break;
        case UOP_CMPGE:    result = decompose_cmpge(ir, node);    decomposed = true; break;
        case UOP_MINIMUM:  result = decompose_minimum(ir, node);  decomposed = true; break;

        case UOP_COS:      result = decompose_cos(ir, node);      decomposed = true; break;
        case UOP_TAN:      result = decompose_tan(ir, node);      decomposed = true; break;
        case UOP_LOG2:     result = decompose_log2(ir, node);     decomposed = true; break;
        case UOP_EXP2:     result = decompose_exp2(ir, node);     decomposed = true; break;
        case UOP_LOG10:    result = decompose_log10(ir, node);    decomposed = true; break;
        case UOP_LOGADDEXP:result = decompose_logaddexp(ir, node);decomposed = true; break;
        case UOP_MOD:      result = decompose_mod(ir, node);      decomposed = true; break;
        case UOP_IDIV:     result = decompose_idiv(ir, node);     decomposed = true; break;
        case UOP_COPYSIGN: result = decompose_copysign(ir, node); decomposed = true; break;
        case UOP_SIGN:     result = decompose_sign(ir, node);     decomposed = true; break;
        case UOP_SINH:     result = decompose_sinh(ir, node);     decomposed = true; break;
        case UOP_COSH:     result = decompose_cosh(ir, node);     decomposed = true; break;
        case UOP_ATANH:    result = decompose_atanh(ir, node);    decomposed = true; break;

        case UOP_MAXPOOL2D:  result = decompose_maxpool2d(ir, node);  decomposed = true; break;
        case UOP_AVGPOOL2D:  result = decompose_avgpool2d(ir, node);  decomposed = true; break;
        case UOP_CONV2D:     result = decompose_conv2d(ir, node);     decomposed = true; break;
        case UOP_CONV3D:     result = decompose_conv3d(ir, node);     decomposed = true; break;

        case UOP_MEAN:       result = decompose_mean(ir, node);       decomposed = true; break;
        case UOP_VAR:        result = decompose_var(ir, node);        decomposed = true; break;
        case UOP_STD:        result = decompose_std(ir, node);        decomposed = true; break;
        case UOP_MIN_REDUCE: result = decompose_min_reduce(ir, node); decomposed = true; break;

        case UOP_LOGICAL_NOT: result = decompose_logical_not(ir, node); decomposed = true; break;
        case UOP_LOGICAL_AND: result = decompose_logical_and(ir, node); decomposed = true; break;
        case UOP_LOGICAL_OR:  result = decompose_logical_or(ir, node);  decomposed = true; break;
        case UOP_CLAMP:       result = decompose_clamp(ir, node);       decomposed = true; break;
        case UOP_ELU:         result = decompose_elu(ir, node);         decomposed = true; break;
        case UOP_SELU:        result = decompose_selu(ir, node);        decomposed = true; break;
        case UOP_CELU:        result = decompose_celu(ir, node);        decomposed = true; break;
        case UOP_SOFTSIGN:    result = decompose_softsign(ir, node);    decomposed = true; break;
        case UOP_SOFTPLUS:    result = decompose_softplus(ir, node);    decomposed = true; break;
        case UOP_LOGSIGMOID:  result = decompose_logsigmoid(ir, node);  decomposed = true; break;
        case UOP_ERFC:        result = decompose_erfc(ir, node);        decomposed = true; break;

        default: break;
        }

        if (result != 0 && decomposed)
            LOG_WARNING("Failed to decompose op type %d", node->type);

        node = next;
    }

    if (atomic_load(&g_decompose_counter) == counter_before)
        break;  /* fixpoint reached — no new nodes created this scan */
    }

    ir->is_decomposed = true;
    LOG_DEBUG("IR decomposition pass complete");
    return 0;
}
