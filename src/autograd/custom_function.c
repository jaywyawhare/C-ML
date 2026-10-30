#include "autograd/custom_function.h"
#include "autograd/autograd.h"
#include "ops/ir/autodiff.h"
#include "ops/ir/execution.h"
#include "ops/ir/internal.h"
#include "tensor/realize.h"
#include "core/logging.h"
#include "alloc/memory_management.h"
#include "alloc/cml_allocator.h"
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

struct CMLAutogradContext {
    Tensor** saved;
    int n_saved;
    void* data;
};

typedef struct CustomNode {
    Tensor* output;
    Tensor** inputs;
    int n_inputs;
    CMLCustomForward fwd;
    CMLCustomBackward bwd;
    void* user_data;
    CMLAutogradContext* ctx;
    bool consumed;
    struct CustomNode* next;
} CustomNode;

static CustomNode* g_nodes_head      = NULL;
static CustomNode* g_nodes_tail      = NULL;
static pthread_mutex_t g_nodes_lock  = PTHREAD_MUTEX_INITIALIZER;
static bool g_warned_double_backward = false;

/**
 * Free a context's saved-tensor array (not the tensors) and the context itself.
 * @param ctx (CMLAutogradContext*)
 * @return (void)
 */
static void ctx_free(CMLAutogradContext* ctx) {
    if (!ctx)
        return;
    if (ctx->saved)
        cml_free(ctx->saved);
    cml_free(ctx);
}

void cml_autograd_ctx_save(CMLAutogradContext* ctx, Tensor** tensors, int n) {
    if (!ctx)
        return;
    if (ctx->saved) {
        cml_free(ctx->saved);
        ctx->saved = NULL;
    }
    ctx->n_saved = 0;
    if (!tensors || n <= 0)
        return;
    ctx->saved = cml_malloc((size_t)n * sizeof(Tensor*));
    if (!ctx->saved) {
        LOG_ERROR("cml_autograd_ctx_save: allocation failed");
        return;
    }
    memcpy(ctx->saved, tensors, (size_t)n * sizeof(Tensor*));
    ctx->n_saved = n;
}

Tensor** cml_autograd_ctx_saved(CMLAutogradContext* ctx, int* out_n) {
    if (out_n)
        *out_n = ctx ? ctx->n_saved : 0;
    return ctx ? ctx->saved : NULL;
}

void cml_autograd_ctx_set_data(CMLAutogradContext* ctx, void* p) {
    if (ctx)
        ctx->data = p;
}

void* cml_autograd_ctx_get_data(CMLAutogradContext* ctx) { return ctx ? ctx->data : NULL; }

/**
 * Materialize a lazy tensor so a callback can read its data.
 * @param t (Tensor*)
 * @return (void)
 */
static void realize_if_lazy(Tensor* t) {
    if (t && t->ir_node && !t->is_executed)
        tensor_realize(t);
}

Tensor* cml_autograd_function_apply(CMLCustomForward fwd, CMLCustomBackward bwd, Tensor** inputs,
                                    int n_inputs, void* user_data) {
    if (!fwd || n_inputs < 0 || (n_inputs > 0 && !inputs)) {
        LOG_ERROR("cml_autograd_function_apply: invalid arguments");
        return NULL;
    }

    for (int i = 0; i < n_inputs; i++)
        realize_if_lazy(inputs[i]);

    CMLAutogradContext* ctx = cml_calloc(1, sizeof(CMLAutogradContext));
    if (!ctx) {
        LOG_ERROR("cml_autograd_function_apply: context allocation failed");
        return NULL;
    }

    Tensor* out = fwd(ctx, inputs, n_inputs, user_data);
    if (!out) {
        ctx_free(ctx);
        return NULL;
    }

    bool any_requires = false;
    for (int i = 0; i < n_inputs; i++)
        if (inputs[i] && inputs[i]->requires_grad)
            any_requires = true;

    if (!any_requires || !autograd_is_grad_enabled() || !bwd) {
        ctx_free(ctx);
        return out;
    }

    /* Detach the output into a concrete leaf so the engine's reverse pass treats
     * it as a differentiable leaf and deposits d(loss)/d(output) into out->grad,
     * which the sweep then consumes. */
    realize_if_lazy(out);
    out->ir_node       = NULL;
    out->ir_context    = NULL;
    out->is_executed   = true;
    out->requires_grad = true;

    CustomNode* node = cml_calloc(1, sizeof(CustomNode));
    Tensor** in_copy = n_inputs > 0 ? cml_malloc((size_t)n_inputs * sizeof(Tensor*)) : NULL;
    if (!node || (n_inputs > 0 && !in_copy)) {
        LOG_ERROR("cml_autograd_function_apply: node allocation failed");
        if (node)
            cml_free(node);
        if (in_copy)
            cml_free(in_copy);
        ctx_free(ctx);
        return out;
    }
    if (n_inputs > 0)
        memcpy(in_copy, inputs, (size_t)n_inputs * sizeof(Tensor*));

    node->output    = out;
    node->inputs    = in_copy;
    node->n_inputs  = n_inputs;
    node->fwd       = fwd;
    node->bwd       = bwd;
    node->user_data = user_data;
    node->ctx       = ctx;
    node->consumed  = false;
    node->next      = NULL;

    pthread_mutex_lock(&g_nodes_lock);
    if (g_nodes_tail)
        g_nodes_tail->next = node;
    else
        g_nodes_head = node;
    g_nodes_tail = node;
    pthread_mutex_unlock(&g_nodes_lock);

    return out;
}

/**
 * Run one node's backward and accumulate the resulting input gradients.
 * @param node (CustomNode*)
 * @return (void)
 */
static void run_node_backward(CustomNode* node) {
    Tensor* go = node->output->grad;
    if (!go)
        return;
    realize_if_lazy(go);

    Tensor** grad_inputs =
        node->n_inputs > 0 ? cml_calloc((size_t)node->n_inputs, sizeof(Tensor*)) : NULL;
    if (node->n_inputs > 0 && !grad_inputs) {
        LOG_ERROR("custom backward: grad_inputs allocation failed");
        return;
    }

    node->bwd(node->ctx, go, grad_inputs, node->n_inputs, node->user_data);

    for (int j = 0; j < node->n_inputs; j++) {
        Tensor* in = node->inputs[j];
        Tensor* g  = grad_inputs ? grad_inputs[j] : NULL;
        if (!in || !g || !in->requires_grad)
            continue;
        realize_if_lazy(g);
        tensor_accumulate_grad(in, g);

        /* Built-in producer upstream of the custom op: push the input gradient
         * on through its own subgraph. First-order only. */
        if (in->ir_node && in->ir_context && !tensor_is_leaf(in) && in->grad) {
            if (cml_ir_grad(in->ir_context, in->ir_node, in->grad, false) == 0)
                cml_ir_execute(in->ir_context);
        }
    }

    if (grad_inputs)
        cml_free(grad_inputs);
}

void cml_autograd_custom_backward_sweep(bool create_graph) {
    if (!g_nodes_head)
        return;

    if (create_graph && !g_warned_double_backward) {
        LOG_WARNING("custom autograd function: create_graph (double-backward) is not "
                    "supported across a custom op; its backward runs first-order only");
        g_warned_double_backward = true;
    }

    pthread_mutex_lock(&g_nodes_lock);

    /* Registration order is topological (a node is appended after its inputs
     * exist), so processing in reverse is a valid reverse-topological order:
     * gradients accumulated into inputs that are earlier custom outputs are
     * reached later in this same pass. */
    int count = 0;
    for (CustomNode* p = g_nodes_head; p; p = p->next)
        count++;
    CustomNode** arr = cml_malloc((size_t)count * sizeof(CustomNode*));
    if (!arr) {
        pthread_mutex_unlock(&g_nodes_lock);
        LOG_ERROR("custom backward sweep: allocation failed");
        return;
    }
    int idx = 0;
    for (CustomNode* p = g_nodes_head; p; p = p->next)
        arr[idx++] = p;

    for (int i = count - 1; i >= 0; i--) {
        CustomNode* node = arr[i];
        if (node->consumed || !node->output->grad)
            continue;
        run_node_backward(node);
        node->consumed = true;
    }

    CustomNode* prev = NULL;
    CustomNode* cur  = g_nodes_head;
    while (cur) {
        CustomNode* next = cur->next;
        if (cur->consumed) {
            if (prev)
                prev->next = next;
            else
                g_nodes_head = next;
            if (g_nodes_tail == cur)
                g_nodes_tail = prev;
            ctx_free(cur->ctx);
            if (cur->inputs)
                cml_free(cur->inputs);
            cml_free(cur);
        } else {
            prev = cur;
        }
        cur = next;
    }

    cml_free(arr);
    pthread_mutex_unlock(&g_nodes_lock);
}

void cml_autograd_custom_reset(void) {
    pthread_mutex_lock(&g_nodes_lock);
    CustomNode* cur = g_nodes_head;
    while (cur) {
        CustomNode* next = cur->next;
        ctx_free(cur->ctx);
        if (cur->inputs)
            cml_free(cur->inputs);
        cml_free(cur);
        cur = next;
    }
    g_nodes_head = NULL;
    g_nodes_tail = NULL;
    pthread_mutex_unlock(&g_nodes_lock);
}
