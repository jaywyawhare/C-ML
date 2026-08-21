/*
 * Real elementwise kernel fusion.
 *
 * Collapses a maximal tree of primitive elementwise ops (all producing the same
 * output shape, connected by single-use edges) into ONE UOP_FUSED_ELEMENTWISE
 * node. cpu_execute_node then runs the whole chain as a single per-element loop
 * with register-held intermediates — the intermediate tensor buffers are never
 * materialized.
 *
 * Safety: an internal edge is only fused when the producer's output is used by
 * exactly one node (its consumer in the chain). A forward intermediate that a
 * backward VJP also reads therefore has use_count >= 2 and is left materialized,
 * so fusion never breaks autodiff. Run AFTER decompose (and after autodiff has
 * emitted the backward graph).
 */
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "ops/ir/intern.h"
#include "ops/uops.h"
#include "tensor/tensor.h"
#include "core/logging.h"
#include "core/cml_flags.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "alloc/cml_allocator.h"


/* Fusion is an optimization, so it is off whenever optimization is off.
 *
 * This lives at the entry to the fusers rather than at their call sites: the
 * execute paths reach them from three places, and gating each one separately
 * had already gone wrong -- NOOPT=1 skipped the passes in cml_ir_optimize but
 * the whole-graph execute paths fused anyway, so "optimization disabled" runs
 * still executed fused kernels. Gate the transform, not the callers. */
int cml_ir_fusion_enabled(void) {
    static int checked = 0;
    static int enabled = 0;
    if (!checked) {
        const char* env = getenv("FUSION_SCHEDULER");
        enabled = !(env && env[0] == '0') && !cml_flag_enabled(CML_FLAG_DISABLE_FUSION) &&
                  !cml_flag_enabled(CML_FLAG_NOOPT);
        checked = 1;
    }
    return enabled;
}

/* Ops the fused elementwise kernel (fused_eval_block / fe_scalar_eval) can run. */
static int fe_supported(UOpType t) {
    switch (t) {
    case UOP_ADD: case UOP_SUB: case UOP_MUL: case UOP_DIV: case UOP_MAX:
    case UOP_MINIMUM: case UOP_POW: case UOP_NEG: case UOP_RECIP: case UOP_EXP:
    case UOP_LOG: case UOP_SQRT: case UOP_SIN: case UOP_COS: case UOP_ABS:
    case UOP_RELU:
    case UOP_CMPLT: case UOP_CMPLE: case UOP_CMPGT: case UOP_CMPGE:
    case UOP_CMPEQ: case UOP_CMPNE: case UOP_WHERE: case UOP_FILL:
        return 1;
    default: return 0;
    }
}

static int find_idx(struct IRNode** nodes, int n, struct IRNode* p) {
    for (int i = 0; i < n; i++) if (nodes[i] == p) return i;
    return -1;
}

/* The fused executor is float32-only. Only fuse nodes whose output tensor is f32. */
static int node_is_f32(struct IRNode* n) {
    return n->output && n->output->dtype == DTYPE_FLOAT32;
}

static size_t node_numel(struct IRNode* n) {
    if (n->output && n->output->numel) return n->output->numel;
    if (n->output_shape && n->output_ndim > 0) {
        size_t p = 1;
        for (int i = 0; i < n->output_ndim; i++) p *= (size_t)n->output_shape[i];
        return p;
    }
    return 0;
}

/* True if `in` broadcasts over `rootnode`'s output shape via contiguous
 * trailing-dim tiling, so out[i] = in[i % in->numel] is the correct numpy
 * broadcast (e.g. a bias [N] over [M,N]). Requires in's dims to equal the
 * trailing dims of the output — every fused executor uses exactly i%numel. */
static int is_trailing_bcast(struct IRNode* rootnode, Tensor* in) {
    if (!in || !in->shape || in->ndim <= 0) return 0;
    int ond  = rootnode->output ? rootnode->output->ndim  : rootnode->output_ndim;
    int* osh = rootnode->output ? rootnode->output->shape : rootnode->output_shape;
    if (!osh || ond <= 0 || in->ndim > ond) return 0;
    for (int i = 0; i < in->ndim; i++)
        if (in->shape[i] != osh[ond - in->ndim + i]) return 0;
    return 1;
}

/* When set, the fuser may fuse requires_grad forward chains too, relying solely
 * on use_count to keep backward-needed intermediates materialized. This is only
 * safe when the WHOLE fwd+bwd graph is present (so use_count reflects backward
 * refs) and forward is not yet realized. tensor_backward sets it around the
 * combined-graph execute; it is 0 (conservative) everywhere else — a lone
 * forward realization must never fuse differentiable nodes. */
static __thread int g_fe_allow_grad = 0;
void cml_ir_fuse_set_allow_grad(int on) { g_fe_allow_grad = on ? 1 : 0; }

int cml_ir_fuse_elementwise(CMLGraph_t ir) {
    if (!ir || !ir->head || !cml_ir_fusion_enabled()) return 0;

    /* index nodes + count graph-internal uses of each node's output */
    int n = 0;
    for (struct IRNode* p = ir->head; p; p = p->next) n++;
    if (n < 2) return 0;

    struct IRNode** nodes = cml_malloc((size_t)n * sizeof(struct IRNode*));
    if (!nodes) return 0;
    { int i = 0; for (struct IRNode* p = ir->head; p; p = p->next) nodes[i++] = p; }

    int* use = cml_calloc((size_t)n, sizeof(int));      /* # graph nodes consuming node i */
    if (!use) { cml_free(nodes); return 0; }
    for (int i = 0; i < n; i++) {
        struct IRNode* nd = nodes[i];
        for (int k = 0; k < nd->num_inputs && nd->inputs; k++) {
            Tensor* in = nd->inputs[k];
            if (in && in->ir_node) {
                int pid = find_idx(nodes, n, (struct IRNode*)in->ir_node);
                if (pid >= 0) use[pid]++;
            }
        }
    }

    char* consumed = cml_calloc((size_t)n, 1);          /* node folded into a fused kernel */
    int fused_count = 0;

    /* Process consumers before producers so a chain's root is handled first. */
    for (int r = n - 1; r >= 0; r--) {
        struct IRNode* root = nodes[r];
        if (consumed[r] || !fe_supported(root->type)) continue;
        if (root->is_executed || (root->output && root->output->is_executed)) continue;
        /* Never fuse a differentiable node UNLESS the whole fwd+bwd graph is
         * present (g_fe_allow_grad, set by tensor_backward around the combined
         * execute). Otherwise forward realization runs BEFORE cml_ir_grad and a
         * fused forward node would be opaque to autodiff. With the flag set,
         * use_count below is the real guard: a forward intermediate a backward
         * VJP reads has use>=2 and stays materialized. Inference forwards
         * (requires_grad=false) and backward VJP chains fuse either way. */
        if (root->requires_grad && !g_fe_allow_grad) continue;
        if (!node_is_f32(root)) continue;
        size_t onumel = node_numel(root);
        if (onumel == 0) continue;

        /* Collect the maximal fusable set feeding `root` via single-use,
         * same-numel internal edges. members[] in reverse-collection order. */
        struct IRNode* members[256];
        int m = 0;
        members[m++] = root;
        consumed[r] = 1;
        /* BFS: for each member, pull in producers that qualify. */
        for (int qi = 0; qi < m; qi++) {
            struct IRNode* cur = members[qi];
            for (int k = 0; k < cur->num_inputs && cur->inputs; k++) {
                Tensor* in = cur->inputs[k];
                if (!in || !in->ir_node) continue;
                struct IRNode* prod = (struct IRNode*)in->ir_node;
                int pid = find_idx(nodes, n, prod);
                if (pid < 0) continue;
                if (consumed[pid]) continue;
                if (!fe_supported(prod->type)) continue;
                if (!node_is_f32(prod)) continue;          /* fused kernel is f32-only */
                if (prod->requires_grad && !g_fe_allow_grad) continue; /* see root guard */
                /* Already-materialized producer: its buffer is the source of truth
                 * (may have been mutated after realization, e.g. zeros() then
                 * filled). Read it as an external input rather than recomputing
                 * from the op's params (a FILL konst would give a stale value). */
                if (prod->is_executed || (prod->output && prod->output->is_executed)) continue;
                if (use[pid] != 1) continue;               /* used elsewhere -> keep materialized */
                if (node_numel(prod) != onumel) continue;  /* different shape -> external broadcast input */
                if (m >= 256) break;
                consumed[pid] = 1;
                members[m++] = prod;
            }
        }
        if (m < 2) { continue; }   /* nothing to fuse (single node) */

        /* Order members topologically = ascending graph position. */
        for (int i = 0; i < m; i++)
            for (int j = i + 1; j < m; j++)
                if (find_idx(nodes, n, members[j]) < find_idx(nodes, n, members[i])) {
                    struct IRNode* t = members[i]; members[i] = members[j]; members[j] = t;
                }

        /* step index of each member = its position in `members` (topo). */
        /* Build external input list + operand refs. */
        Tensor* ext[32]; int num_ext = 0;
        UOpType* ops = cml_malloc((size_t)m * sizeof(UOpType));
        int* aa = cml_malloc((size_t)m * sizeof(int));
        int* bb = cml_malloc((size_t)m * sizeof(int));
        int* cc = cml_malloc((size_t)m * sizeof(int));
        float* kk = cml_malloc((size_t)m * sizeof(float));
        if (!ops || !aa || !bb || !cc || !kk) { cml_free(ops); cml_free(aa); cml_free(bb); cml_free(cc); cml_free(kk); continue; }

        int ok = 1;
        for (int s = 0; s < m; s++) {
            struct IRNode* nd = members[s];
            ops[s] = nd->type;
            kk[s]  = 0.0f;
            int ref[3] = {FUSED_UNUSED_REF, FUSED_UNUSED_REF, FUSED_UNUSED_REF};
            if (nd->type == UOP_FILL) {
                FillParams* f = (FillParams*)nd->params;
                kk[s] = f ? f->value : 0.0f;
            } else {
                for (int k = 0; k < nd->num_inputs && k < 3; k++) {
                    Tensor* in = nd->inputs[k];
                    int internal = -1;
                    if (in && in->ir_node) {
                        struct IRNode* prod = (struct IRNode*)in->ir_node;
                        for (int j = 0; j < s; j++) if (members[j] == prod) { internal = j; break; }
                    }
                    if (internal >= 0) {
                        ref[k] = -(internal + 1);           /* prior step */
                    } else {
                        /* external input — every executor broadcasts via i%numel.
                         * Correct for full-size, scalar, OR a contiguous trailing
                         * broadcast (bias [N] over [M,N]). Any other partial
                         * broadcast (e.g. leading-dim [M,1]) is NOT i%n and aborts. */
                        size_t en = in ? in->numel : 0;
                        if (en != onumel && en != 1 &&
                            !(en > 0 && onumel % en == 0 && is_trailing_bcast(root, in))) { ok = 0; break; }
                        if (!in || in->dtype != DTYPE_FLOAT32) { ok = 0; break; }
                        int idx = -1;
                        for (int e = 0; e < num_ext; e++) if (ext[e] == in) { idx = e; break; }
                        if (idx < 0) { if (num_ext >= 32) { ok = 0; break; } idx = num_ext; ext[num_ext++] = in; }
                        ref[k] = idx;
                    }
                }
            }
            if (!ok) break;
            aa[s] = ref[0]; bb[s] = ref[1]; cc[s] = ref[2];
        }
        if (!ok) { cml_free(ops); cml_free(aa); cml_free(bb); cml_free(cc); cml_free(kk); continue; }

        FusedElementwiseParams* fp = cml_malloc(sizeof(FusedElementwiseParams));
        if (!fp) { cml_free(ops); cml_free(aa); cml_free(bb); cml_free(cc); cml_free(kk); continue; }
        fp->num_steps = m; fp->op = ops; fp->a = aa; fp->b = bb; fp->c = cc; fp->konst = kk;

        /* Create the fused node (not yet linked) with external inputs. */
        struct IRNode* fnode = cml_calloc(1, sizeof(struct IRNode));
        if (!fnode) { cml_free(fp); cml_free(ops); cml_free(aa); cml_free(bb); cml_free(cc); cml_free(kk); continue; }
        fnode->type = UOP_FUSED_ELEMENTWISE;
        /* Inherit provenance from the chain. The fused node is manufactured by
         * this pass, long after the model was built, so it has no creation stack
         * or module scope of its own -- and being the single hottest node in a
         * run, it would otherwise be the one frame the profile cannot attribute.
         * The root alone is not enough: it is often itself a lowered node from
         * decompose, so take the first member that carries provenance. */
        fnode->build_stack = NULL;
        fnode->scope       = NULL;
        for (int mi = 0; mi < m; mi++) {
            if (!fnode->build_stack && members[mi]->build_stack)
                fnode->build_stack = cml_strdup(members[mi]->build_stack);
            if (!fnode->scope && members[mi]->scope)
                fnode->scope = cml_strdup(members[mi]->scope);
        }
        fnode->num_inputs = num_ext;
        fnode->params = fp;
        fnode->inputs = cml_malloc((size_t)num_ext * sizeof(Tensor*));
        fnode->input_names = cml_calloc((size_t)num_ext, sizeof(char*));
        for (int e = 0; e < num_ext; e++) fnode->inputs[e] = ext[e];
        int ond = root->output ? root->output->ndim : root->output_ndim;
        int* osh = root->output ? root->output->shape : root->output_shape;
        if (osh && ond > 0) {
            fnode->output_shape = cml_malloc((size_t)ond * sizeof(int));
            memcpy(fnode->output_shape, osh, (size_t)ond * sizeof(int));
            fnode->output_ndim = ond;
        }
        fnode->requires_grad = root->requires_grad;

        /* Take over root's output tensor so downstream references stay valid. */
        Tensor* out = root->output;
        fnode->output = out;
        if (out) out->ir_node = fnode;

        /* Splice fnode into the list in place of root; unlink all members. */
        /* First, mark members for removal and find list neighbours of root. */
        struct IRNode* prev = NULL;
        for (struct IRNode* p = ir->head; p && p != root; p = p->next) prev = p;
        struct IRNode* after = root->next;
        if (prev) prev->next = fnode; else ir->head = fnode;
        fnode->next = after;
        if (ir->tail == root) ir->tail = fnode;

        /* Re-walk and unlink every consumed member still in the list. We free the
         * NODE structs but NEVER the member output tensors — those may be owned by
         * the user (e.g. an intermediate returned by cml_mul that the caller later
         * tensor_free()s) or by the IR context. We just detach them (ir_node=NULL)
         * so nothing dereferences the freed node. root's output was transferred to
         * fnode above, so its shell is freed with output already NULLed. */
        struct IRNode* pp = NULL;
        struct IRNode* cnode = ir->head;
        while (cnode) {
            int is_member = 0;
            for (int s = 0; s < m; s++) if (members[s] == cnode) { is_member = 1; break; }
            if (!is_member) { pp = cnode; cnode = cnode->next; continue; }
            struct IRNode* nx = cnode->next;
            if (pp) pp->next = nx; else ir->head = nx;
            if (ir->tail == cnode) ir->tail = pp;
            if (cnode == root) {
                cnode->output = NULL;   /* transferred to fnode */
            } else if (cnode->output) {
                cnode->output->ir_node = NULL;  /* detach; owner frees the tensor */
                cnode->output = NULL;
            }
            if (cnode->input_names) { for (int i = 0; i < cnode->num_inputs; i++) cml_free(cnode->input_names[i]); cml_free(cnode->input_names); }
            cml_free(cnode->inputs); cml_free(cnode->output_name); cml_free(cnode->output_shape);
            /* Leave the CSE table before the memory goes: it keys on node
             * identity, so a fused-away node left interned is dereferenced by
             * the next lookup that probes its slot. */
            cml_intern_remove(ir->intern_table, cnode);
            cml_ir_free_node_params(cnode); cml_free(cnode);
            cnode = nx;
        }

        /* root was spliced out of the list before the walk, so the loop above
         * never visited it — free its orphaned shell here. Its output tensor was
         * transferred to fnode, so NULL it first. */
        root->output = NULL;
        if (root->input_names) { for (int i = 0; i < root->num_inputs; i++) cml_free(root->input_names[i]); cml_free(root->input_names); }
        cml_free(root->inputs); cml_free(root->output_name); cml_free(root->output_shape);
        cml_intern_remove(ir->intern_table, root);
        cml_ir_free_node_params(root); cml_free(root);

        ir->node_count -= (m - 1);
        fused_count++;
        /* rebuild the index (positions shifted) */
        n = 0; for (struct IRNode* p = ir->head; p; p = p->next) n++;
        /* nodes[]/use[]/consumed[] are now stale; restart the whole pass. */
        cml_free(nodes); cml_free(use); cml_free(consumed);
        return fused_count + cml_ir_fuse_elementwise(ir);  /* re-run until fixpoint */
    }

    cml_free(nodes); cml_free(use); cml_free(consumed);
    return fused_count;
}

/* ── Matmul epilogue fusion ───────────────────────────────────────────────
 * Fold a bias-add + activation chain (already collapsed by the elementwise
 * fuser into ONE UOP_FUSED_ELEMENTWISE that reads the gemm output) INTO the
 * matmul node, so it is applied in-place to the M*N result — eliminating the
 * separate elementwise kernel's extra read+write pass over the output. The
 * epilogue is stored as a FusedElementwiseParams on the matmul node's params
 * (MATMUL_ACC_REF = the gemm result); every backend (BLAS, interpreter, JIT)
 * applies it. Runs AFTER cml_ir_fuse_elementwise. Returns #matmuls fused. */
int cml_ir_fuse_matmul_epilogue(CMLGraph_t ir) {
    if (!ir || !ir->head || !cml_ir_fusion_enabled()) return 0;
    int n = 0;
    for (struct IRNode* p = ir->head; p; p = p->next) n++;
    if (n < 2) return 0;

    struct IRNode** nodes = cml_malloc((size_t)n * sizeof(struct IRNode*));
    if (!nodes) return 0;
    { int i = 0; for (struct IRNode* p = ir->head; p; p = p->next) nodes[i++] = p; }
    int* use = cml_calloc((size_t)n, sizeof(int));
    if (!use) { cml_free(nodes); return 0; }
    for (int i = 0; i < n; i++) {
        struct IRNode* nd = nodes[i];
        for (int k = 0; k < nd->num_inputs && nd->inputs; k++) {
            Tensor* in = nd->inputs[k];
            if (in && in->ir_node) { int pid = find_idx(nodes, n, (struct IRNode*)in->ir_node); if (pid >= 0) use[pid]++; }
        }
    }

    int fused = 0;
    for (int i = 0; i < n; i++) {
        struct IRNode* M = nodes[i];
        if (M->type != UOP_MATMUL || M->params) continue;             /* plain gemm only */
        if (M->is_executed || (M->output && M->output->is_executed)) continue;
        /* Inference only: folding bias+activation into the gemm eliminates the
         * pre-activation intermediate, which the backward pass (e.g. relu's VJP)
         * needs. Never fold a differentiable matmul/epilogue. */
        if (M->requires_grad) continue;
        if (use[i] != 1 || !M->output) continue;                       /* single consumer */
        /* find the unique consumer F */
        struct IRNode* F = NULL; int mi = -1;
        for (int j = 0; j < n && !F; j++) {
            struct IRNode* c = nodes[j];
            for (int k = 0; k < c->num_inputs && c->inputs; k++)
                if (c->inputs[k] == M->output) { F = c; mi = k; break; }
        }
        if (!F || F->type != UOP_FUSED_ELEMENTWISE || !F->params) continue;
        if (F->requires_grad) continue;                  /* inference only (see above) */
        if (!F->output || !node_is_f32(F) || !node_is_f32(M)) continue;
        if (node_numel(F) != node_numel(M)) continue;   /* epilogue is elementwise over the gemm output */
        FusedElementwiseParams* sf = (FusedElementwiseParams*)F->params;
        if (F->num_inputs > 33 || sf->num_steps <= 0) continue;

        /* Build the epilogue: copy F's chain, remap operand refs. The input at
         * index mi (the gemm output) becomes MATMUL_ACC_REF; other external
         * inputs are re-indexed to the matmul's appended epilogue inputs. */
        int ns = sf->num_steps;
        FusedElementwiseParams* ep = cml_malloc(sizeof(FusedElementwiseParams));
        if (!ep) continue;
        ep->num_steps = ns;
        ep->op    = cml_malloc((size_t)ns * sizeof(UOpType));
        ep->a     = cml_malloc((size_t)ns * sizeof(int));
        ep->b     = cml_malloc((size_t)ns * sizeof(int));
        ep->c     = cml_malloc((size_t)ns * sizeof(int));
        ep->konst = cml_malloc((size_t)ns * sizeof(float));
        if (!ep->op || !ep->a || !ep->b || !ep->c || !ep->konst) {
            cml_free(ep->op); cml_free(ep->a); cml_free(ep->b); cml_free(ep->c); cml_free(ep->konst); cml_free(ep); continue;
        }
        #define REMAP(r) ( (r) < 0 ? (r) : ((r) == mi ? MATMUL_ACC_REF : ((r) > mi ? (r) - 1 : (r))) )
        for (int s = 0; s < ns; s++) {
            ep->op[s] = sf->op[s]; ep->konst[s] = sf->konst[s];
            ep->a[s] = REMAP(sf->a[s]); ep->b[s] = REMAP(sf->b[s]); ep->c[s] = REMAP(sf->c[s]);
        }
        #undef REMAP

        /* Append F's non-gemm inputs (bias, etc.) to M's inputs after A,B. */
        int num_epi = F->num_inputs - 1;
        Tensor** newin = cml_malloc((size_t)(2 + num_epi) * sizeof(Tensor*));
        if (!newin) { cml_free(ep->op); cml_free(ep->a); cml_free(ep->b); cml_free(ep->c); cml_free(ep->konst); cml_free(ep); continue; }
        newin[0] = M->inputs[0]; newin[1] = M->inputs[1];
        int w = 2;
        for (int k = 0; k < F->num_inputs; k++) if (k != mi) newin[w++] = F->inputs[k];
        cml_free(M->inputs);
        M->inputs = newin;
        /* input_names is indexed [0..num_inputs) by free_ir_node; growing
         * num_inputs without growing input_names would read out of bounds and
         * free garbage. Drop the old names (debug-only) and NULL it. */
        if (M->input_names) {
            for (int k = 0; k < M->num_inputs; k++) cml_free(M->input_names[k]);
            cml_free(M->input_names);
            M->input_names = NULL;
        }
        M->num_inputs = 2 + num_epi;
        M->params = ep;

        /* M now produces F's output (downstream references stay valid). The old
         * gemm-output tensor was single-use (only F); detach it. */
        if (M->output) M->output->ir_node = NULL;
        M->output = F->output;
        if (M->output) M->output->ir_node = M;
        if (M->output_shape) { cml_free(M->output_shape); M->output_shape = NULL; }
        if (F->output_shape && F->output_ndim > 0) {
            M->output_shape = cml_malloc((size_t)F->output_ndim * sizeof(int));
            if (M->output_shape) { memcpy(M->output_shape, F->output_shape, (size_t)F->output_ndim * sizeof(int)); M->output_ndim = F->output_ndim; }
        }

        /* Unlink + free F's shell (its output was transferred to M). */
        struct IRNode* prev = NULL;
        for (struct IRNode* p = ir->head; p && p != F; p = p->next) prev = p;
        if (prev) prev->next = F->next; else ir->head = F->next;
        if (ir->tail == F) ir->tail = prev;
        F->output = NULL;
        if (F->input_names) { for (int k = 0; k < F->num_inputs; k++) cml_free(F->input_names[k]); cml_free(F->input_names); }
        cml_free(F->inputs); cml_free(F->output_name); cml_free(F->output_shape);
        cml_ir_free_node_params(F); cml_free(F);

        ir->node_count -= 1;
        fused++;
        cml_free(nodes); cml_free(use);
        return fused + cml_ir_fuse_matmul_epilogue(ir);   /* indices shifted; restart */
    }

    cml_free(nodes); cml_free(use);
    return fused;
}
