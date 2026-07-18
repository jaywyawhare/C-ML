/*
 * Graph-level reverse-mode autodiff.
 *
 * Instead of the eager cpu_backward_node engine (which writes ->grad->data
 * directly), this computes gradients by walking the primitive UOP graph in
 * reverse and emitting each primitive's VJP as *more UOP nodes* into the same
 * lazy graph, via the ordinary uop_* builders. The result: every parameter's
 * ->grad becomes a lazy IR subgraph, realized (and fusable) alongside forward.
 *
 * Run AFTER cml_ir_decompose so we only need VJPs for the minimal primitive
 * set. Covered here: ADD MUL MAX MATMUL SUM RESHAPE PERMUTE EXPAND EXP LOG
 * RECIP SQRT SIN NEG; FILL/CONST/CMPLT contribute no gradient. Ops without a
 * rule yet are skipped (grad does not flow through them) — the eager engine
 * remains the default until coverage is complete.
 */
#include "ops/ir/autodiff.h"
#include "ops/ir/internal.h"
#include "ops/ir/decompose.h"
#include "ops/uops.h"
#include "tensor/tensor.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include "alloc/cml_allocator.h"

/* Backward-mode selector: graph-level autodiff is now the DEFAULT (gradients
 * emitted as UOPs into the same lazy graph). The eager cpu_backward_node engine
 * remains available as a fallback via GRAD_MODE=eager (or =0). Cached. */
int cml_autodiff_use_graph(void) {
    static int cached = -1;
    if (cached < 0) {
        const char* m = getenv("GRAD_MODE");
        cached = (m && (strcmp(m, "eager") == 0 || strcmp(m, "0") == 0)) ? 0 : 1;
    }
    return cached;
}

/* ── value→grad map (linear scan; graphs are modest) ──────────────────── */
typedef struct { Tensor* val; Tensor* grad; } GradPair;
typedef struct { GradPair* items; int count, cap; } GradMap;

static Tensor* gm_get(GradMap* m, Tensor* v) {
    for (int i = 0; i < m->count; i++)
        if (m->items[i].val == v) return m->items[i].grad;
    return NULL;
}
static void gm_put(GradMap* m, Tensor* v, Tensor* g) {
    for (int i = 0; i < m->count; i++)
        if (m->items[i].val == v) { m->items[i].grad = g; return; }
    if (m->count == m->cap) {
        m->cap = m->cap ? m->cap * 2 : 16;
        m->items = cml_realloc(m->items, (size_t)m->cap * sizeof(GradPair));
    }
    m->items[m->count].val = v;
    m->items[m->count].grad = g;
    m->count++;
}
/* accumulate: grad[v] += contrib (sum on fan-out) */
static void gm_accum(GradMap* m, Tensor* v, Tensor* contrib) {
    if (!contrib) return;
    Tensor* g = gm_get(m, v);
    gm_put(m, v, g ? uop_add(g, contrib) : contrib);
}

/* ── param'd movement builders ────────────────────────────────────────── */
static Tensor* ad_reshape(Tensor* x, const int* shape, int ndim) {
    ReshapeParams p;
    int buf[16]; memcpy(buf, shape, (size_t)ndim * sizeof(int));
    p.new_shape = buf; p.new_ndim = ndim;
    return uop_reshape(x, &p);
}
static Tensor* ad_expand(Tensor* x, const int* shape, int ndim) {
    ExpandParams p;
    int buf[16]; memcpy(buf, shape, (size_t)ndim * sizeof(int));
    p.new_shape = buf; p.new_ndim = ndim;
    return uop_expand(x, &p);
}
static Tensor* ad_permute(Tensor* x, const int* perm, int ndim) {
    PermuteParams p;
    int buf[16]; memcpy(buf, perm, (size_t)ndim * sizeof(int));
    p.perm = buf; p.num_dims = ndim;
    return uop_permute(x, &p);
}
static Tensor* ad_sum(Tensor* x, const int* dims, int ndims, bool keepdim) {
    ReduceParams p;
    int buf[16];
    if (dims && ndims > 0) { memcpy(buf, dims, (size_t)ndims * sizeof(int)); p.dims = buf; }
    else p.dims = NULL;
    p.num_dims = ndims; p.keepdim = keepdim;
    return uop_sum(x, &p);
}
/* transpose the last two axes of an ndim tensor */
static Tensor* ad_transpose(Tensor* x, int ndim) {
    int perm[16];
    for (int i = 0; i < ndim; i++) perm[i] = i;
    if (ndim >= 2) { perm[ndim - 2] = ndim - 1; perm[ndim - 1] = ndim - 2; }
    return ad_permute(x, perm, ndim);
}

/* Reduce `g` (whose shape == some broadcast of `tshape`) back down to tshape:
 * sum over any leading extra axes and any axis where the target size is 1. */
static Tensor* unbroadcast(Tensor* g, const int* tshape, int tndim) {
    if (!g) return NULL;
    int gnd = g->ndim;
    if (gnd == tndim) {
        bool same = true;
        for (int i = 0; i < tndim; i++) if (g->shape[i] != tshape[i]) { same = false; break; }
        if (same) return g;
    }
    /* sum leading extra dims */
    int lead = gnd - tndim;
    for (int i = 0; i < lead; i++) {
        int d0[1] = {0};
        g = ad_sum(g, d0, 1, false);   /* drop axis 0 */
    }
    /* now g->ndim == tndim; sum axes where target is 1 but g is not */
    for (int i = 0; i < tndim; i++) {
        if (tshape[i] == 1 && g->shape[i] != 1) {
            int di[1] = {i};
            g = ad_sum(g, di, 1, true);  /* keepdim to preserve rank */
        }
    }
    return g;
}

/* ── the reverse pass ─────────────────────────────────────────────────── */
int cml_ir_grad(CMLGraph_t ir, struct IRNode* loss_node) {
    if (!ir || !loss_node || !loss_node->output) return -1;

    /* Ensure the FORWARD graph is fully decomposed to primitives before the
     * reverse walk. The is_decomposed flag is sticky per context; if the
     * context was realized early (e.g. weight init) any composites added
     * afterward (TANH, SIGMOID, …) stay un-lowered and would have no VJP.
     * Safe here: no backward nodes exist yet, so this only touches forward. */
    Tensor* loss_out = loss_node->output;   /* survives decompose (kept tensor) */
    ir->is_decomposed = false;
    cml_ir_decompose(ir);
    struct IRNode* resolved = (struct IRNode*)loss_out->ir_node;
    if (resolved) loss_node = resolved;

    /* collect nodes in forward (topological) order */
    int n = 0;
    for (struct IRNode* p = ir->head; p; p = p->next) n++;
    if (n == 0) return 0;
    struct IRNode** nodes = cml_malloc((size_t)n * sizeof(struct IRNode*));
    if (!nodes) return -1;
    { int i = 0; for (struct IRNode* p = ir->head; p; p = p->next) nodes[i++] = p; }

    GradMap map = {0};

    /* seed: d loss / d loss = ones */
    Tensor* lo = loss_node->output;
    gm_put(&map, lo, uop_fill(lo->shape, lo->ndim, 1.0f));

    for (int i = n - 1; i >= 0; i--) {
        struct IRNode* nd = nodes[i];
        Tensor* out = nd->output;
        Tensor* g = gm_get(&map, out);
        if (!g) continue;                 /* nothing downstream needs this */

        Tensor* a = nd->num_inputs > 0 ? nd->inputs[0] : NULL;
        Tensor* b = nd->num_inputs > 1 ? nd->inputs[1] : NULL;

        switch (nd->type) {
        case UOP_ADD:
            gm_accum(&map, a, unbroadcast(g, a->shape, a->ndim));
            gm_accum(&map, b, unbroadcast(g, b->shape, b->ndim));
            break;
        case UOP_MUL:
            gm_accum(&map, a, unbroadcast(uop_mul(g, b), a->shape, a->ndim));
            gm_accum(&map, b, unbroadcast(uop_mul(g, a), b->shape, b->ndim));
            break;
        case UOP_MAX: {
            /* grad flows to the larger operand (ties → a) */
            Tensor* ma = uop_cmpge(a, b);          /* 1 where a>=b */
            Tensor* mb = uop_cmplt(a, b);          /* 1 where a< b */
            gm_accum(&map, a, unbroadcast(uop_mul(g, ma), a->shape, a->ndim));
            gm_accum(&map, b, unbroadcast(uop_mul(g, mb), b->shape, b->ndim));
            break;
        }
        case UOP_MATMUL: {
            /* C=A@B: dA = dC @ Bᵀ, dB = Aᵀ @ dC (batched: transpose last two) */
            gm_accum(&map, a, uop_matmul(g, ad_transpose(b, b->ndim)));
            gm_accum(&map, b, uop_matmul(ad_transpose(a, a->ndim), g));
            break;
        }
        case UOP_LINEAR: {
            /* out = x @ Wᵀ + b, W=[N,K]. dX = g@W, dW = (flatten g)ᵀ @ (flatten x),
             * db = sum(flatten g over rows). Flatten batch dims to rows. */
            Tensor* W = b;                       /* inputs[1] */
            int Kf = a->shape[a->ndim - 1];
            int Nf = W->shape[0];
            int TR = (int)(a->numel / (size_t)Kf);
            gm_accum(&map, a, uop_matmul(g, W));
            int gf_s[2] = {TR, Nf}, xf_s[2] = {TR, Kf};
            Tensor* gf = ad_reshape(g, gf_s, 2);
            Tensor* xf = ad_reshape(a, xf_s, 2);
            gm_accum(&map, W, uop_matmul(ad_transpose(gf, 2), xf));
            if (nd->num_inputs > 2 && nd->inputs[2]) {
                int d0[1] = {0};
                gm_accum(&map, nd->inputs[2], ad_sum(gf, d0, 1, false));
            }
            break;
        }
        case UOP_SUM: {
            /* dX = broadcast(dOut) back to X's shape (via keepdim reshape) */
            ReduceParams* rp = (ReduceParams*)nd->params;
            int ks[16]; for (int d = 0; d < a->ndim; d++) ks[d] = a->shape[d];
            if (rp && rp->dims && rp->num_dims > 0)
                for (int k = 0; k < rp->num_dims; k++) {
                    int d = rp->dims[k]; if (d < 0) d += a->ndim; if (d >= 0 && d < a->ndim) ks[d] = 1;
                }
            else for (int d = 0; d < a->ndim; d++) ks[d] = 1;
            Tensor* gk = ad_reshape(g, ks, a->ndim);          /* to keepdim shape */
            gm_accum(&map, a, ad_expand(gk, a->shape, a->ndim));
            break;
        }
        case UOP_RESHAPE:
            gm_accum(&map, a, ad_reshape(g, a->shape, a->ndim));
            break;
        case UOP_PERMUTE: {
            PermuteParams* pp = (PermuteParams*)nd->params;
            int inv[16];
            if (pp && pp->perm) { for (int d = 0; d < pp->num_dims; d++) inv[pp->perm[d]] = d;
                                  gm_accum(&map, a, ad_permute(g, inv, pp->num_dims)); }
            break;
        }
        case UOP_EXPAND:
            gm_accum(&map, a, unbroadcast(g, a->shape, a->ndim));
            break;
        case UOP_UNFOLD: {                    /* dX = fold(g) — scatter windows back */
            UnfoldParams* up = (UnfoldParams*)nd->params;
            if (up) gm_accum(&map, a, uop_fold(g, up->kernel_size, up->stride, a->shape[a->ndim - 1]));
            break;
        }
        case UOP_FOLD: {                      /* dX = unfold(g) — the adjoint */
            FoldParams* fp = (FoldParams*)nd->params;
            if (fp) gm_accum(&map, a, uop_unfold(g, fp->kernel_size, fp->stride));
            break;
        }
        case UOP_GATHER: {                    /* dInput = scatter_add(idx, g); idx no grad */
            GatherParams* gp = (GatherParams*)nd->params;
            int dim = gp ? gp->dim : -1;
            if (dim < 0) dim += a->ndim;
            if (dim >= 0 && dim < a->ndim && nd->num_inputs > 1) {
                Tensor* idx = nd->inputs[1];
                /* NumPy-style gather with a 1-D index collapses the gathered dim,
                 * so g and idx have rank a->ndim-1. scatter_add is same-rank, so
                 * reshape both to a's shape with a size-1 slot at `dim` — the
                 * scatter then expands that slot back to a->shape[dim]. (This is
                 * the cross-entropy path: a=[N,C], dim=1, g/idx=[N] -> [N,1].)
                 * When g already matches a's rank (same-rank gather), fall through
                 * to the direct scatter. */
                int rshape[16];
                size_t want = 1;
                if (a->ndim <= 16) {
                    for (int i = 0; i < a->ndim; i++) rshape[i] = a->shape[i];
                    rshape[dim] = 1;
                    for (int i = 0; i < a->ndim; i++) want *= (size_t)rshape[i];
                }
                if (a->ndim <= 16 && g->numel == want && idx->numel == want) {
                    /* Reshape only the gradient: scatter_add's rank check is
                     * against src(=g), and its execution reads idx linearly, so
                     * the 1-D index can be passed as-is (no extra reshape node on
                     * the integer index tensor). */
                    Tensor* g2 = ad_reshape(g, rshape, a->ndim);
                    gm_accum(&map, a, uop_scatter_add(idx, g2, dim, a->shape[dim]));
                } else {
                    gm_accum(&map, a, uop_scatter_add(idx, g, dim, a->shape[dim]));
                }
            }
            break;
        }
        case UOP_STACK: {                     /* each input is slice t along stack dim */
            StackParams* sp = (StackParams*)nd->params;
            int sd = sp ? sp->dim : 0; if (sd < 0) sd += out->ndim;
            for (int t = 0; t < nd->num_inputs; t++) {
                Tensor* inp = nd->inputs[t]; if (!inp) continue;
                int st[16], en[16];
                for (int d = 0; d < out->ndim; d++) { st[d] = 0; en[d] = out->shape[d]; }
                st[sd] = t; en[sd] = t + 1;
                Tensor* sl = uop_shrink(g, st, en, out->ndim);      /* [.., 1, ..] */
                gm_accum(&map, inp, ad_reshape(sl, inp->shape, inp->ndim));
            }
            break;
        }
        case UOP_CAT: {                       /* each input is a contiguous span */
            CatParams* cp = (CatParams*)nd->params;
            int cd = cp ? cp->dim : 0; if (cd < 0) cd += out->ndim;
            int off = 0;
            for (int t = 0; t < nd->num_inputs; t++) {
                Tensor* inp = nd->inputs[t]; if (!inp) continue;
                int sz = inp->shape[cd];
                int st[16], en[16];
                for (int d = 0; d < out->ndim; d++) { st[d] = 0; en[d] = out->shape[d]; }
                st[cd] = off; en[cd] = off + sz;
                gm_accum(&map, inp, uop_shrink(g, st, en, out->ndim));
                off += sz;
            }
            break;
        }
        case UOP_SLICE: {                     /* dX = pad(g) back (step==1 only) */
            SliceParams* sp = (SliceParams*)nd->params;
            if (sp && sp->start && sp->end && a->ndim <= 16) {
                bool step1 = true;
                for (int d = 0; d < a->ndim; d++) if ((sp->step ? sp->step[d] : 1) != 1) step1 = false;
                if (step1) {
                    int pw[32];
                    for (int d = 0; d < a->ndim; d++) {
                        int s = sp->start[d]; if (s < 0) s += a->shape[d]; if (s < 0) s = 0; if (s > a->shape[d]) s = a->shape[d];
                        int e = sp->end[d];   if (e < 0) e += a->shape[d]; if (e < 0) e = 0; if (e > a->shape[d]) e = a->shape[d];
                        pw[2*d] = s; pw[2*d+1] = a->shape[d] - e;
                    }
                    gm_accum(&map, a, uop_pad(g, pw, a->ndim, 0.0f));
                }
            }
            break;
        }
        case UOP_EXP:                                     /* d/dx e^x = out */
            gm_accum(&map, a, uop_mul(g, out));
            break;
        case UOP_LOG:                                     /* 1/x */
            gm_accum(&map, a, uop_mul(g, uop_recip(a)));
            break;
        case UOP_RECIP:                                   /* -1/x² = -out² */
            gm_accum(&map, a, uop_mul(g, uop_neg(uop_mul(out, out))));
            break;
        case UOP_SQRT:                                    /* 0.5/sqrt(x) = 0.5/out */
            gm_accum(&map, a, uop_mul(g, uop_mul(uop_fill(out->shape, out->ndim, 0.5f), uop_recip(out))));
            break;
        case UOP_SIN:
            gm_accum(&map, a, uop_mul(g, uop_cos(a)));
            break;
        case UOP_NEG:
            gm_accum(&map, a, uop_neg(g));
            break;
        case UOP_POW: {                       /* out=aᵇ: da=g·b·out/a, db=g·out·ln(a) */
            gm_accum(&map, a, unbroadcast(uop_mul(uop_mul(uop_mul(g, b), out), uop_recip(a)), a->shape, a->ndim));
            gm_accum(&map, b, unbroadcast(uop_mul(uop_mul(g, out), uop_log(a)), b->shape, b->ndim));
            break;
        }
        case UOP_WHERE: {                     /* inputs [cond,a,b]; grad only to a,b */
            Tensor* cond = nd->inputs[0];
            Tensor* ta = nd->inputs[1], *tb = nd->inputs[2];
            Tensor* one = uop_fill(cond->shape, cond->ndim, 1.0f);
            Tensor* ncond = uop_sub(one, cond);
            gm_accum(&map, ta, unbroadcast(uop_mul(g, cond), ta->shape, ta->ndim));
            gm_accum(&map, tb, unbroadcast(uop_mul(g, ncond), tb->shape, tb->ndim));
            break;
        }
        case UOP_MAX_REDUCE: {                /* grad to the max position(s) */
            ReduceParams* rp = (ReduceParams*)nd->params;
            int ks[16]; for (int d = 0; d < a->ndim; d++) ks[d] = a->shape[d];
            if (rp && rp->dims && rp->num_dims > 0)
                for (int k = 0; k < rp->num_dims; k++) { int d = rp->dims[k]; if (d < 0) d += a->ndim; if (d>=0&&d<a->ndim) ks[d]=1; }
            else for (int d = 0; d < a->ndim; d++) ks[d] = 1;
            Tensor* ge = ad_expand(ad_reshape(g, ks, a->ndim), a->shape, a->ndim);
            Tensor* oe = ad_expand(ad_reshape(out, ks, a->ndim), a->shape, a->ndim);
            Tensor* mask = uop_cmpge(a, oe);          /* 1 at max positions */
            gm_accum(&map, a, uop_mul(ge, mask));
            break;
        }
        case UOP_PAD: {                       /* dX = shrink(g) to the interior */
            PadParams* pp = (PadParams*)nd->params;
            if (pp && pp->pad_widths) {
                int st[16], en[16];
                for (int d = 0; d < a->ndim; d++) { st[d] = pp->pad_widths[2*d]; en[d] = st[d] + a->shape[d]; }
                gm_accum(&map, a, uop_shrink(g, st, en, a->ndim));
            }
            break;
        }
        case UOP_SHRINK: {                    /* dX = pad(g) back to full */
            ShrinkParams* sp = (ShrinkParams*)nd->params;
            if (sp && sp->starts && sp->ends) {
                int pw[32];
                for (int d = 0; d < a->ndim; d++) { pw[2*d] = sp->starts[d]; pw[2*d+1] = a->shape[d] - sp->ends[d]; }
                gm_accum(&map, a, uop_pad(g, pw, a->ndim, 0.0f));
            }
            break;
        }
        /* no gradient: constants and boolean-producing ops */
        case UOP_FILL: case UOP_CONST:
        case UOP_CMPLT: case UOP_CMPGE:
        case UOP_CMPLE: case UOP_CMPGT: case UOP_CMPEQ: case UOP_CMPNE:
            break;
        default:
            /* uncovered primitive: gradient does not flow (yet) */
            break;
        }
    }

    /* publish: every value with requires_grad gets its lazy grad tensor.
     * Pin the grad as an external reference of its parent value. A grad may be
     * (or alias) a graph node output; without the pin, the graph teardown frees
     * it while the surviving parent still holds it via ->grad and frees it again
     * on destruction — a double free that corrupts the exec buffer cache across
     * models. Pinning makes the teardown detach-and-keep it instead; the parent
     * then owns it and releases the pin in tensor_free. Release any prior grad
     * (e.g. a parameter's grad from the previous step) first. */
    for (int i = 0; i < map.count; i++) {
        Tensor* v = map.items[i].val;
        if (v && v->requires_grad) {
            Tensor* newg = map.items[i].grad;
            if (v->grad != newg) {
                if (v->grad) tensor_release(v->grad);
                v->grad = newg;
                if (newg) tensor_pin(newg);
            }
        }
    }

    cml_free(map.items);
    cml_free(nodes);

    /* NOTE: the VJPs may emit a few composite ops (NEG/SUB/CMPGE/COS…). We do
     * NOT re-run cml_ir_decompose here — a second decompose pass over the mixed
     * forward+backward graph corrupts references (e.g. zeroes MAX_REDUCE masks).
     * The emitted composites execute correctly via the executor's own kernels;
     * lowering the backward to pure primitives is deferred. */
    return 0;
}
