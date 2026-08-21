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
/* A constant shaped like `t`, for gradient expressions that need one. */
static Tensor* ad_k(Tensor* t, float v) { return uop_fill(t->shape, t->ndim, v); }

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
/* Reverse cumulative sum along `dim`: revcum(x)[i] = sum over j >= i of x[j].
 *
 * The cumulative ops all need it and there is no flip primitive, so it is built
 * from the identity revcum = total - cumsum + x, which needs only a reduction
 * and a forward cumsum. */
static Tensor* ad_revcumsum(Tensor* x, int dim) {
    if (!x) return NULL;
    int d[1] = { dim };
    ReduceParams rp = { d, 1, true };
    Tensor* total = uop_sum(x, &rp);                     /* keepdim, broadcasts back */
    if (!total) return NULL;
    Tensor* tot_b = ad_expand(total, x->shape, x->ndim);
    Tensor* cum   = uop_cumsum(x, dim);
    if (!tot_b || !cum) return NULL;
    return uop_add(uop_sub(tot_b, cum), x);
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
/* Sum over several axes.
 *
 * The reduce kernels only implement num_dims == 1 -- a ReduceParams naming two
 * axes does not reduce both -- so this peels them one at a time. Highest axis
 * first, because dropping an axis shifts every index above it. */
static Tensor* ad_sum_axes(Tensor* x, int* axes, int n) {
    if (!x || n <= 0) return x;
    int a[16];
    memcpy(a, axes, (size_t)n * sizeof(int));
    for (int i = 0; i < n; i++)                       /* descending */
        for (int j = i + 1; j < n; j++)
            if (a[j] > a[i]) { int t = a[i]; a[i] = a[j]; a[j] = t; }
    Tensor* r = x;
    for (int i = 0; i < n && r; i++) {
        int d[1] = { a[i] };
        r = ad_sum(r, d, 1, false);
    }
    return r;
}

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
        case UOP_IM2COL: {                    /* dX = col2im(g) — scatter windows back */
            Im2colParams* ip = (Im2colParams*)nd->params;
            if (ip && a->ndim == 4) {
                Col2imParams cp = {ip->kh, ip->kw, ip->sh, ip->sw, ip->ph, ip->pw, ip->dh,
                                   ip->dw, a->shape[1], a->shape[2], a->shape[3]};
                gm_accum(&map, a, uop_col2im(g, &cp));
            }
            break;
        }
        case UOP_COL2IM: {                    /* dX = im2col(g) — the adjoint */
            Col2imParams* cp = (Col2imParams*)nd->params;
            if (cp) {
                Im2colParams ip = {cp->kh, cp->kw, cp->sh, cp->sw, cp->ph, cp->pw, cp->dh, cp->dw};
                gm_accum(&map, a, uop_im2col(g, &ip));
            }
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
        /* ── Inverse trig / hyperbolic, erf, saturating activations ────────
         * These had rules in the eager backward but not here, and this is the
         * path tensor_backward takes -- so a model using any of them trained
         * with no gradient and no error at all.
         *
         * The radicand is floored before the reciprocal square root, matching
         * the eager versions: asin/acos at |x|=1 and acosh at x=1 have an
         * infinite derivative, and clamping after would already have produced
         * an inf to propagate. */
        case UOP_ASIN:                        /* 1/sqrt(1-x²) */
            gm_accum(&map, a, uop_mul(g, uop_rsqrt(uop_max(uop_sub(ad_k(a, 1.0f), uop_square(a)),
                                                           ad_k(a, 1e-12f)))));
            break;
        case UOP_ACOS:                        /* -1/sqrt(1-x²) */
            gm_accum(&map, a, uop_neg(uop_mul(g, uop_rsqrt(uop_max(uop_sub(ad_k(a, 1.0f), uop_square(a)),
                                                                   ad_k(a, 1e-12f))))));
            break;
        case UOP_ATAN:                        /* 1/(1+x²) */
            gm_accum(&map, a, uop_mul(g, uop_recip(uop_add(ad_k(a, 1.0f), uop_square(a)))));
            break;
        case UOP_ASINH:                       /* 1/sqrt(x²+1) */
            gm_accum(&map, a, uop_mul(g, uop_rsqrt(uop_add(uop_square(a), ad_k(a, 1.0f)))));
            break;
        case UOP_ACOSH:                       /* 1/sqrt(x²-1) */
            gm_accum(&map, a, uop_mul(g, uop_rsqrt(uop_max(uop_sub(uop_square(a), ad_k(a, 1.0f)),
                                                           ad_k(a, 1e-12f)))));
            break;
        case UOP_ATANH:                       /* 1/(1-x²) */
            gm_accum(&map, a, uop_mul(g, uop_recip(uop_max(uop_sub(ad_k(a, 1.0f), uop_square(a)),
                                                           ad_k(a, 1e-12f)))));
            break;
        case UOP_ERF:                         /* 2/sqrt(pi) · e^(-x²) */
            gm_accum(&map, a, uop_mul(g, uop_mul(ad_k(a, 1.1283791670955126f),
                                                 uop_exp(uop_neg(uop_square(a))))));
            break;
        case UOP_SINH:                        /* cosh(x) */
            gm_accum(&map, a, uop_mul(g, uop_cosh(a)));
            break;
        case UOP_COSH:                        /* sinh(x) */
            gm_accum(&map, a, uop_mul(g, uop_sinh(a)));
            break;
        case UOP_LOG2:                        /* 1/(x·ln2) */
            gm_accum(&map, a, uop_mul(g, uop_recip(uop_mul(a, ad_k(a, 0.6931471805599453f)))));
            break;
        case UOP_LOG10:                       /* 1/(x·ln10) */
            gm_accum(&map, a, uop_mul(g, uop_recip(uop_mul(a, ad_k(a, 2.302585092994046f)))));
            break;
        case UOP_EXP2:                        /* 2^x·ln2 = out·ln2 */
            gm_accum(&map, a, uop_mul(g, uop_mul(out, ad_k(a, 0.6931471805599453f))));
            break;
        /* Saturating activations: the derivative is a constant inside the
         * linear band and zero outside it, so the mask is built from the
         * comparisons rather than from the (already clamped) output. */
        case UOP_HARD_SIGMOID:                /* 1/6 on (-3, 3), else 0 */
            gm_accum(&map, a, uop_mul(uop_mul(g, ad_k(a, 1.0f / 6.0f)),
                                      uop_mul(uop_cmpgt(a, ad_k(a, -3.0f)),
                                              uop_cmplt(a, ad_k(a,  3.0f)))));
            break;
        case UOP_HARD_TANH:                   /* 1 on (-1, 1), else 0 */
            gm_accum(&map, a, uop_mul(g, uop_mul(uop_cmpgt(a, ad_k(a, -1.0f)),
                                                 uop_cmplt(a, ad_k(a,  1.0f)))));
            break;
        case UOP_RELU6:                       /* 1 on (0, 6), else 0 */
            gm_accum(&map, a, uop_mul(g, uop_mul(uop_cmpgt(a, ad_k(a, 0.0f)),
                                                 uop_cmplt(a, ad_k(a, 6.0f)))));
            break;
        case UOP_QUICK_GELU: {                /* x·s(1.702x): s + 1.702·x·s·(1-s) */
            Tensor* sg = uop_sigmoid(uop_mul(a, ad_k(a, 1.702f)));
            Tensor* d  = uop_add(sg, uop_mul(uop_mul(ad_k(a, 1.702f), a),
                                             uop_mul(sg, uop_sub(ad_k(a, 1.0f), sg))));
            gm_accum(&map, a, uop_mul(g, d));
            break;
        }
        case UOP_SOFTPLUS:                    /* sigmoid(x) */
            gm_accum(&map, a, uop_mul(g, uop_sigmoid(a)));
            break;
        case UOP_SOFTSIGN:                    /* 1/(1+|x|)² */
            gm_accum(&map, a, uop_mul(g, uop_recip(uop_square(uop_add(ad_k(a, 1.0f), uop_abs(a))))));
            break;
        case UOP_LOGSIGMOID:                  /* 1 - sigmoid(x) = sigmoid(-x) */
            gm_accum(&map, a, uop_mul(g, uop_sigmoid(uop_neg(a))));
            break;
        /* Derivative is zero almost everywhere. Emitting *zeros* rather than
         * nothing matters: returning no gradient leaves the upstream chain
         * unwritten, so every parameter feeding a rounded value silently stops
         * training. This terminates the chain instead of breaking it. */
        case UOP_FLOOR:
        case UOP_CEIL:
        case UOP_ROUND:
        case UOP_SIGN:
            gm_accum(&map, a, ad_k(a, 0.0f));
            break;
        /* ── Structural / masking ops ─────────────────────────────────── */
        case UOP_TRIU: case UOP_TRIL: {       /* masked entries contributed nothing */
            TriParams* tp = (TriParams*)nd->params;
            int k = tp ? tp->diagonal : 0;
            gm_accum(&map, a, nd->type == UOP_TRIU ? uop_triu(g, k) : uop_tril(g, k));
            break;
        }
        case UOP_ROLL: {                      /* shift the gradient back */
            RollParams* rp = (RollParams*)nd->params;
            gm_accum(&map, a, uop_roll(g, rp ? -rp->shift : 0, rp ? rp->dim : 0));
            break;
        }
        case UOP_UNFLATTEN:                   /* pure relayout */
        case UOP_FLATTEN:
            gm_accum(&map, a, ad_reshape(g, a->shape, a->ndim));
            break;
        case UOP_MASKED_FILL:                 /* filled positions are constants */
            if (nd->num_inputs >= 2 && b)
                gm_accum(&map, a, uop_mul(g, uop_sub(ad_k(g, 1.0f), b)));
            else
                gm_accum(&map, a, g);
            break;
        /* ── Reductions ───────────────────────────────────────────────── */
        case UOP_PROD: {                      /* d/dx_i prod = prod / x_i */
            Tensor* ob = ad_expand(out, a->shape, a->ndim);
            Tensor* gb = ad_expand(g, a->shape, a->ndim);
            if (ob && gb) gm_accum(&map, a, uop_mul(gb, uop_div(ob, a)));
            break;
        }
        case UOP_LOGSUMEXP: {                 /* softmax(x) = e^(x - out) */
            Tensor* ob = ad_expand(out, a->shape, a->ndim);
            Tensor* gb = ad_expand(g, a->shape, a->ndim);
            if (ob && gb) gm_accum(&map, a, uop_mul(gb, uop_exp(uop_sub(a, ob))));
            break;
        }
        case UOP_TRACE: {                     /* only the diagonal contributes */
            Tensor* eye = uop_eye_op(a->shape[a->ndim - 2], a->dtype, a->device);
            Tensor* gb  = ad_expand(g, a->shape, a->ndim);
            if (eye && gb) gm_accum(&map, a, uop_mul(gb, eye));
            break;
        }
        /* ── Cumulative ops ───────────────────────────────────────────── */
        case UOP_CUMSUM: {                    /* each x_i feeds every later output */
            CumsumParams* cp = (CumsumParams*)nd->params;
            gm_accum(&map, a, ad_revcumsum(g, cp ? cp->dim : 0));
            break;
        }
        case UOP_CUMPROD: {                   /* revcum(g·out)/x */
            CumsumParams* cp = (CumsumParams*)nd->params;
            Tensor* r = ad_revcumsum(uop_mul(g, out), cp ? cp->dim : 0);
            if (r) gm_accum(&map, a, uop_div(r, a));
            break;
        }
        case UOP_LOGCUMSUMEXP: {              /* e^x_i · revcum(g·e^-out) */
            CumsumParams* cp = (CumsumParams*)nd->params;
            Tensor* r = ad_revcumsum(uop_mul(g, uop_exp(uop_neg(out))), cp ? cp->dim : 0);
            if (r) gm_accum(&map, a, uop_mul(uop_exp(a), r));
            break;
        }
        case UOP_CUMMAX: case UOP_CUMMIN: {
            /* The running extremum is carried by whichever element set it, so
             * the gradient goes to the positions where input equals output.
             * On a tie this spreads across the tied positions instead of
             * picking the first, because the op does not surface its indices. */
            CumsumParams* cp = (CumsumParams*)nd->params;
            (void)cp;
            gm_accum(&map, a, uop_mul(g, uop_cmpeq(a, out)));
            break;
        }
        /* ── Repetition: fold the copies back together ────────────────── */
        case UOP_TILE: {
            /* Exact rather than the eager rule's flat `i % numel` fold, which
             * only lands correctly when the repeats sit on the leading axis.
             * out[i0,i1,..] = in[i0 % s0, i1 % s1, ..], so viewing the gradient
             * as [r0,s0,r1,s1,..] puts every copy of an element on the even
             * axes; summing those is the fold, whatever the layout. */
            TileParams* tp = (TileParams*)nd->params;
            if (tp && tp->repeats && a->ndim <= 8 && a->ndim == out->ndim) {
                int vs[16], sd[8], nsd = 0;
                for (int d = 0; d < a->ndim; d++) {
                    vs[2*d]     = tp->repeats[d];
                    vs[2*d + 1] = a->shape[d];
                    if (tp->repeats[d] > 1) sd[nsd++] = 2*d;
                }
                Tensor* v = ad_reshape(g, vs, a->ndim * 2);
                Tensor* r = nsd ? ad_sum_axes(v, sd, nsd) : v;
                if (r) gm_accum(&map, a, ad_reshape(r, a->shape, a->ndim));
            }
            break;
        }
        case UOP_REPEAT_INTERLEAVE: {
            /* Each element became `reps` adjacent copies along `dim`, so that
             * axis views as [size, reps] and the gradient sums over reps. */
            RepeatInterleaveParams* rp = (RepeatInterleaveParams*)nd->params;
            int reps = rp && rp->repeats > 0 ? rp->repeats : 1;
            int dim  = rp ? rp->dim : 0; if (dim < 0) dim += a->ndim;
            if (reps > 1 && dim >= 0 && dim < a->ndim && a->ndim < 8) {
                int vs[16], k = 0;
                for (int d = 0; d < a->ndim; d++) {
                    vs[k++] = a->shape[d];
                    if (d == dim) vs[k++] = reps;
                }
                int sd[1] = { dim + 1 };
                Tensor* v = ad_reshape(g, vs, k);
                Tensor* r = ad_sum_axes(v, sd, 1);
                if (r) gm_accum(&map, a, ad_reshape(r, a->shape, a->ndim));
            } else {
                gm_accum(&map, a, ad_reshape(g, a->shape, a->ndim));
            }
            break;
        }
        /* ── Diagonals: diag and diagonal are each other's gradient ────── */
        case UOP_DIAG: {
            DiagParams* dp = (DiagParams*)nd->params;
            int off = dp ? dp->offset : 0;
            if (a->ndim == 1) gm_accum(&map, a, uop_diagonal(g, off, 0, 1));  /* 1-D -> matrix */
            else              gm_accum(&map, a, uop_diag(g, off));            /* matrix -> 1-D */
            break;
        }
        case UOP_DIAGONAL: {
            DiagParams* dp = (DiagParams*)nd->params;
            int off = dp ? dp->offset : 0;
            if (a->ndim == 2) gm_accum(&map, a, uop_diag(g, off));
            break;
        }
        /* ── Interpolation ────────────────────────────────────────────── */
        case UOP_LERP: {                      /* a + t·(b-a) */
            Tensor* t = (nd->num_inputs >= 3) ? nd->inputs[2] : NULL;
            if (t) {
                gm_accum(&map, a, unbroadcast(uop_mul(g, uop_sub(ad_k(g, 1.0f), t)), a->shape, a->ndim));
                if (b) gm_accum(&map, b, unbroadcast(uop_mul(g, t), b->shape, b->ndim));
                if (b) gm_accum(&map, t, unbroadcast(uop_mul(g, uop_sub(b, a)), t->shape, t->ndim));
            }
            break;
        }
        /* ── Scatter: the destination keeps everything it was not overwritten
         * at, and the source picks up the gradient at the positions it wrote. */
        case UOP_SCATTER: {
            if (nd->num_inputs >= 3) {
                Tensor* idx = nd->inputs[1];
                Tensor* src = nd->inputs[2];
                ScatterParams* sp = (ScatterParams*)nd->params;
                int dim = sp ? sp->dim : 0; if (dim < 0) dim += a->ndim;
                if (idx && src) {
                    gm_accum(&map, src, uop_gather(g, idx, dim));
                    gm_accum(&map, a, uop_scatter(g, dim, idx, ad_k(src, 0.0f)));
                }
            }
            break;
        }
        /* ── Permutations: send each output back where it came from ───── */
        case UOP_SORT: {
            /* sorted[i] = a[perm[i]], so the gradient of a[perm[i]] is g[i].
             * The node keeps only (dim, descending), not the permutation, so it
             * is recomputed -- sorting is deterministic, so recomputing gives
             * the same perm the forward used. */
            SortParams* sp = (SortParams*)nd->params;
            int dim = sp ? sp->dim : 0; if (dim < 0) dim += a->ndim;
            if (dim >= 0 && dim < a->ndim) {
                Tensor* perm = uop_argsort(a, dim, sp ? sp->descending : false);
                if (perm) gm_accum(&map, a, uop_scatter_add(perm, g, dim, a->shape[dim]));
            }
            break;
        }
        case UOP_TOPK: {
            /* Same idea, restricted to the k winners: the first k of the full
             * ordering are exactly the indices topk returned. */
            TopkParams* tp = (TopkParams*)nd->params;
            int dim = tp ? tp->dim : 0; if (dim < 0) dim += a->ndim;
            int k   = tp ? tp->k : 0;
            if (tp && dim >= 0 && dim < a->ndim && k > 0 && k <= a->shape[dim] && a->ndim <= 16) {
                Tensor* order = uop_argsort(a, dim, tp->largest);
                if (order) {
                    int st[16], en[16];
                    for (int d = 0; d < a->ndim; d++) { st[d] = 0; en[d] = a->shape[d]; }
                    en[dim] = k;
                    Tensor* idx = uop_shrink(order, st, en, a->ndim);
                    if (idx) gm_accum(&map, a, uop_scatter_add(idx, g, dim, a->shape[dim]));
                }
            }
            break;
        }
        case UOP_SCATTER_ADD: {
            /* out[index[i]] += src[i]; index carries no gradient. */
            if (nd->num_inputs >= 2) {
                Tensor* idx = nd->inputs[0];
                Tensor* src = nd->inputs[1];
                ScatterAddParams* sp = (ScatterAddParams*)nd->params;
                int dim = sp ? sp->dim : 0; if (dim < 0 && src) dim += src->ndim;
                if (idx && src) gm_accum(&map, src, uop_gather(g, idx, dim));
            }
            break;
        }
        case UOP_MASKED_SELECT: {
            /* out is the selected elements in order, so the j-th selected
             * element of `a` takes g[j]. cumsum(mask)-1 is exactly that j for
             * every selected position; multiplying by the mask discards what
             * the unselected positions gathered. */
            if (nd->num_inputs >= 2 && b) {
                Tensor* pos = uop_sub(uop_cumsum(b, 0), ad_k(b, 1.0f));
                Tensor* gg  = uop_gather(g, pos, 0);
                if (gg) gm_accum(&map, a, uop_mul(gg, b));
            }
            break;
        }
        case UOP_STRIDE:
            /* A restride is a relayout of the same elements. */
            if (a->numel == out->numel) gm_accum(&map, a, ad_reshape(g, a->shape, a->ndim));
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
            /* A comparison is flat wherever it is differentiable, so no
             * gradient flows through it.
             *
             * Emitting an explicit zero here instead would be closer to what
             * other frameworks report, and would give sign() -- which lowers to
             * (x>0)-(x<0) -- zeros rather than nothing. It is not done because
             * feeding a fresh constant back into the gradient map from inside
             * the backward walk expands the graph without bound and hangs. The
             * ops that need zero-terminating do it directly (see FLOOR/CEIL/
             * ROUND below). */
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
