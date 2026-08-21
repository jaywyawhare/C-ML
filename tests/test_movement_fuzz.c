/* Movement and shape ops over random ranks, axes and parameters.
 *
 * Every kernel in this family was written as `if (ndim == 1) ... else if
 * (ndim == 2) ...` with no general case, and the fall-through was never an
 * error -- roll, cat and repeat_interleave returned all-zeros on rank 3, pad
 * ignored its pad widths and flat-copied, scatter and nonzero silently did
 * nothing, and diagonal ignored dim1/dim2 entirely (it never even received
 * them: uop_diagonal validated the pair and then dropped it, since DiagParams
 * only carried `offset`). Every existing test used rank 1 or 2 inputs.
 *
 * So this fuzzes rank-1..4 shapes with random axes and checks every output
 * value against a reference computed here from strides, not by library code.
 */

#include "cml.h"
#include "test_require.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static int checks = 0, failures = 0;

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static uint64_t rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
static int rng_int(int lo, int hi) { return lo + (int)(rng_next() % (uint64_t)(hi - lo + 1)); }

#define MAXD 4

typedef enum {
    M_ROLL, M_REPEAT_INTERLEAVE, M_CAT, M_STACK, M_TILE,
    M_PAD_CONST, M_PAD_REFLECT, M_PAD_REPLICATE,
    M_DIAGONAL, M_SCATTER, M_MASKED_SELECT, M_GATHER, M_NONZERO,
    NUM_MV_OPS
} MvOp;

static const char* mv_name(int o) {
    static const char* n[NUM_MV_OPS] = {
        "roll", "repeat_interleave", "cat", "stack", "tile",
        "pad_const", "pad_reflect", "pad_replicate",
        "diagonal", "scatter", "masked_select", "gather", "nonzero"};
    return n[o];
}

static TensorConfig cfg(void) {
    TensorConfig c = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                      .has_dtype = true, .has_device = true};
    return c;
}

/* Distinct values so a misplaced element is always visible. */
static Tensor* mk(const int* shape, int ndim, float base, float step) {
    TensorConfig c = cfg();
    Tensor* t = tensor_zeros((int*)shape, ndim, &c);
    if (!t) return NULL;
    for (size_t i = 0; i < t->numel; i++)
        tensor_set_float(t, i, base + step * (float)i);
    return t;
}

static void strides_of(const int* shape, int ndim, size_t* st) {
    size_t acc = 1;
    for (int d = ndim - 1; d >= 0; d--) { st[d] = acc; acc *= (size_t)shape[d]; }
}

static void coords_of(size_t flat, const int* shape, int ndim, int* c) {
    for (int d = ndim - 1; d >= 0; d--) {
        c[d] = (int)(flat % (size_t)shape[d]);
        flat /= (size_t)shape[d];
    }
}

static size_t flat_of(const int* c, const size_t* st, int ndim) {
    size_t f = 0;
    for (int d = 0; d < ndim; d++) f += (size_t)c[d] * st[d];
    return f;
}

static size_t numel_of(const int* shape, int ndim) {
    size_t n = 1;
    for (int d = 0; d < ndim; d++) n *= (size_t)shape[d];
    return n;
}

static void report(int op, int iter, const char* what, const int* sh, int nd, int p0, int p1) {
    printf("  %-18s iter %d: %s   input [", mv_name(op), iter, what);
    for (int d = 0; d < nd; d++) printf("%d%s", sh[d], d + 1 < nd ? "," : "");
    printf("] params (%d, %d)\n", p0, p1);
    failures++;
}

/* Compare the produced tensor against an explicit expected shape and a
 * per-element reference callback over output coordinates. */
typedef float (*RefFn)(const int* out_coords, void* ctx);

static void check(int op, int iter, Tensor* r, const int* want_shape, int want_nd,
                  RefFn ref, void* ctx, const int* in_shape, int in_nd, int p0, int p1) {
    checks++;
    if (!r) { report(op, iter, "returned NULL", in_shape, in_nd, p0, p1); return; }
    tensor_ensure_executed(r);

    if (r->ndim != want_nd) {
        char buf[128];
        snprintf(buf, sizeof buf, "ndim %d, expected %d", r->ndim, want_nd);
        report(op, iter, buf, in_shape, in_nd, p0, p1);
        return;
    }
    for (int d = 0; d < want_nd; d++) {
        if (r->shape[d] != want_shape[d]) {
            char buf[160];
            int n = snprintf(buf, sizeof buf, "shape [");
            for (int i = 0; i < r->ndim; i++)
                n += snprintf(buf + n, sizeof buf - (size_t)n, "%d%s", r->shape[i],
                              i + 1 < r->ndim ? "," : "");
            n += snprintf(buf + n, sizeof buf - (size_t)n, "], expected [");
            for (int i = 0; i < want_nd; i++)
                n += snprintf(buf + n, sizeof buf - (size_t)n, "%d%s", want_shape[i],
                              i + 1 < want_nd ? "," : "");
            snprintf(buf + n, sizeof buf - (size_t)n, "]");
            report(op, iter, buf, in_shape, in_nd, p0, p1);
            return;
        }
    }

    size_t n_out = numel_of(want_shape, want_nd);
    for (size_t i = 0; i < n_out; i++) {
        int c[MAXD + 2];
        coords_of(i, want_shape, want_nd, c);
        float want = ref(c, ctx);
        float got  = tensor_get_float(r, i);
        if (fabsf(got - want) > fabsf(want) * 1e-5f + 1e-5f) {
            char buf[128];
            snprintf(buf, sizeof buf, "[%zu] = %g, expected %g", i, got, want);
            report(op, iter, buf, in_shape, in_nd, p0, p1);
            return;
        }
    }
}

/* ---- per-op reference contexts ---- */

typedef struct {
    const float* in;
    int shape[MAXD + 2];
    size_t st[MAXD + 2];
    int nd;
    int dim, p, q;
    const float* in2;
    PadMode mode;
    float pad_value;
    int pad[2 * MAXD];
} Ctx;

static float ref_roll(const int* c, void* v) {
    Ctx* x = v;
    int cc[MAXD + 2];
    memcpy(cc, c, sizeof(int) * (size_t)x->nd);
    int n = x->shape[x->dim];
    /* roll shifts content forward, so the source is `shift` behind. */
    int s = ((c[x->dim] - x->p) % n + n) % n;
    cc[x->dim] = s;
    return x->in[flat_of(cc, x->st, x->nd)];
}

static float ref_repeat_interleave(const int* c, void* v) {
    Ctx* x = v;
    int cc[MAXD + 2];
    memcpy(cc, c, sizeof(int) * (size_t)x->nd);
    cc[x->dim] = c[x->dim] / x->p;
    return x->in[flat_of(cc, x->st, x->nd)];
}

static float ref_cat(const int* c, void* v) {
    Ctx* x = v;
    int cc[MAXD + 2];
    memcpy(cc, c, sizeof(int) * (size_t)x->nd);
    int n = x->shape[x->dim];
    const float* src = x->in;
    if (c[x->dim] >= n) { cc[x->dim] = c[x->dim] - n; src = x->in2; }
    return src[flat_of(cc, x->st, x->nd)];
}

static float ref_stack(const int* c, void* v) {
    Ctx* x = v;
    /* output has nd+1 dims; drop the stacked axis to index the source */
    int cc[MAXD + 2];
    int k = 0;
    for (int d = 0; d <= x->nd; d++) {
        if (d == x->dim) continue;
        cc[k++] = c[d];
    }
    const float* src = (c[x->dim] == 0) ? x->in : x->in2;
    return src[flat_of(cc, x->st, x->nd)];
}

static float ref_tile(const int* c, void* v) {
    Ctx* x = v;
    int cc[MAXD + 2];
    for (int d = 0; d < x->nd; d++) cc[d] = c[d] % x->shape[d];
    return x->in[flat_of(cc, x->st, x->nd)];
}

static float ref_pad(const int* c, void* v) {
    Ctx* x = v;
    int cc[MAXD + 2];
    for (int d = 0; d < x->nd; d++) {
        int sc = c[d] - x->pad[2 * d];
        int n  = x->shape[d];
        if (sc < 0 || sc >= n) {
            if (x->mode == PAD_REFLECT) {
                sc = (sc < 0) ? -sc : 2 * n - 2 - sc;
                if (sc < 0) sc = 0;
                if (sc >= n) sc = n - 1;
            } else if (x->mode == PAD_REPLICATE) {
                sc = (sc < 0) ? 0 : n - 1;
            } else {
                return x->pad_value;
            }
        }
        cc[d] = sc;
    }
    return x->in[flat_of(cc, x->st, x->nd)];
}

static float ref_diagonal(const int* c, void* v) {
    Ctx* x = v;
    /* output drops dim2; dim1's slot carries the diagonal index */
    int cc[MAXD + 2];
    int slot = 0, oi = 0;
    int map[MAXD + 2];
    for (int d = 0; d < x->nd; d++) {
        if (d == x->q) continue;
        if (d == x->p) slot = oi;
        map[oi++] = d;
    }
    int k = c[slot];
    for (int j = 0; j < oi; j++)
        if (j != slot) cc[map[j]] = c[j];
    int off = x->dim; /* offset lives in ->dim for this op */
    cc[x->p] = (off >= 0) ? k : k - off;
    cc[x->q] = (off >= 0) ? k + off : k;
    return x->in[flat_of(cc, x->st, x->nd)];
}

static float ref_gather(const int* c, void* v) {
    Ctx* x = v;
    /* index-select semantics: 1-D indices along ->dim (see uops.h) */
    int cc[MAXD + 2];
    memcpy(cc, c, sizeof(int) * (size_t)x->nd);
    cc[x->dim] = (int)x->in2[c[x->dim]];
    return x->in[flat_of(cc, x->st, x->nd)];
}

/* ---- cases ---- */

static void one_case(int op, int iter) {
    int nd = rng_int(1, MAXD);
    int shape[MAXD];
    for (int d = 0; d < nd; d++) shape[d] = rng_int(1, 4);
    int dim = rng_int(0, nd - 1);

    Tensor* a = mk(shape, nd, 1.0f, 1.0f);
    if (!a) { failures++; cml_reset_ir_context(); return; }
    const float* in = (const float*)tensor_data_ptr(a);
    float saved[512];
    if (a->numel > 512) { cml_reset_ir_context(); return; }
    memcpy(saved, in, a->numel * sizeof(float));

    Ctx x;
    memset(&x, 0, sizeof x);
    x.in = saved;
    x.nd = nd;
    x.dim = dim;
    memcpy(x.shape, shape, sizeof(int) * (size_t)nd);
    strides_of(shape, nd, x.st);

    int want[MAXD + 2];
    memcpy(want, shape, sizeof(int) * (size_t)nd);

    switch (op) {
    case M_ROLL: {
        int shift = rng_int(-3, 3);
        x.p = shift;
        check(op, iter, uop_roll(a, shift, dim), want, nd, ref_roll, &x, shape, nd, shift, dim);
        break;
    }
    case M_REPEAT_INTERLEAVE: {
        int rep = rng_int(1, 3);
        x.p = rep;
        want[dim] = shape[dim] * rep;
        check(op, iter, uop_repeat_interleave(a, rep, dim), want, nd,
              ref_repeat_interleave, &x, shape, nd, rep, dim);
        break;
    }
    case M_CAT: {
        Tensor* b = mk(shape, nd, 100.0f, 1.0f);
        float saved2[512];
        memcpy(saved2, tensor_data_ptr(b), b->numel * sizeof(float));
        x.in2 = saved2;
        want[dim] = shape[dim] * 2;
        Tensor* ts[2] = {a, b};
        check(op, iter, uop_cat(ts, 2, dim), want, nd, ref_cat, &x, shape, nd, 2, dim);
        break;
    }
    case M_STACK: {
        Tensor* b = mk(shape, nd, 100.0f, 1.0f);
        float saved2[512];
        memcpy(saved2, tensor_data_ptr(b), b->numel * sizeof(float));
        x.in2 = saved2;
        int sdim = rng_int(0, nd);          /* stack may insert at the end */
        x.dim = sdim;
        int w2[MAXD + 2];
        int k = 0;
        for (int d = 0; d <= nd; d++) w2[d] = 0;
        for (int d = 0; d < sdim; d++) w2[k++] = shape[d];
        w2[k++] = 2;
        for (int d = sdim; d < nd; d++) w2[k++] = shape[d];
        Tensor* ts[2] = {a, b};
        check(op, iter, uop_stack(ts, 2, sdim), w2, nd + 1, ref_stack, &x, shape, nd, 2, sdim);
        break;
    }
    case M_TILE: {
        int reps[MAXD];
        for (int d = 0; d < nd; d++) {
            reps[d] = rng_int(1, 2);
            want[d] = shape[d] * reps[d];
        }
        check(op, iter, uop_tile(a, reps, nd), want, nd, ref_tile, &x, shape, nd, reps[0], nd);
        break;
    }
    case M_PAD_CONST:
    case M_PAD_REFLECT:
    case M_PAD_REPLICATE: {
        int pad[2 * MAXD];
        for (int d = 0; d < nd; d++) {
            /* reflect needs at least 2 elements along a padded axis and cannot
             * pad by more than n-1; replicate and constant are unconstrained. */
            int cap = (op == M_PAD_REFLECT) ? shape[d] - 1 : 2;
            if (cap < 0) cap = 0;
            pad[2 * d]     = rng_int(0, cap);
            pad[2 * d + 1] = rng_int(0, cap);
            want[d] = shape[d] + pad[2 * d] + pad[2 * d + 1];
        }
        memcpy(x.pad, pad, sizeof pad);
        x.mode = (op == M_PAD_CONST) ? PAD_CONSTANT
               : (op == M_PAD_REFLECT) ? PAD_REFLECT : PAD_REPLICATE;
        x.pad_value = -7.0f;
        Tensor* r = (op == M_PAD_CONST)   ? uop_pad(a, pad, nd, -7.0f)
                  : (op == M_PAD_REFLECT) ? uop_pad_reflect(a, pad, nd)
                                          : uop_pad_replicate(a, pad, nd);
        check(op, iter, r, want, nd, ref_pad, &x, shape, nd, pad[0], pad[1]);
        break;
    }
    case M_DIAGONAL: {
        if (nd < 2) { cml_reset_ir_context(); return; }
        int d1 = rng_int(0, nd - 1), d2 = rng_int(0, nd - 1);
        if (d1 == d2) { cml_reset_ir_context(); return; }
        int rows = shape[d1], cols = shape[d2];
        int diag = rows < cols ? rows : cols;            /* offset 0 only */
        x.p = d1; x.q = d2; x.dim = 0;                   /* ->dim carries the offset */
        int w2[MAXD + 2];
        int k = 0;
        for (int d = 0; d < nd; d++) {
            if (d == d2) continue;
            w2[k++] = (d == d1) ? diag : shape[d];
        }
        check(op, iter, uop_diagonal(a, 0, d1, d2), w2, nd - 1, ref_diagonal, &x,
              shape, nd, d1, d2);
        break;
    }
    case M_SCATTER: {
        /* index and src share a's shape; last write along `dim` wins. */
        Tensor* idx = mk(shape, nd, 0.0f, 0.0f);
        Tensor* src = mk(shape, nd, 100.0f, 10.0f);
        int target = rng_int(0, shape[dim] - 1);
        for (size_t i = 0; i < idx->numel; i++) tensor_set_float(idx, i, (float)target);
        float srcv[512];
        memcpy(srcv, tensor_data_ptr(src), src->numel * sizeof(float));

        Tensor* r = uop_scatter(a, dim, idx, src);
        checks++;
        if (!r) { report(op, iter, "returned NULL", shape, nd, dim, target); break; }
        tensor_ensure_executed(r);

        /* Reference: copy, then replay every write in order. */
        float exp_[512];
        memcpy(exp_, saved, a->numel * sizeof(float));
        for (size_t j = 0; j < a->numel; j++) {
            int c[MAXD];
            coords_of(j, shape, nd, c);
            c[dim] = target;
            exp_[flat_of(c, x.st, nd)] = srcv[j];
        }
        for (size_t i = 0; i < a->numel; i++) {
            float got = tensor_get_float(r, i);
            if (fabsf(got - exp_[i]) > 1e-5f) {
                char buf[96];
                snprintf(buf, sizeof buf, "[%zu] = %g, expected %g", i, got, exp_[i]);
                report(op, iter, buf, shape, nd, dim, target);
                break;
            }
        }
        break;
    }
    case M_MASKED_SELECT: {
        Tensor* m = mk(shape, nd, 0.0f, 0.0f);
        size_t expect = 0;
        float keep[512];
        for (size_t i = 0; i < a->numel; i++) {
            int on = rng_int(0, 1);
            tensor_set_float(m, i, (float)on);
            if (on) keep[expect++] = saved[i];
        }
        Tensor* r = uop_masked_select(a, m);
        checks++;
        if (!r) { report(op, iter, "returned NULL", shape, nd, 0, 0); break; }
        tensor_ensure_executed(r);
        if (r->ndim != 1 || r->numel != expect) {
            char buf[96];
            snprintf(buf, sizeof buf, "ndim %d numel %zu, expected 1-D numel %zu",
                     r->ndim, r->numel, expect);
            report(op, iter, buf, shape, nd, 0, 0);
            break;
        }
        for (size_t i = 0; i < expect; i++) {
            float got = tensor_get_float(r, i);
            if (fabsf(got - keep[i]) > 1e-5f) {
                char buf[96];
                snprintf(buf, sizeof buf, "[%zu] = %g, expected %g", i, got, keep[i]);
                report(op, iter, buf, shape, nd, 0, 0);
                break;
            }
        }
        break;
    }
    case M_GATHER: {
        /* uops.h: indices must be 1-D; output takes their length on `dim`. */
        int n_idx = rng_int(1, 4);
        int ish[1] = {n_idx};
        Tensor* idx = mk(ish, 1, 0.0f, 0.0f);
        float idxv[8];
        for (int i = 0; i < n_idx; i++) {
            idxv[i] = (float)rng_int(0, shape[dim] - 1);
            tensor_set_float(idx, (size_t)i, idxv[i]);
        }
        x.in2 = idxv;
        want[dim] = n_idx;
        /* only the documented dim=0 (row select) and last-dim forms are general */
        if (dim != 0) { cml_reset_ir_context(); return; }
        check(op, iter, uop_gather(a, idx, dim), want, nd, ref_gather, &x, shape, nd,
              n_idx, dim);
        break;
    }
    case M_NONZERO: {
        /* uop_nonzero counts at build time, so the shape is exact: [count] for
         * rank 1, else [count, ndim] with one row of coordinates per hit. */
        Tensor* z = mk(shape, nd, 0.0f, 0.0f);
        int hits[512], nhit = 0;
        for (size_t i = 0; i < a->numel; i++) {
            int on = rng_int(0, 1);
            tensor_set_float(z, i, on ? (float)(i + 1) : 0.0f);
            if (on) hits[nhit++] = (int)i;
        }
        Tensor* r = uop_nonzero(z);
        checks++;
        if (!r) { report(op, iter, "returned NULL", shape, nd, nhit, nd); break; }
        tensor_ensure_executed(r);

        int want_nd = (nd == 1) ? 1 : 2;
        size_t want_numel = (nd == 1) ? (size_t)nhit : (size_t)nhit * (size_t)nd;
        if (r->ndim != want_nd || r->numel != want_numel) {
            char buf[112];
            snprintf(buf, sizeof buf, "ndim %d numel %zu, expected ndim %d numel %zu",
                     r->ndim, r->numel, want_nd, want_numel);
            report(op, iter, buf, shape, nd, nhit, nd);
            break;
        }
        for (int h = 0; h < nhit; h++) {
            int c[MAXD];
            coords_of((size_t)hits[h], shape, nd, c);
            for (int d = 0; d < nd; d++) {
                float want = (nd == 1) ? (float)hits[h] : (float)c[d];
                float got  = tensor_get_float(r, (size_t)h * (size_t)nd + (size_t)d);
                if (fabsf(got - want) > 1e-5f) {
                    char buf[96];
                    snprintf(buf, sizeof buf, "row %d dim %d = %g, expected %g", h, d, got, want);
                    report(op, iter, buf, shape, nd, nhit, nd);
                    h = nhit;
                    break;
                }
            }
        }
        break;
    }
    default: break;
    }
    cml_reset_ir_context();
}

int main(void) {
    cml_init();
    printf("Movement ops over random ranks, axes and parameters (%d ops):\n", NUM_MV_OPS);

    const int ITERS = 60;
    for (int op = 0; op < NUM_MV_OPS && failures < 20; op++)
        for (int i = 0; i < ITERS && failures < 20; i++) one_case(op, i);

    printf("\n%d cases, %d failures\n", checks, failures);
    cml_cleanup();
    return failures ? 1 : 0;
}
