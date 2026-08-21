/* Parameterized ops -- the family every earlier fuzzer structurally skipped.
 *
 * The existing fuzzers each enumerate one *signature shape*: test_unary_fuzz
 * takes the `Tensor* uop_x(Tensor*)` entry points, test_broadcast_fuzz the
 * two-tensor binaries, test_ternary_fuzz the three-tensor forms, and
 * test_axis_ops_fuzz everything that accepts a ReduceParams. An op whose entry
 * point carries an extra scalar or its own params struct -- uop_elu(x, alpha),
 * uop_sum_dim(a, dim, keepdim), uop_split(a, n, dim, &out) -- matches none of
 * those shapes, so it was invisible to all four by construction rather than by
 * oversight. Cross-checking uops.h against every tests/*.c found 36 public
 * entry points that no test so much as named.
 *
 * That blind spot is why celu(NaN) still returned 0 after the NaN sweep fixed
 * relu, relu6, hard_tanh, clamp, max and the reductions: celu is reached only
 * through uop_celu(x, alpha), so no enumerated list contained it.
 *
 * This covers the entry points in that set with an independently checkable
 * reference -- shape and every value, computed here from strides.
 */

#include "cml.h"
#include "test_require.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static int checks = 0, failures = 0;

static uint64_t rng_state = 0xD1B54A32D192ED03ULL;
static uint64_t rng_next(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return rng_state;
}
static int rng_int(int lo, int hi) { return lo + (int)(rng_next() % (uint64_t)(hi - lo + 1)); }

#define MAXD 4
#define MAXN 256

static Tensor* mk(const int* shape, int ndim, int seed) {
    TensorConfig c = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                      .has_dtype = true, .has_device = true};
    Tensor* t = tensor_zeros((int*)shape, ndim, &c);
    if (!t) return NULL;
    /* spans both sides of zero so the negative branch of elu/celu/leaky is hit */
    for (size_t i = 0; i < t->numel; i++)
        tensor_set_float(t, i, (float)(((int)(i * 7 + (size_t)seed * 3) % 23) - 11) * 0.5f);
    return t;
}

static void fail_shape(const char* name, Tensor* r, const int* want, int want_nd) {
    printf("  %-14s shape [", name);
    for (int i = 0; i < r->ndim; i++) printf("%d%s", r->shape[i], i + 1 < r->ndim ? "," : "");
    printf("], expected [");
    for (int i = 0; i < want_nd; i++) printf("%d%s", want[i], i + 1 < want_nd ? "," : "");
    printf("]\n");
    failures++;
}

static int shape_is(Tensor* r, const int* want, int want_nd) {
    if (r->ndim != want_nd) return 0;
    for (int i = 0; i < want_nd; i++) if (r->shape[i] != want[i]) return 0;
    return 1;
}

static int close_enough(float got, double want) {
    if (isnan(want)) return isnan(got);
    double tol = fabs(want) * 1e-4 + 1e-4;
    return fabs((double)got - want) <= tol;
}

/* ---------- 1. parameterized elementwise activations ---------- */

typedef enum { A_ELU, A_CELU, A_LEAKY, NUM_ACTS } ActOp;
static const char* act_name(int o) {
    static const char* n[NUM_ACTS] = {"elu", "celu", "leaky_relu"};
    return n[o];
}

static double act_ref(int op, double x, double p) {
    switch (op) {
    case A_ELU:   return x > 0 ? x : p * (exp(x) - 1.0);
    case A_CELU:  return (x > 0 ? x : 0.0) + fmin(0.0, p * (exp(x / p) - 1.0));
    case A_LEAKY: return x > 0 ? x : p * x;
    default:      return 0;
    }
}

static Tensor* act_build(int op, Tensor* a, float p) {
    switch (op) {
    case A_ELU:   return uop_elu(a, p);
    case A_CELU:  return uop_celu(a, p);
    case A_LEAKY: return uop_leaky_relu(a, p);
    default:      return NULL;
    }
}

static void act_case(int op, int iter) {
    int nd = rng_int(1, MAXD);
    int shape[MAXD];
    for (int i = 0; i < nd; i++) shape[i] = rng_int(1, 4);

    Tensor* a = mk(shape, nd, iter + 1);
    if (!a) { failures++; return; }
    if (a->numel > MAXN) { cml_reset_ir_context(); return; }

    float saved[MAXN];
    size_t n = a->numel;
    memcpy(saved, tensor_data_ptr(a), n * sizeof(float));

    /* alpha of a CELU must stay away from 0: the reference and the kernel both
     * divide by it. */
    float p = (op == A_LEAKY) ? (float)rng_int(1, 40) * 0.01f
                              : (float)rng_int(5, 30) * 0.1f;

    Tensor* r = act_build(op, a, p);
    checks++;
    if (!r) { printf("  %-14s NULL (rank %d)\n", act_name(op), nd); failures++;
              cml_reset_ir_context(); return; }
    tensor_ensure_executed(r);

    if (!shape_is(r, shape, nd)) { fail_shape(act_name(op), r, shape, nd);
                                   cml_reset_ir_context(); return; }

    for (size_t i = 0; i < n; i++) {
        double want = act_ref(op, saved[i], p);
        float got = tensor_get_float(r, i);
        if (!close_enough(got, want)) {
            printf("  %-14s [%zu] = %g, expected %g   (x=%g, param=%g)\n",
                   act_name(op), i, got, want, saved[i], p);
            failures++;
            break;
        }
    }
    cml_reset_ir_context();
}

/* A NaN reaching a clamping activation must come out the other side. An
 * ordered compare -- and equally fmaxf/fminf, which return the *non*-NaN
 * operand per IEEE minNum/maxNum -- silently replaces it with a plausible
 * finite number, which is the failure mode that hides a diverged tensor. */
static void act_nan_case(int op) {
    int shape[1] = {3};
    Tensor* a = mk(shape, 1, 1);
    if (!a) { failures++; return; }
    tensor_set_float(a, 0, NAN);

    float p = (op == A_LEAKY) ? 0.01f : 1.0f;
    Tensor* r = act_build(op, a, p);
    checks++;
    if (!r) { printf("  %-14s nan: NULL\n", act_name(op)); failures++;
              cml_reset_ir_context(); return; }
    tensor_ensure_executed(r);
    float v = tensor_get_float(r, 0);
    if (!isnan(v)) {
        printf("  %-14s nan: got %g, expected nan (NaN was swallowed)\n", act_name(op), v);
        failures++;
    }
    cml_reset_ir_context();
}

/* ---------- 2. dim reductions with keepdim ---------- */

typedef enum { D_SUM, D_MEAN, D_MAX, NUM_DIMOPS } DimOp;
static const char* dim_name(int o) {
    static const char* n[NUM_DIMOPS] = {"sum_dim", "mean_dim", "max_dim"};
    return n[o];
}

static void dim_case(int op, int iter) {
    int nd = rng_int(1, MAXD);
    int shape[MAXD];
    for (int i = 0; i < nd; i++) shape[i] = rng_int(1, 4);
    int dim     = rng_int(0, nd - 1);
    bool keep   = (rng_int(0, 1) == 1);

    Tensor* a = mk(shape, nd, iter + 7);
    if (!a) { failures++; return; }
    if (a->numel > MAXN) { cml_reset_ir_context(); return; }

    float saved[MAXN];
    memcpy(saved, tensor_data_ptr(a), a->numel * sizeof(float));

    Tensor* r = op == D_SUM  ? uop_sum_dim(a, dim, keep)
              : op == D_MEAN ? uop_mean_dim(a, dim, keep)
                             : uop_max_reduce_dim(a, dim, keep);
    checks++;
    if (!r) { printf("  %-14s NULL (rank %d dim %d keepdim %d)\n",
                     dim_name(op), nd, dim, (int)keep);
              failures++; cml_reset_ir_context(); return; }
    tensor_ensure_executed(r);

    /* keepdim leaves the reduced axis as 1; otherwise it is dropped, except
     * that reducing the only axis of a rank-1 tensor still yields [1]. */
    int want[MAXD + 1]; int wn = 0;
    if (keep) {
        for (int i = 0; i < nd; i++) want[wn++] = (i == dim) ? 1 : shape[i];
    } else if (nd == 1) {
        want[wn++] = 1;
    } else {
        for (int i = 0; i < nd; i++) if (i != dim) want[wn++] = shape[i];
    }
    if (!shape_is(r, want, wn)) {
        printf("  %-14s (rank %d dim %d keepdim %d) ", dim_name(op), nd, dim, (int)keep);
        fail_shape("", r, want, wn);
        cml_reset_ir_context();
        return;
    }

    size_t outer = 1, inner = 1, count = (size_t)shape[dim];
    for (int i = 0; i < dim; i++) outer *= (size_t)shape[i];
    for (int i = dim + 1; i < nd; i++) inner *= (size_t)shape[i];

    for (size_t o = 0; o < outer; o++) {
        for (size_t m = 0; m < inner; m++) {
            size_t base = o * count * inner + m;
            double want_v;
            if (op == D_MAX) {
                want_v = saved[base];
                for (size_t j = 1; j < count; j++) {
                    double v = saved[base + j * inner];
                    if (v > want_v) want_v = v;
                }
            } else {
                want_v = 0;
                for (size_t j = 0; j < count; j++) want_v += saved[base + j * inner];
                if (op == D_MEAN) want_v /= (double)count;
            }
            size_t oi = o * inner + m;
            if (oi >= r->numel) continue;
            float got = tensor_get_float(r, oi);
            if (!close_enough(got, want_v)) {
                printf("  %-14s [%zu] = %g, expected %g   rank %d dim %d keepdim %d\n",
                       dim_name(op), oi, got, want_v, nd, dim, (int)keep);
                failures++;
                cml_reset_ir_context();
                return;
            }
        }
    }
    cml_reset_ir_context();
}

/* ---------- 3. softmax over an arbitrary dim ---------- */

static void softmax_case(int iter) {
    int nd = rng_int(1, MAXD);
    int shape[MAXD];
    for (int i = 0; i < nd; i++) shape[i] = rng_int(1, 4);
    int dim = rng_int(0, nd - 1);

    Tensor* a = mk(shape, nd, iter + 13);
    if (!a) { failures++; return; }
    if (a->numel > MAXN) { cml_reset_ir_context(); return; }

    float saved[MAXN];
    memcpy(saved, tensor_data_ptr(a), a->numel * sizeof(float));

    Tensor* r = uop_softmax(a, dim);
    checks++;
    if (!r) { printf("  %-14s NULL (rank %d dim %d)\n", "softmax", nd, dim); failures++;
              cml_reset_ir_context(); return; }
    tensor_ensure_executed(r);

    if (!shape_is(r, shape, nd)) { fail_shape("softmax", r, shape, nd);
                                   cml_reset_ir_context(); return; }

    size_t outer = 1, inner = 1, count = (size_t)shape[dim];
    for (int i = 0; i < dim; i++) outer *= (size_t)shape[i];
    for (int i = dim + 1; i < nd; i++) inner *= (size_t)shape[i];

    for (size_t o = 0; o < outer; o++) {
        for (size_t m = 0; m < inner; m++) {
            size_t base = o * count * inner + m;
            double mx = saved[base];
            for (size_t j = 1; j < count; j++)
                if (saved[base + j * inner] > mx) mx = saved[base + j * inner];
            double sum = 0;
            for (size_t j = 0; j < count; j++) sum += exp(saved[base + j * inner] - mx);
            for (size_t j = 0; j < count; j++) {
                double want = exp(saved[base + j * inner] - mx) / sum;
                size_t oi = base + j * inner;
                float got = tensor_get_float(r, oi);
                if (!close_enough(got, want)) {
                    printf("  %-14s [%zu] = %g, expected %g   rank %d dim %d\n",
                           "softmax", oi, got, want, nd, dim);
                    failures++;
                    cml_reset_ir_context();
                    return;
                }
            }
        }
    }
    cml_reset_ir_context();
}

/* ---------- 4. masked_fill ---------- */

static void masked_fill_case(int iter) {
    int nd = rng_int(1, MAXD);
    int shape[MAXD];
    for (int i = 0; i < nd; i++) shape[i] = rng_int(1, 4);

    Tensor* a = mk(shape, nd, iter + 17);
    if (!a) { failures++; return; }
    if (a->numel > MAXN) { cml_reset_ir_context(); return; }

    TensorConfig c = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                      .has_dtype = true, .has_device = true};
    Tensor* mask = tensor_zeros(shape, nd, &c);
    if (!mask) { failures++; cml_reset_ir_context(); return; }

    float saved[MAXN], msk[MAXN];
    size_t n = a->numel;
    memcpy(saved, tensor_data_ptr(a), n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        float m = (float)rng_int(0, 1);
        tensor_set_float(mask, i, m);
        msk[i] = m;
    }

    float fill = (float)rng_int(-5, 5);
    Tensor* r = uop_masked_fill(a, mask, fill);
    checks++;
    if (!r) { printf("  %-14s NULL (rank %d)\n", "masked_fill", nd); failures++;
              cml_reset_ir_context(); return; }
    tensor_ensure_executed(r);

    if (!shape_is(r, shape, nd)) { fail_shape("masked_fill", r, shape, nd);
                                   cml_reset_ir_context(); return; }

    for (size_t i = 0; i < n; i++) {
        double want = (msk[i] != 0.0f) ? (double)fill : (double)saved[i];
        float got = tensor_get_float(r, i);
        if (!close_enough(got, want)) {
            printf("  %-14s [%zu] = %g, expected %g   (mask=%g fill=%g)\n",
                   "masked_fill", i, got, want, msk[i], fill);
            failures++;
            break;
        }
    }
    cml_reset_ir_context();
}

/* ---------- 5. unflatten ---------- */

static void unflatten_case(int iter) {
    int nd = rng_int(1, 3);
    int shape[MAXD];
    for (int i = 0; i < nd; i++) shape[i] = rng_int(1, 4);
    int dim = rng_int(0, nd - 1);

    /* factor shape[dim] into two sizes so the split is exact */
    int d = shape[dim], f = 1;
    for (int cand = d; cand >= 1; cand--) if (d % cand == 0) { f = cand; break; }
    int sizes[2] = {f, d / f};

    Tensor* a = mk(shape, nd, iter + 23);
    if (!a) { failures++; return; }
    if (a->numel > MAXN) { cml_reset_ir_context(); return; }

    float saved[MAXN];
    size_t n = a->numel;
    memcpy(saved, tensor_data_ptr(a), n * sizeof(float));

    Tensor* r = uop_unflatten(a, dim, sizes, 2);
    checks++;
    if (!r) { printf("  %-14s NULL (rank %d dim %d -> %d,%d)\n",
                     "unflatten", nd, dim, sizes[0], sizes[1]);
              failures++; cml_reset_ir_context(); return; }
    tensor_ensure_executed(r);

    int want[MAXD + 1]; int wn = 0;
    for (int i = 0; i < nd; i++) {
        if (i == dim) { want[wn++] = sizes[0]; want[wn++] = sizes[1]; }
        else          { want[wn++] = shape[i]; }
    }
    if (!shape_is(r, want, wn)) { fail_shape("unflatten", r, want, wn);
                                  cml_reset_ir_context(); return; }

    /* unflatten only re-labels axes: row-major order is untouched. */
    for (size_t i = 0; i < n; i++) {
        float got = tensor_get_float(r, i);
        if (!close_enough(got, saved[i])) {
            printf("  %-14s [%zu] = %g, expected %g (data must be unchanged)\n",
                   "unflatten", i, got, saved[i]);
            failures++;
            break;
        }
    }
    cml_reset_ir_context();
}

/* ---------- 6. split / chunk ---------- */

static void split_case(int iter, int use_chunk) {
    const char* nm = use_chunk ? "chunk" : "split";
    int nd = rng_int(1, 3);
    int shape[MAXD];
    for (int i = 0; i < nd; i++) shape[i] = rng_int(1, 5);
    int dim = rng_int(0, nd - 1);

    Tensor* a = mk(shape, nd, iter + 29);
    if (!a) { failures++; return; }
    if (a->numel > MAXN) { cml_reset_ir_context(); return; }

    float saved[MAXN];
    memcpy(saved, tensor_data_ptr(a), a->numel * sizeof(float));

    int dim_size = shape[dim];
    int arg = rng_int(1, dim_size);
    int piece = use_chunk ? (dim_size + arg - 1) / arg : arg;   /* elements per piece */
    int n_out = 0;
    Tensor** out = use_chunk ? uop_chunk(a, arg, dim, &n_out)
                             : uop_split(a, arg, dim, &n_out);
    checks++;
    if (!out) { printf("  %-14s NULL (rank %d dim %d arg %d)\n", nm, nd, dim, arg);
                failures++; cml_reset_ir_context(); return; }

    int want_n = (dim_size + piece - 1) / piece;
    if (n_out != want_n) {
        printf("  %-14s produced %d pieces, expected %d (dim_size %d, per piece %d)\n",
               nm, n_out, want_n, dim_size, piece);
        failures++;
        cml_free(out);
        cml_reset_ir_context();
        return;
    }

    size_t outer = 1, inner = 1;
    for (int i = 0; i < dim; i++) outer *= (size_t)shape[i];
    for (int i = dim + 1; i < nd; i++) inner *= (size_t)shape[i];

    int bad = 0;
    for (int k = 0; k < n_out && !bad; k++) {
        int start = k * piece;
        int end   = start + piece; if (end > dim_size) end = dim_size;
        int len   = end - start;

        Tensor* p = out[k];
        if (!p) { printf("  %-14s piece %d NULL\n", nm, k); failures++; break; }
        tensor_ensure_executed(p);

        int want[MAXD];
        for (int i = 0; i < nd; i++) want[i] = (i == dim) ? len : shape[i];
        if (!shape_is(p, want, nd)) {
            printf("  %-14s piece %d ", nm, k);
            fail_shape("", p, want, nd);
            bad = 1;
            break;
        }

        for (size_t o = 0; o < outer && !bad; o++) {
            for (int j = 0; j < len && !bad; j++) {
                for (size_t m = 0; m < inner; m++) {
                    size_t src = o * (size_t)dim_size * inner + (size_t)(start + j) * inner + m;
                    size_t dst = o * (size_t)len * inner + (size_t)j * inner + m;
                    float got = tensor_get_float(p, dst);
                    if (!close_enough(got, saved[src])) {
                        printf("  %-14s piece %d [%zu] = %g, expected %g   "
                               "rank %d dim %d arg %d\n",
                               nm, k, dst, got, saved[src], nd, dim, arg);
                        failures++;
                        bad = 1;
                        break;
                    }
                }
            }
        }
    }
    cml_free(out);
    cml_reset_ir_context();
}

/* ---------- 7. expand_to ---------- */

static void expand_case(int iter) {
    int nd = rng_int(1, 3);
    int in_shape[MAXD];
    for (int i = 0; i < nd; i++) in_shape[i] = (rng_int(0, 1) == 0) ? 1 : rng_int(2, 3);

    /* target may gain leading dims; each size-1 input dim may stretch */
    int prepend = rng_int(0, 1);
    int out_nd  = nd + prepend;
    int out_shape[MAXD + 1];
    for (int i = 0; i < prepend; i++) out_shape[i] = rng_int(1, 3);
    for (int i = 0; i < nd; i++)
        out_shape[prepend + i] = (in_shape[i] == 1) ? rng_int(1, 3) : in_shape[i];

    Tensor* a = mk(in_shape, nd, iter + 31);
    if (!a) { failures++; return; }
    if (a->numel > MAXN) { cml_reset_ir_context(); return; }

    float saved[MAXN];
    memcpy(saved, tensor_data_ptr(a), a->numel * sizeof(float));

    Tensor* r = uop_expand_to(a, out_shape, out_nd);
    checks++;
    if (!r) { printf("  %-14s NULL (rank %d -> %d)\n", "expand_to", nd, out_nd); failures++;
              cml_reset_ir_context(); return; }
    tensor_ensure_executed(r);

    if (!shape_is(r, out_shape, out_nd)) { fail_shape("expand_to", r, out_shape, out_nd);
                                           cml_reset_ir_context(); return; }

    size_t in_str[MAXD]; size_t s = 1;
    for (int i = nd - 1; i >= 0; i--) { in_str[i] = s; s *= (size_t)in_shape[i]; }

    for (size_t i = 0; i < r->numel; i++) {
        size_t rem = i, src = 0;
        for (int d2 = out_nd - 1; d2 >= 0; d2--) {
            int coord = (int)(rem % (size_t)out_shape[d2]);
            rem /= (size_t)out_shape[d2];
            int id = d2 - prepend;                       /* right-aligned input axis */
            if (id >= 0 && in_shape[id] != 1) src += (size_t)coord * in_str[id];
        }
        float got = tensor_get_float(r, i);
        if (!close_enough(got, saved[src])) {
            printf("  %-14s [%zu] = %g, expected %g   in rank %d out rank %d\n",
                   "expand_to", i, got, saved[src], nd, out_nd);
            failures++;
            break;
        }
    }
    cml_reset_ir_context();
}

int main(void) {
    cml_init();
    printf("Parameterized ops (entry points no other fuzzer enumerates):\n");

    const int ITERS = 40;
    const int CAP   = 20;

    for (int op = 0; op < NUM_ACTS && failures < CAP; op++) {
        for (int i = 0; i < ITERS && failures < CAP; i++) act_case(op, i);
        act_nan_case(op);
    }
    for (int op = 0; op < NUM_DIMOPS && failures < CAP; op++)
        for (int i = 0; i < ITERS && failures < CAP; i++) dim_case(op, i);
    for (int i = 0; i < ITERS && failures < CAP; i++) softmax_case(i);
    for (int i = 0; i < ITERS && failures < CAP; i++) masked_fill_case(i);
    for (int i = 0; i < ITERS && failures < CAP; i++) unflatten_case(i);
    for (int i = 0; i < ITERS && failures < CAP; i++) split_case(i, 0);
    for (int i = 0; i < ITERS && failures < CAP; i++) split_case(i, 1);
    for (int i = 0; i < ITERS && failures < CAP; i++) expand_case(i);

    printf("\n%d cases, %d failures\n", checks, failures);
    cml_cleanup();
    return failures ? 1 : 0;
}
