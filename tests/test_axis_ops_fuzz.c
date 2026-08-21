/* Axis-taking ops over random shapes and random axes.
 *
 * Reductions, cumulative ops and sorts all pick an axis, and an axis bug is
 * invisible whenever the test only uses 1-D inputs or only reduces the last
 * dimension. Earlier in this codebase a multi-axis reduce silently ignored
 * every axis after the first, and topk's multi-dimensional branch was a bulk
 * memcpy -- both had passing tests.
 *
 * Each case: a random rank-1..4 shape, a random axis, and both properties
 * checked against a reference computed here using the (outer, count, inner)
 * decomposition -- the output shape, and every value.
 */

#include "cml.h"
#include "test_require.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static int checks = 0, failures = 0;

static uint64_t rng_state = 0x853C49E6748FEA9BULL;
static uint64_t rng_next(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return rng_state;
}
static int rng_int(int lo, int hi) { return lo + (int)(rng_next() % (uint64_t)(hi - lo + 1)); }

#define MAXD 4

typedef enum {
    R_SUM, R_MEAN, R_PROD, R_MAX, R_MIN, R_VAR, R_STD, R_ANY, R_ALL, R_LOGSUMEXP,
    R_ARGMAX, R_ARGMIN,
    C_CUMSUM, C_CUMPROD, C_CUMMAX, C_CUMMIN,
    S_SORT_ASC, S_SORT_DESC, S_ARGSORT,
    NUM_AX_OPS
} AxOp;

static const char* ax_name(int o) {
    static const char* n[NUM_AX_OPS] = {
        "sum","mean","prod","max_reduce","min_reduce","var","std","any","all","logsumexp",
        "argmax","argmin","cumsum","cumprod","cummax","cummin",
        "sort_asc","sort_desc","argsort"};
    return n[o];
}
static int is_reduction(int o) { return o <= R_ARGMIN; }

static Tensor* mk(const int* shape, int ndim, int seed) {
    TensorConfig c = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                      .has_dtype = true, .has_device = true};
    Tensor* t = tensor_zeros((int*)shape, ndim, &c);
    if (!t) return NULL;
    /* distinct values so argmax/argmin/sort have a unique answer */
    for (size_t i = 0; i < t->numel; i++)
        tensor_set_float(t, i, (float)(((int)(i * 7 + (size_t)seed * 3) % 23) - 11) * 0.5f);
    return t;
}

static Tensor* build(int op, Tensor* a, int axis) {
    int dims[1] = {axis};
    ReduceParams rp = {.dims = dims, .num_dims = 1, .keepdim = false};
    switch (op) {
    case R_SUM:       return uop_sum(a, &rp);
    case R_MEAN:      return uop_mean(a, &rp);
    case R_PROD:      return uop_prod(a, &rp);
    case R_MAX:       return uop_max_reduce(a, &rp);
    case R_MIN:       return uop_min_reduce(a, &rp);
    case R_VAR:       return uop_var(a, &rp);
    case R_STD:       return uop_std(a, &rp);
    case R_ANY:       return uop_any(a, &rp);
    case R_ALL:       return uop_all(a, &rp);
    case R_LOGSUMEXP: return uop_logsumexp(a, &rp);
    case R_ARGMAX:    return uop_argmax(a, &rp);
    case R_ARGMIN:    return uop_argmin(a, &rp);
    case C_CUMSUM:    return uop_cumsum(a, axis);
    case C_CUMPROD:   return uop_cumprod(a, axis);
    case C_CUMMAX:    return uop_cummax(a, axis);
    case C_CUMMIN:    return uop_cummin(a, axis);
    case S_SORT_ASC:  return uop_sort(a, axis, false);
    case S_SORT_DESC: return uop_sort(a, axis, true);
    case S_ARGSORT:   return uop_argsort(a, axis, false);
    default: return NULL;
    }
}

/* Reference over one lane: elements are at base + j*inner for j in [0,count). */
static void ref_lane(int op, const float* in, size_t base, size_t inner, size_t count,
                     float* out_lane) {
    if (is_reduction(op)) {
        double acc = 0;
        size_t bi = 0;
        switch (op) {
        case R_SUM: case R_MEAN:
            for (size_t j = 0; j < count; j++) acc += in[base + j * inner];
            if (op == R_MEAN) acc /= (double)count;
            break;
        case R_PROD:
            acc = 1;
            for (size_t j = 0; j < count; j++) acc *= in[base + j * inner];
            break;
        case R_MAX: case R_ARGMAX:
            acc = in[base];
            for (size_t j = 1; j < count; j++)
                if (in[base + j * inner] > acc) { acc = in[base + j * inner]; bi = j; }
            if (op == R_ARGMAX) acc = (double)bi;
            break;
        case R_MIN: case R_ARGMIN:
            acc = in[base];
            for (size_t j = 1; j < count; j++)
                if (in[base + j * inner] < acc) { acc = in[base + j * inner]; bi = j; }
            if (op == R_ARGMIN) acc = (double)bi;
            break;
        case R_VAR: case R_STD: {
            double m = 0;
            for (size_t j = 0; j < count; j++) m += in[base + j * inner];
            m /= (double)count;
            double v = 0;
            for (size_t j = 0; j < count; j++) {
                double d = in[base + j * inner] - m;
                v += d * d;
            }
            v /= (double)count;                     /* biased, as documented */
            acc = (op == R_STD) ? sqrt(v) : v;
            break;
        }
        case R_ANY:
            acc = 0;
            for (size_t j = 0; j < count; j++) if (in[base + j * inner] != 0) { acc = 1; break; }
            break;
        case R_ALL:
            acc = 1;
            for (size_t j = 0; j < count; j++) if (in[base + j * inner] == 0) { acc = 0; break; }
            break;
        default: {                                  /* logsumexp */
            double mx = in[base];
            for (size_t j = 1; j < count; j++) if (in[base + j * inner] > mx) mx = in[base + j * inner];
            double s = 0;
            for (size_t j = 0; j < count; j++) s += exp(in[base + j * inner] - mx);
            acc = mx + log(s);
            break;
        }
        }
        out_lane[0] = (float)acc;
        return;
    }

    if (op >= S_SORT_ASC) {                          /* sort / argsort */
        size_t idx[64];
        for (size_t j = 0; j < count; j++) idx[j] = j;
        for (size_t x = 0; x < count; x++)           /* selection sort, stable enough */
            for (size_t y = x + 1; y < count; y++) {
                float vx = in[base + idx[x] * inner], vy = in[base + idx[y] * inner];
                int swap = (op == S_SORT_DESC) ? (vy > vx) : (vy < vx);
                if (swap) { size_t t = idx[x]; idx[x] = idx[y]; idx[y] = t; }
            }
        for (size_t j = 0; j < count; j++)
            out_lane[j] = (op == S_ARGSORT) ? (float)idx[j] : in[base + idx[j] * inner];
        return;
    }

    double run = (op == C_CUMPROD) ? 1 : (op == C_CUMMAX) ? -INFINITY
               : (op == C_CUMMIN) ? INFINITY : 0;
    for (size_t j = 0; j < count; j++) {
        double v = in[base + j * inner];
        switch (op) {
        case C_CUMSUM:  run += v; break;
        case C_CUMPROD: run *= v; break;
        case C_CUMMAX:  if (v > run) run = v; break;
        default:        if (v < run) run = v; break;
        }
        out_lane[j] = (float)run;
    }
}

static void one_case(int op, int iter) {
    int nd = rng_int(1, MAXD);
    int shape[MAXD];
    for (int i = 0; i < nd; i++) shape[i] = rng_int(1, 4);
    int axis = rng_int(0, nd - 1);

    Tensor* a = mk(shape, nd, iter + 1);
    if (!a) { failures++; return; }
    /* prod over a long axis overflows f32 precision; keep the lane short */
    if (op == C_CUMPROD || op == R_PROD) {
        if (shape[axis] > 3) { cml_reset_ir_context(); return; }
    }

    const float* in = (const float*)tensor_data_ptr(a);
    float saved[256];
    size_t n_in = a->numel;
    if (n_in > 256) { cml_reset_ir_context(); return; }
    memcpy(saved, in, n_in * sizeof(float));

    Tensor* r = build(op, a, axis);
    checks++;
    if (!r) {
        printf("  %-11s iter %d: NULL (rank %d axis %d)\n", ax_name(op), iter, nd, axis);
        failures++;
        cml_reset_ir_context();
        return;
    }
    tensor_ensure_executed(r);

    size_t outer = 1, inner = 1, count = (size_t)shape[axis];
    for (int i = 0; i < axis; i++) outer *= (size_t)shape[i];
    for (int i = axis + 1; i < nd; i++) inner *= (size_t)shape[i];

    /* Property 1: shape. Reductions drop the axis (keepdim=false); the others keep it. */
    int want_nd = is_reduction(op) ? (nd == 1 ? 1 : nd - 1) : nd;
    int want[MAXD]; int w = 0;
    if (is_reduction(op)) {
        if (nd == 1) { want[w++] = 1; }
        else for (int i = 0; i < nd; i++) if (i != axis) want[w++] = shape[i];
    } else {
        for (int i = 0; i < nd; i++) want[w++] = shape[i];
    }
    int bad = (r->ndim != want_nd);
    for (int i = 0; !bad && i < want_nd; i++) if (r->shape[i] != want[i]) bad = 1;
    if (bad) {
        printf("  %-11s iter %d: shape [", ax_name(op), iter);
        for (int i = 0; i < r->ndim; i++) printf("%d%s", r->shape[i], i + 1 < r->ndim ? "," : "");
        printf("], expected [");
        for (int i = 0; i < want_nd; i++) printf("%d%s", want[i], i + 1 < want_nd ? "," : "");
        printf("]   input [");
        for (int i = 0; i < nd; i++) printf("%d%s", shape[i], i + 1 < nd ? "," : "");
        printf("] axis %d\n", axis);
        failures++;
        cml_reset_ir_context();
        return;
    }

    /* Property 2: values, lane by lane. */
    float lane[64];
    for (size_t o = 0; o < outer && !bad; o++) {
        for (size_t m = 0; m < inner; m++) {
            size_t base = o * count * inner + m;
            ref_lane(op, saved, base, inner, count, lane);
            size_t nout = is_reduction(op) ? 1 : count;
            for (size_t j = 0; j < nout; j++) {
                size_t oi = is_reduction(op) ? (o * inner + m) : (base + j * inner);
                if (oi >= r->numel) continue;
                float got = tensor_get_float(r, oi);
                float tol = fabsf(lane[j]) * 1e-4f + 1e-4f;
                if (fabsf(got - lane[j]) > tol) {
                    printf("  %-11s iter %d: [%zu] = %g, expected %g   shape [",
                           ax_name(op), iter, oi, got, lane[j]);
                    for (int i = 0; i < nd; i++) printf("%d%s", shape[i], i + 1 < nd ? "," : "");
                    printf("] axis %d\n", axis);
                    failures++;
                    bad = 1;
                    break;
                }
            }
            if (bad) break;
        }
    }
    cml_reset_ir_context();
}

int main(void) {
    cml_init();
    printf("Axis ops over random shapes and axes (%d ops):\n", NUM_AX_OPS);

    const int ITERS = 60;
    for (int op = 0; op < NUM_AX_OPS && failures < 15; op++)
        for (int i = 0; i < ITERS && failures < 15; i++) one_case(op, i);

    printf("\n%d cases, %d failures\n", checks, failures);
    cml_cleanup();
    return failures ? 1 : 0;
}
