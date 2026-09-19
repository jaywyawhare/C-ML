/* Randomised broadcasting: shapes and values against an independent reference.
 *
 * The existing op fuzzer randomises shapes but gives both operands the *same*
 * shape, so it never exercises broadcasting at all. The broadcasting bug found
 * earlier (operands indexed with `i % numel`, which is only correct when the
 * operand tiles the output's trailing dims) needed a leading-dim broadcast such
 * as [2,3] op [2,1] to show up, and it was found by hand-picking that case.
 * Hand-picked cases only cover what someone thought of.
 *
 * This generates random broadcast pairs -- random rank, random dims, random
 * choice of which dims are stretched and whether an operand has fewer dims --
 * and checks two independent properties per case:
 *
 *   1. the output SHAPE follows the numpy rule (right-aligned, each dim the max
 *      of the two, a 1 stretches, a mismatch is an error);
 *   2. every output VALUE equals a straight shape-aware reference computed here,
 *      not by any library code.
 *
 * Values are small integers, so a mismatch is a logic error and never rounding.
 */

#include "cml.h"
#include "test_require.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static int checks = 0, failures = 0;

static uint64_t rng_state = 0x2545F4914F6CDD1DULL;
static uint64_t rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
static int rng_int(int lo, int hi) { /* [lo, hi] */
    return lo + (int)(rng_next() % (uint64_t)(hi - lo + 1));
}

#define MAXD 4

/* Row-major strides, with 0 for a stretched dim so indexing repeats it. */
static void bcast_strides(const int* shape, int ndim, const int* out_shape, int out_ndim,
                          size_t* strides) {
    size_t acc = 1;
    size_t own[MAXD];
    for (int i = ndim - 1; i >= 0; i--) {
        own[i] = acc;
        acc *= (size_t)shape[i];
    }
    for (int i = 0; i < out_ndim; i++)
        strides[i] = 0;
    for (int k = 1; k <= ndim; k++) {
        int si = ndim - k, oi = out_ndim - k;
        strides[oi] = (shape[si] == 1 && out_shape[oi] != 1) ? 0 : own[si];
    }
}

static size_t ref_index(const size_t* strides, const int* out_shape, int out_ndim, size_t flat) {
    size_t idx = 0;
    for (int i = out_ndim - 1; i >= 0; i--) {
        size_t coord = flat % (size_t)out_shape[i];
        flat /= (size_t)out_shape[i];
        idx += coord * strides[i];
    }
    return idx;
}

static int g_lo = -5, g_hi = 5;

static Tensor* make(const int* shape, int ndim, int seed) {
    TensorConfig c = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* t = tensor_zeros((int*)shape, ndim, &c);
    if (!t)
        return NULL;
    for (size_t i = 0; i < t->numel; i++)
        tensor_set_float(
            t, i,
            (float)(g_lo + (int)(((size_t)i * 3 + (size_t)seed * 7) % (size_t)(g_hi - g_lo + 1))));
    return t;
}

/* Every op that takes two tensors and broadcasts. matmul and masked_select are
 * excluded: they are not elementwise. */
typedef enum {
    BOP_ADD,
    BOP_SUB,
    BOP_MUL,
    BOP_DIV,
    BOP_MAX,
    BOP_MIN,
    BOP_POW,
    BOP_CMPLT,
    BOP_CMPGT,
    BOP_CMPLE,
    BOP_CMPGE,
    BOP_CMPEQ,
    BOP_CMPNE,
    BOP_COPYSIGN,
    BOP_LOGADDEXP,
    BOP_LOGICAL_AND,
    BOP_LOGICAL_OR,
    BOP_IDIV,
    BOP_MOD,
    BOP_AND,
    BOP_OR,
    BOP_XOR,
    BOP_SHL,
    BOP_SHR,
    NUM_OPS
} OpSel;

static const char* op_name(int o) {
    static const char* n[NUM_OPS] = {
        "add",   "sub",         "mul",        "div",         "max",        "minimum",
        "pow",   "cmplt",       "cmpgt",      "cmple",       "cmpge",      "cmpeq",
        "cmpne", "copysign",    "logaddexp",  "logical_and", "logical_or", "idiv",
        "mod",   "bitwise_and", "bitwise_or", "bitwise_xor", "lshift",     "rshift"};
    return n[o];
}

static Tensor* build_op(int o, Tensor* a, Tensor* b) {
    switch (o) {
    case BOP_ADD:
        return uop_add(a, b);
    case BOP_SUB:
        return uop_sub(a, b);
    case BOP_MUL:
        return uop_mul(a, b);
    case BOP_DIV:
        return uop_div(a, b);
    case BOP_MAX:
        return uop_max(a, b);
    case BOP_MIN:
        return uop_minimum(a, b);
    case BOP_POW:
        return uop_pow(a, b);
    case BOP_CMPLT:
        return uop_cmplt(a, b);
    case BOP_CMPGT:
        return uop_cmpgt(a, b);
    case BOP_CMPLE:
        return uop_cmple(a, b);
    case BOP_CMPGE:
        return uop_cmpge(a, b);
    case BOP_CMPEQ:
        return uop_cmpeq(a, b);
    case BOP_CMPNE:
        return uop_cmpne(a, b);
    case BOP_COPYSIGN:
        return uop_copysign(a, b);
    case BOP_LOGADDEXP:
        return uop_logaddexp(a, b);
    case BOP_LOGICAL_AND:
        return uop_logical_and(a, b);
    case BOP_LOGICAL_OR:
        return uop_logical_or(a, b);
    case BOP_IDIV:
        return uop_idiv(a, b);
    case BOP_MOD:
        return uop_mod(a, b);
    case BOP_AND:
        return uop_bitwise_and(a, b);
    case BOP_OR:
        return uop_bitwise_or(a, b);
    case BOP_XOR:
        return uop_bitwise_xor(a, b);
    case BOP_SHL:
        return uop_lshift(a, b);
    case BOP_SHR:
        return uop_rshift(a, b);
    default:
        return NULL;
    }
}

static float ref_op(int o, float x, float y) {
    switch (o) {
    case BOP_ADD:
        return x + y;
    case BOP_SUB:
        return x - y;
    case BOP_MUL:
        return x * y;
    case BOP_DIV:
        return x / y;
    case BOP_MAX:
        return x > y ? x : y;
    case BOP_MIN:
        return x < y ? x : y;
    case BOP_POW:
        return powf(x, y);
    case BOP_CMPLT:
        return x < y ? 1.0f : 0.0f;
    case BOP_CMPGT:
        return x > y ? 1.0f : 0.0f;
    case BOP_CMPLE:
        return x <= y ? 1.0f : 0.0f;
    case BOP_CMPGE:
        return x >= y ? 1.0f : 0.0f;
    case BOP_CMPEQ:
        return x == y ? 1.0f : 0.0f;
    case BOP_CMPNE:
        return x != y ? 1.0f : 0.0f;
    case BOP_COPYSIGN:
        return copysignf(x, y);
    case BOP_LOGADDEXP: {
        float m = x > y ? x : y;
        return m + logf(expf(x - m) + expf(y - m));
    }
    case BOP_LOGICAL_AND:
        return (x != 0.0f && y != 0.0f) ? 1.0f : 0.0f;
    case BOP_LOGICAL_OR:
        return (x != 0.0f || y != 0.0f) ? 1.0f : 0.0f;
    case BOP_IDIV:
        return floorf(x / y); /* documented as floor(a/b) */
    case BOP_MOD:
        return fmodf(x, y);
    case BOP_AND:
        return (float)((int32_t)x & (int32_t)y);
    case BOP_OR:
        return (float)((int32_t)x | (int32_t)y);
    case BOP_XOR:
        return (float)((int32_t)x ^ (int32_t)y);
    case BOP_SHL:
        return (float)((int32_t)x << (int32_t)y);
    case BOP_SHR:
        return (float)((int32_t)x >> (int32_t)y);
    default:
        return 0.0f;
    }
}

/* Operand ranges chosen so each op is well defined: no division by zero, no
 * negative base for pow, small non-negative shift counts. */
static void op_ranges(int o, int* alo, int* ahi, int* blo, int* bhi) {
    *alo = -5;
    *ahi = 5;
    *blo = -5;
    *bhi = 5;
    switch (o) {
    case BOP_DIV:
    case BOP_MOD:
    case BOP_IDIV:
        *blo = 1;
        *bhi = 5;
        break;
    case BOP_POW:
        *alo = 1;
        *ahi = 4;
        *blo = 0;
        *bhi = 3;
        break;
    case BOP_SHL:
    case BOP_SHR:
        *alo = 0;
        *ahi = 7;
        *blo = 0;
        *bhi = 3;
        break;
    case BOP_AND:
    case BOP_OR:
    case BOP_XOR:
        *alo = 0;
        *ahi = 15;
        *blo = 0;
        *bhi = 15;
        break;
    case BOP_LOGADDEXP:
        *alo = -3;
        *ahi = 3;
        *blo = -3;
        *bhi = 3;
        break;
    default:
        break;
    }
}

static void one_case(int iter, int opsel) {
    /* Generate the two operand shapes first and derive the expected output by
     * the numpy rule. (Working backwards from a chosen output shape is wrong:
     * if both operands have fewer dims, the real result has lower rank.) */
    int and_ = rng_int(1, MAXD), bnd = rng_int(1, MAXD);
    int ash[MAXD], bsh[MAXD];
    for (int i = 0; i < and_; i++)
        ash[i] = rng_int(0, 2) ? rng_int(1, 4) : 1;
    for (int i = 0; i < bnd; i++)
        bsh[i] = rng_int(0, 2) ? rng_int(1, 4) : 1;

    /* numpy rule, right-aligned: dims must be equal or one of them 1. */
    int ond = and_ > bnd ? and_ : bnd;
    int osh[MAXD];
    for (int k = 1; k <= ond; k++) {
        int ai = and_ - k, bi = bnd - k;
        int av = (ai >= 0) ? ash[ai] : 1;
        int bv = (bi >= 0) ? bsh[bi] : 1;
        if (av != bv && av != 1 && bv != 1)
            return; /* incompatible: skip */
        osh[ond - k] = av > bv ? av : bv;
    }

    int alo, ahi, blo, bhi;
    op_ranges(opsel, &alo, &ahi, &blo, &bhi);
    g_lo      = alo;
    g_hi      = ahi;
    Tensor* a = make(ash, and_, iter + 1);
    g_lo      = blo;
    g_hi      = bhi;
    Tensor* b = make(bsh, bnd, iter + 2);
    if (!a || !b) {
        failures++;
        cml_reset_ir_context();
        return;
    }

    Tensor* r = build_op(opsel, a, b);
    checks++;
    if (!r) {
        printf("  iter %d op%d: returned NULL for a rank-%d/rank-%d pair\n", iter, opsel, and_,
               bnd);
        failures++;
        cml_reset_ir_context();
        return;
    }
    tensor_ensure_executed(r);

    /* Property 1: shape follows the numpy rule. */
    int bad_shape = (r->ndim != ond);
    for (int i = 0; !bad_shape && i < ond; i++)
        if (r->shape[i] != osh[i])
            bad_shape = 1;
    if (bad_shape) {
        printf("  %s iter %d: shape [", op_name(opsel), iter);
        for (int i = 0; i < r->ndim; i++)
            printf("%d%s", r->shape[i], i + 1 < r->ndim ? "," : "");
        printf("], expected [");
        for (int i = 0; i < ond; i++)
            printf("%d%s", osh[i], i + 1 < ond ? "," : "");
        printf("]   a=[");
        for (int k = 0; k < and_; k++)
            printf("%d%s", ash[k], k + 1 < and_ ? "," : "");
        printf("] b=[");
        for (int k = 0; k < bnd; k++)
            printf("%d%s", bsh[k], k + 1 < bnd ? "," : "");
        printf("]\n");
        failures++;
        cml_reset_ir_context();
        return;
    }

    /* Property 2: values match a reference computed here. */
    size_t as[MAXD], bs[MAXD];
    bcast_strides(ash, and_, osh, ond, as);
    bcast_strides(bsh, bnd, osh, ond, bs);
    for (size_t i = 0; i < r->numel; i++) {
        float av   = tensor_get_float(a, ref_index(as, osh, ond, i));
        float bv   = tensor_get_float(b, ref_index(bs, osh, ond, i));
        float want = ref_op(opsel, av, bv);
        float got  = tensor_get_float(r, i);
        float tol  = (opsel == BOP_POW || opsel == BOP_LOGADDEXP || opsel == BOP_DIV)
                         ? (fabsf(want) * 1e-5f + 1e-5f)
                         : 0.0f;
        if (fabsf(got - want) > tol) {
            printf("  %s iter %d: [%zu] = %g, expected %g   a=[", op_name(opsel), iter, i, got,
                   want);
            for (int k = 0; k < and_; k++)
                printf("%d%s", ash[k], k + 1 < and_ ? "," : "");
            printf("] b=[");
            for (int k = 0; k < bnd; k++)
                printf("%d%s", bsh[k], k + 1 < bnd ? "," : "");
            printf("] out=[");
            for (int k = 0; k < ond; k++)
                printf("%d%s", osh[k], k + 1 < ond ? "," : "");
            printf("]\n");
            failures++;
            break;
        }
    }
    cml_reset_ir_context();
}

int main(void) {
    cml_init();
    printf("Randomised broadcasting (shape + value vs an independent reference):\n");

    /* EVERY broadcast-capable binary op, not a sample. These do NOT share one
     * implementation: each had its own loop, and covering add/mul left max,
     * cmplt, the bitwise trio, pow, masked_fill and lerp broken. Sampling a
     * few ops proves nothing about the rest. */
    const int ITERS = 120;
    for (int op = 0; op < NUM_OPS && failures < 12; op++)
        for (int i = 0; i < ITERS && failures < 12; i++)
            one_case(i, op);

    printf("\n%d cases, %d failures\n", checks, failures);
    cml_cleanup();
    return failures ? 1 : 0;
}
