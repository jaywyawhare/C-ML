/* The three-operand ops, over random broadcast shapes.
 *
 * uops.h has exactly two elementwise three-tensor entry points -- uop_where and
 * uop_lerp -- and nothing exercised either with operands of differing shapes.
 * Three-way broadcasting is strictly harder than two-way: `where` had its
 * condition indexed separately from its two value operands, and lerp indexed
 * all three with `i % numel`, so a stretched dimension picked the wrong element
 * from whichever operand needed stretching.
 *
 * Same two properties as the binary fuzzer: the output shape follows the numpy
 * rule across all three operands, and every value matches a stride-based
 * reference computed here rather than by library code.
 */

#include "cml.h"
#include "test_require.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static int checks = 0, failures = 0;

static uint64_t rng_state = 0xD1B54A32D192ED03ULL;
static uint64_t rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
static int rng_int(int lo, int hi) { return lo + (int)(rng_next() % (uint64_t)(hi - lo + 1)); }

#define MAXD 4

static void bcast_strides(const int* shape, int ndim, const int* osh, int ond, size_t* st) {
    size_t acc = 1, own[MAXD];
    for (int i = ndim - 1; i >= 0; i--) {
        own[i] = acc;
        acc *= (size_t)shape[i];
    }
    for (int i = 0; i < ond; i++)
        st[i] = 0;
    for (int k = 1; k <= ndim; k++) {
        int si = ndim - k, oi = ond - k;
        st[oi] = (shape[si] == 1 && osh[oi] != 1) ? 0 : own[si];
    }
}

static size_t ref_index(const size_t* st, const int* osh, int ond, size_t flat) {
    size_t idx = 0;
    for (int i = ond - 1; i >= 0; i--) {
        size_t coord = flat % (size_t)osh[i];
        flat /= (size_t)osh[i];
        idx += coord * st[i];
    }
    return idx;
}

static Tensor* make(const int* shape, int ndim, int seed, int lo, int hi) {
    TensorConfig c = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* t = tensor_zeros((int*)shape, ndim, &c);
    if (!t)
        return NULL;
    for (size_t i = 0; i < t->numel; i++)
        tensor_set_float(t, i,
                         (float)(lo + (int)((i * 3 + (size_t)seed * 7) % (size_t)(hi - lo + 1))));
    return t;
}

/* opsel 0 = where(cond, a, b), 1 = lerp(a, b, t) */
static const char* op_label(int o) {
    return o == 0 ? "where(composed)" : o == 1 ? "lerp" : "UOP_WHERE(direct)";
}

static void one_case(int iter, int opsel) {
    int nd[3], sh[3][MAXD];
    for (int k = 0; k < 3; k++) {
        nd[k] = rng_int(1, MAXD);
        for (int i = 0; i < nd[k]; i++)
            sh[k][i] = rng_int(0, 2) ? rng_int(1, 3) : 1;
    }

    /* numpy rule over all three, right-aligned. */
    int ond = nd[0] > nd[1] ? nd[0] : nd[1];
    if (nd[2] > ond)
        ond = nd[2];
    int osh[MAXD];
    for (int k = 1; k <= ond; k++) {
        int mx = 1;
        for (int t = 0; t < 3; t++) {
            int idx = nd[t] - k;
            int v   = (idx >= 0) ? sh[t][idx] : 1;
            if (v != 1 && mx != 1 && v != mx)
                return; /* incompatible: skip */
            if (v > mx)
                mx = v;
        }
        osh[ond - k] = mx;
    }

    /* where's condition is a mask; lerp's t is a weight in [0,1]. */
    int cond_like = (opsel != 1);
    Tensor* t0    = make(sh[0], nd[0], iter + 1, cond_like ? 0 : -4, cond_like ? 1 : 4);
    Tensor* t1    = make(sh[1], nd[1], iter + 2, -4, 4);
    Tensor* t2    = make(sh[2], nd[2], iter + 3, cond_like ? -4 : 0, cond_like ? 4 : 1);
    if (!t0 || !t1 || !t2) {
        failures++;
        cml_reset_ir_context();
        return;
    }

    Tensor* r;
    if (opsel == 0) {
        WhereParams wp = {.cond = t0, .a = t1, .b = t2};
        r              = uop_where(&wp);
    } else if (opsel == 1) {
        r = uop_lerp(t0, t1, t2);
    } else {
        /* uop_where composes as cond*a + (1-cond)*b, so it never emits a
         * UOP_WHERE node -- yet decompose *does*, when lowering ops to
         * primitives, so that kernel is live and otherwise untested. Build the
         * node directly to reach it. */
        CMLGraph_t ir  = cml_ir_get_or_create_context();
        Tensor* ins[3] = {t0, t1, t2};
        if (!ir || cml_ir_add_uop(ir, UOP_WHERE, ins, 3, NULL) != 0) {
            cml_reset_ir_context();
            return;
        }
        struct IRNode* node = cml_ir_get_tail(ir);
        if (cml_ir_compute_broadcast_shape(node) != 0) {
            cml_reset_ir_context();
            return;
        }
        r = tensor_from_ir_node(node, ir);
    }
    checks++;
    if (!r) {
        printf("  %s iter %d: returned NULL\n", op_label(opsel), iter);
        failures++;
        cml_reset_ir_context();
        return;
    }
    tensor_ensure_executed(r);

    int bad = (r->ndim != ond);
    for (int i = 0; !bad && i < ond; i++)
        if (r->shape[i] != osh[i])
            bad = 1;
    if (bad) {
        printf("  %s iter %d: shape [", op_label(opsel), iter);
        for (int i = 0; i < r->ndim; i++)
            printf("%d%s", r->shape[i], i + 1 < r->ndim ? "," : "");
        printf("], expected [");
        for (int i = 0; i < ond; i++)
            printf("%d%s", osh[i], i + 1 < ond ? "," : "");
        printf("]   operands [");
        for (int k = 0; k < 3; k++) {
            for (int i = 0; i < nd[k]; i++)
                printf("%d%s", sh[k][i], i + 1 < nd[k] ? "," : "");
            printf(k < 2 ? "] [" : "]\n");
        }
        failures++;
        cml_reset_ir_context();
        return;
    }

    size_t st[3][MAXD];
    for (int k = 0; k < 3; k++)
        bcast_strides(sh[k], nd[k], osh, ond, st[k]);
    Tensor* ops[3] = {t0, t1, t2};
    for (size_t i = 0; i < r->numel; i++) {
        float v[3];
        for (int k = 0; k < 3; k++)
            v[k] = tensor_get_float(ops[k], ref_index(st[k], osh, ond, i));
        float want = (opsel == 1) ? (v[0] + v[2] * (v[1] - v[0])) : (v[0] != 0.0f ? v[1] : v[2]);
        float got  = tensor_get_float(r, i);
        if (fabsf(got - want) > fabsf(want) * 1e-5f + 1e-5f) {
            printf("  %s iter %d: [%zu] = %g, expected %g   operands [", op_label(opsel), iter, i,
                   got, want);
            for (int k = 0; k < 3; k++) {
                for (int j = 0; j < nd[k]; j++)
                    printf("%d%s", sh[k][j], j + 1 < nd[k] ? "," : "");
                printf(k < 2 ? "] [" : "]");
            }
            printf(" out=[");
            for (int j = 0; j < ond; j++)
                printf("%d%s", osh[j], j + 1 < ond ? "," : "");
            printf("]\n");
            failures++;
            break;
        }
    }
    cml_reset_ir_context();
}

int main(void) {
    cml_init();
    printf("Three-operand ops over random broadcast shapes:\n");

    const int ITERS = 200;
    for (int op = 0; op < 3 && failures < 10; op++)
        for (int i = 0; i < ITERS && failures < 10; i++)
            one_case(i, op);

    printf("\n%d cases, %d failures\n", checks, failures);
    cml_cleanup();
    return failures ? 1 : 0;
}
