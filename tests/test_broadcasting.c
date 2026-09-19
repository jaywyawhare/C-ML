/* Numpy broadcasting must hold on every execution path.
 *
 * Several kernels indexed a broadcast operand with `i % operand_numel`. That is
 * only the right mapping when the operand tiles the output's *trailing* dims
 * (a bias [N] over [M,N]). Whenever a dimension has to stretch it is wrong:
 * for [2,3] op [2,1] it walks the operand 0,1,0,1,0,1 when it must be
 * 0,0,0,1,1,1. So `x - mean(x, dim=1, keepdim=True)` -- the shape at the centre
 * of layernorm, softmax and per-row normalisation -- silently mixed rows
 * together instead of subtracting each row's own mean.
 *
 * It survived because it is path-dependent: the interpreter's _broadcast_idx is
 * shape-aware and was always right, the fusion pass explicitly refuses to fuse
 * these operands, but the per-node LLVM JIT (on by default) and both non-f32
 * paths used the modulo. Anything checking only same-shape operands, or only
 * trailing broadcasts, passes regardless -- so this test uses leading-dim and
 * interior-dim broadcasts specifically, across dtypes.
 *
 * Run it with and without JIT=0: they are different emitters.
 */

#include "cml.h"
#include "test_require.h"
#include <stdio.h>

static int checks = 0, failures = 0;

static Tensor* make(DType dt, const int* shape, int ndim, const float* v) {
    TensorConfig c = {.dtype = dt, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* t      = tensor_zeros((int*)shape, ndim, &c);
    if (t && v)
        for (size_t i = 0; i < t->numel; i++)
            tensor_set_float(t, i, v[i]);
    return t;
}

static void expect(const char* name, DType dt, Tensor* r, const float* want, int n) {
    checks++;
    const char* dn = (dt == DTYPE_FLOAT32) ? "f32" : (dt == DTYPE_FLOAT64) ? "f64" : "i32";
    if (!r) {
        printf("  %-20s %s built NULL\n", name, dn);
        failures++;
        return;
    }
    tensor_ensure_executed(r);
    if ((int)r->numel != n) {
        printf("  %-20s %s numel %zu, expected %d\n", name, dn, r->numel, n);
        failures++;
        cml_reset_ir_context();
        return;
    }
    for (int i = 0; i < n; i++) {
        if (tensor_get_float(r, (size_t)i) != want[i]) {
            printf("  %-20s %s [%d] = %g, expected %g\n", name, dn, i,
                   tensor_get_float(r, (size_t)i), want[i]);
            failures++;
            break;
        }
    }
    cml_reset_ir_context();
}

int main(void) {
    cml_init();
    printf("Broadcasting:\n");

    const DType dtypes[3] = {DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_INT32};

    for (int k = 0; k < 3; k++) {
        DType dt = dtypes[k];

        /* Leading-dim broadcast -- the case that was wrong. */
        {
            const int as[2] = {2, 3}, bs[2] = {2, 1};
            const float av[6] = {1, 2, 3, 4, 5, 6}, bv[2] = {10, 20};
            const float want[6] = {11, 12, 13, 24, 25, 26};
            expect("[2,3] + [2,1]", dt, uop_add(make(dt, as, 2, av), make(dt, bs, 2, bv)), want, 6);
        }
        {
            const int as[2] = {2, 3}, bs[2] = {2, 1};
            const float av[6] = {1, 2, 3, 4, 5, 6}, bv[2] = {2, 3};
            const float want[6] = {2, 4, 6, 12, 15, 18};
            expect("[2,3] * [2,1]", dt, uop_mul(make(dt, as, 2, av), make(dt, bs, 2, bv)), want, 6);
        }
        /* Interior-dim broadcast in 3-D. */
        {
            const int as[3] = {2, 2, 2}, bs[3] = {2, 1, 2};
            const float av[8] = {1, 2, 3, 4, 5, 6, 7, 8}, bv[4] = {10, 20, 30, 40};
            const float want[8] = {11, 22, 13, 24, 35, 46, 37, 48};
            expect("[2,2,2] + [2,1,2]", dt, uop_add(make(dt, as, 3, av), make(dt, bs, 3, bv)), want,
                   8);
        }
        /* Trailing broadcast and scalar: these were always right, keep them so. */
        {
            const int as[2] = {2, 3}, bs[1] = {3};
            const float av[6] = {1, 2, 3, 4, 5, 6}, bv[3] = {10, 20, 30};
            const float want[6] = {11, 22, 33, 14, 25, 36};
            expect("[2,3] + [3]", dt, uop_add(make(dt, as, 2, av), make(dt, bs, 1, bv)), want, 6);
        }
        {
            const int as[2] = {2, 3}, bs[1] = {1};
            const float av[6] = {1, 2, 3, 4, 5, 6}, bv[1] = {100};
            const float want[6] = {101, 102, 103, 104, 105, 106};
            expect("[2,3] + [1]", dt, uop_add(make(dt, as, 2, av), make(dt, bs, 1, bv)), want, 6);
        }
    }

    /* Outer product shape: both operands stretch. */
    {
        const int as[2] = {3, 1}, bs[2] = {1, 4};
        const float av[3] = {1, 2, 3}, bv[4] = {10, 20, 30, 40};
        const float want[12] = {11, 21, 31, 41, 12, 22, 32, 42, 13, 23, 33, 43};
        expect("[3,1] + [1,4]", DTYPE_FLOAT32,
               uop_add(make(DTYPE_FLOAT32, as, 2, av), make(DTYPE_FLOAT32, bs, 2, bv)), want, 12);
    }

    printf("\n%d checks, %d failures\n", checks, failures);
    cml_cleanup();
    return failures ? 1 : 0;
}
