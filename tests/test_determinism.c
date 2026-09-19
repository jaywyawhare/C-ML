/* A fixed seed must reproduce a run exactly.
 *
 * cml_manual_seed() seeds the library's counter-based RNG (CMLRNGState), but the
 * lazy materialisers behind uop_rand_uniform/normal/int drew from libc rand(),
 * which that seed never touches. Seeding therefore had no effect on any of those
 * ops: two runs with the same seed produced different numbers, so a "seeded"
 * experiment was not reproducible. Layer initialisation already used the real
 * RNG, which is why the breakage was invisible to anything that only checked
 * weight init.
 */

#include "cml.h"
#include "test_require.h"
#include <stdio.h>
#include <string.h>

static int checks = 0, failures = 0;

#define N 64

static void draw(uint64_t seed, float* out, int kind) {
    cml_manual_seed(seed);
    int shape[1] = {N};
    Tensor* t    = (kind == 0)   ? uop_rand_uniform(shape, 1, DTYPE_FLOAT32, DEVICE_CPU)
                   : (kind == 1) ? uop_rand_normal(shape, 1, DTYPE_FLOAT32, DEVICE_CPU)
                                 : uop_rand_int(0, 100, shape, 1, DTYPE_FLOAT32, DEVICE_CPU);
    tensor_ensure_executed(t);
    for (int i = 0; i < N; i++)
        out[i] = tensor_get_float(t, (size_t)i);
    cml_reset_ir_context();
}

static int identical(const float* a, const float* b) {
    for (int i = 0; i < N; i++)
        if (a[i] != b[i])
            return 0;
    return 1;
}

static void check_kind(const char* name, int kind) {
    float a[N], b[N], c[N];
    draw(1234, a, kind);
    draw(1234, b, kind);
    draw(9999, c, kind);

    checks++;
    if (!identical(a, b)) {
        printf("  %-8s same seed produced different values (seeding has no effect)\n", name);
        failures++;
    }
    /* A generator that ignores its seed could still pass the check above by
     * being constant, so require different seeds to diverge. */
    checks++;
    if (identical(a, c)) {
        printf("  %-8s different seeds produced identical values\n", name);
        failures++;
    }
}

int main(void) {
    cml_init();
    printf("Seed determinism:\n");

    check_kind("uniform", 0);
    check_kind("normal", 1);
    check_kind("randint", 2);

    /* Layer init draws from the same global RNG and must also be reproducible. */
    {
        float w[2][16];
        int got[2] = {0, 0};
        for (int pass = 0; pass < 2; pass++) {
            cml_manual_seed(777);
            Linear* l      = cml_nn_linear(4, 4, DTYPE_FLOAT32, DEVICE_CPU, false);
            Parameter** ps = NULL;
            int np         = 0;
            if (l && module_collect_parameters((Module*)l, &ps, &np, NULL) == 0 && ps && np > 0 &&
                ps[0] && ps[0]->tensor) {
                Tensor* t = ps[0]->tensor;
                tensor_ensure_executed(t);
                int n = (int)(t->numel < 16 ? t->numel : 16);
                for (int i = 0; i < n; i++)
                    w[pass][i] = tensor_get_float(t, (size_t)i);
                got[pass] = n;
            }
            cml_reset_ir_context();
        }
        checks++;
        if (got[0] == 0 || got[0] != got[1]) {
            printf("  layerinit could not read parameters\n");
            failures++;
        } else {
            for (int i = 0; i < got[0]; i++)
                if (w[0][i] != w[1][i]) {
                    printf("  layerinit same seed produced different weights\n");
                    failures++;
                    break;
                }
        }
    }

    printf("\n%d checks, %d failures\n", checks, failures);
    cml_cleanup();
    return failures ? 1 : 0;
}
