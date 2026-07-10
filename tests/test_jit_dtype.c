/*
 * JIT typed codegen: the LLVM backend now emits native float64 kernels for
 * elementwise binary/unary ops (previously it deferred all non-f32 to the
 * interpreter). Invokes cml_llvm_execute directly so the JIT path is exercised;
 * f64-precision values (1e8+1) confirm the kernel is genuinely double, not f32.
 */
#include <stdio.h>
#include <math.h>

#include "tensor/tensor.h"
#include "ops/uops.h"
#include "ops/ir/context.h"
#include "ops/ir/llvm/llvm_backend.h"
#include <string.h>

static int g_pass = 0, g_total = 0;
static int check(const char* name, int ok) {
    g_total++;
    if (ok) { g_pass++; printf("  PASS: %s\n", name); }
    else    { printf("  FAIL: %s\n", name); }
    return ok;
}

static const TensorConfig cfg_f64 = {.dtype = DTYPE_FLOAT64, .device = DEVICE_CPU,
                                     .has_dtype = true, .has_device = true};
static const TensorConfig cfg_f32 = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                                     .has_dtype = true, .has_device = true};

int main(void) {
    printf("=== JIT typed codegen (f64) ===\n");

    CMLLLVMBackend* be = cml_llvm_backend_init();
    if (!be) {
        printf("  SKIP: LLVM backend unavailable\n");
        return 0;
    }

    /* f64 binary add through the JIT */
    double av[] = {1.5, 2.5, 1e8};
    double bv[] = {0.5, 0.5, 1.0};
    Tensor* a = tensor_from_data(av, (int[]){3}, 1, &cfg_f64);
    Tensor* b = tensor_from_data(bv, (int[]){3}, 1, &cfg_f64);
    Tensor* c = uop_add(a, b);          /* lazy: builds an IR node, not executed */
    Tensor* m = uop_mul(a, b);
    Tensor* n = uop_neg(a);             /* f64 unary */

    CMLGraph_t ir = cml_ir_get_or_create_context();
    int rc = cml_llvm_execute(be, ir);

    int ok = rc == 0 && c->data && c->dtype == DTYPE_FLOAT64;
    if (ok) {
        const double* cd = (const double*)c->data;
        const double* md = (const double*)m->data;
        const double* nd = (const double*)n->data;
        ok = cd[0] == 2.0 && cd[1] == 3.0 && cd[2] == 100000001.0 &&  /* f64, not f32 */
             fabs(md[0] - 0.75) < 1e-12 && fabs(md[1] - 1.25) < 1e-12 &&
             nd[0] == -1.5 && nd[2] == -1e8;
    }
    check("jit_f64_elementwise", ok);

    /* Sanity: f32 still works through the JIT (regression). */
    float xf[] = {1.0f, 2.0f, 3.0f};
    float yf[] = {4.0f, 5.0f, 6.0f};
    Tensor* x = tensor_from_data(xf, (int[]){3}, 1, &cfg_f32);
    Tensor* y = tensor_from_data(yf, (int[]){3}, 1, &cfg_f32);
    Tensor* z = uop_add(x, y);
    cml_llvm_execute(be, cml_ir_get_or_create_context());
    const float* zd = (const float*)z->data;
    check("jit_f32_still_works",
          z->dtype == DTYPE_FLOAT32 && zd && zd[0] == 5.0f && zd[1] == 7.0f && zd[2] == 9.0f);

    /* JIT per-axis reduction: [2,3] f32, sum over each axis. */
    float rv[] = {1, 2, 3, 4, 5, 6};   /* [[1,2,3],[4,5,6]] */
    Tensor* rt   = tensor_from_data(rv, (int[]){2, 3}, 2, &cfg_f32);
    int d1 = 1; ReduceParams rp1 = {&d1, 1, false};
    Tensor* s1t  = uop_sum(rt, &rp1);   /* -> [6, 15] */
    int d0 = 0; ReduceParams rp0 = {&d0, 1, false};
    Tensor* s0t  = uop_sum(rt, &rp0);   /* -> [5, 7, 9] */
    Tensor* gsum = uop_sum(rt, &(ReduceParams){NULL, 0, false});  /* global -> 21 */
    cml_llvm_execute(be, cml_ir_get_or_create_context());
    int rok = s1t->data && s0t->data && gsum->data;
    if (rok) {
        const float* a1 = (const float*)s1t->data;
        const float* a0 = (const float*)s0t->data;
        rok = s1t->numel == 2 && a1[0] == 6 && a1[1] == 15 &&
              s0t->numel == 3 && a0[0] == 5 && a0[1] == 7 && a0[2] == 9 &&
              ((const float*)gsum->data)[0] == 21.0f;
    }
    check("jit_reduction_axis_and_global", rok);

    cml_llvm_backend_destroy(be);
    printf("\nResults: %d/%d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
