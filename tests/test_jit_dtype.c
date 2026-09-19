/*
 * JIT typed codegen: the LLVM backend now emits native float64 kernels for
 * elementwise binary/unary ops (previously it deferred all non-f32 to the
 * interpreter). Invokes cml_llvm_execute directly so the JIT path is exercised;
 * f64-precision values (1e8+1) confirm the kernel is genuinely double, not f32.
 */
#include <stdio.h>
#include <math.h>

#include "tensor/tensor.h"
#include "test_harness.h"
#include "ops/uops.h"
#include "ops/ir/context.h"
#include "ops/ir/llvm/llvm_backend.h"
#include <string.h>

#ifdef CML_HAS_LLVM_BACKEND
static int check(const char* name, int ok) {
    tests_run++;
    if (ok) {
        tests_passed++;
        printf("  PASS: %s\n", name);
    } else {
        printf("  FAIL: %s\n", name);
    }
    return ok;
}

static const TensorConfig cfg_f64 = {
    .dtype = DTYPE_FLOAT64, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
static const TensorConfig cfg_f32 = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
static const TensorConfig cfg_i32 = {
    .dtype = DTYPE_INT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
static const TensorConfig cfg_i64 = {
    .dtype = DTYPE_INT64, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
#endif /* CML_HAS_LLVM_BACKEND */

int main(void) {
    printf("=== JIT typed codegen (f64) ===\n");
#ifndef CML_HAS_LLVM_BACKEND
    printf("  SKIP: built without LLVM backend\n");
    return 0;
#else

    CMLLLVMBackend* be = cml_llvm_backend_init();
    if (!be) {
        printf("  SKIP: LLVM backend unavailable\n");
        return 0;
    }

    /* f64 binary add through the JIT */
    double av[] = {1.5, 2.5, 1e8};
    double bv[] = {0.5, 0.5, 1.0};
    Tensor* a   = tensor_from_data(av, (int[]){3}, 1, &cfg_f64);
    Tensor* b   = tensor_from_data(bv, (int[]){3}, 1, &cfg_f64);
    Tensor* c   = uop_add(a, b); /* lazy: builds an IR node, not executed */
    Tensor* m   = uop_mul(a, b);
    Tensor* n   = uop_neg(a); /* f64 unary */

    CMLGraph_t ir = cml_ir_get_or_create_context();
    int rc        = cml_llvm_execute(be, ir);

    int ok = rc == 0 && c->data && c->dtype == DTYPE_FLOAT64;
    if (ok) {
        const double* cd = (const double*)c->data;
        const double* md = (const double*)m->data;
        const double* nd = (const double*)n->data;
        ok = cd[0] == 2.0 && cd[1] == 3.0 && cd[2] == 100000001.0 && /* f64, not f32 */
             fabs(md[0] - 0.75) < 1e-12 && fabs(md[1] - 1.25) < 1e-12 && nd[0] == -1.5 &&
             nd[2] == -1e8;
    }
    check("jit_f64_elementwise", ok);

    /* Sanity: f32 still works through the JIT (regression). */
    float xf[] = {1.0f, 2.0f, 3.0f};
    float yf[] = {4.0f, 5.0f, 6.0f};
    Tensor* x  = tensor_from_data(xf, (int[]){3}, 1, &cfg_f32);
    Tensor* y  = tensor_from_data(yf, (int[]){3}, 1, &cfg_f32);
    Tensor* z  = uop_add(x, y);
    cml_llvm_execute(be, cml_ir_get_or_create_context());
    const float* zd = (const float*)z->data;
    check("jit_f32_still_works",
          z->dtype == DTYPE_FLOAT32 && zd && zd[0] == 5.0f && zd[1] == 7.0f && zd[2] == 9.0f);

    /* JIT per-axis reduction: [2,3] f32, sum over each axis. */
    float rv[]       = {1, 2, 3, 4, 5, 6}; /* [[1,2,3],[4,5,6]] */
    Tensor* rt       = tensor_from_data(rv, (int[]){2, 3}, 2, &cfg_f32);
    int d1           = 1;
    ReduceParams rp1 = {&d1, 1, false};
    Tensor* s1t      = uop_sum(rt, &rp1); /* -> [6, 15] */
    int d0           = 0;
    ReduceParams rp0 = {&d0, 1, false};
    Tensor* s0t      = uop_sum(rt, &rp0);                            /* -> [5, 7, 9] */
    Tensor* gsum     = uop_sum(rt, &(ReduceParams){NULL, 0, false}); /* global -> 21 */
    cml_llvm_execute(be, cml_ir_get_or_create_context());
    int rok = s1t->data && s0t->data && gsum->data;
    if (rok) {
        const float* a1 = (const float*)s1t->data;
        const float* a0 = (const float*)s0t->data;
        rok = s1t->numel == 2 && a1[0] == 6 && a1[1] == 15 && s0t->numel == 3 && a0[0] == 5 &&
              a0[1] == 7 && a0[2] == 9 && ((const float*)gsum->data)[0] == 21.0f;
    }
    check("jit_reduction_axis_and_global", rok);

    /* JIT gather (cross-entropy shape): input [3,4], indices [3], gather last dim. */
    float gin[]  = {0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23}; /* [3,4] */
    float gidx[] = {2, 0, 3};                                    /* pick col 2,0,3 */
    Tensor* gt   = tensor_from_data(gin, (int[]){3, 4}, 2, &cfg_f32);
    Tensor* it   = tensor_from_data(gidx, (int[]){3}, 1, &cfg_f32);
    Tensor* go   = uop_gather(gt, it, -1); /* -> [2, 10, 23] */
    cml_llvm_execute(be, cml_ir_get_or_create_context());
    int gok = go && go->data && go->numel == 3;
    if (gok) {
        const float* gd = (const float*)go->data;
        gok             = gd[0] == 2.0f && gd[1] == 10.0f && gd[2] == 23.0f;
    }
    check("jit_gather_2d", gok);

    /* JIT integer codegen: i32/i64 add/mul/neg with exact integer results. */
    int32_t ia[] = {16777216, 3, -5}; /* 2^24 (not exact in f32) */
    int32_t ib[] = {1, 4, 5};
    Tensor* i0   = tensor_from_data(ia, (int[]){3}, 1, &cfg_i32);
    Tensor* i1   = tensor_from_data(ib, (int[]){3}, 1, &cfg_i32);
    Tensor* isum = uop_add(i0, i1); /* [16777217, 7, 0] */
    Tensor* ineg = uop_neg(i0);     /* [-16777216, -3, 5] */
    int64_t la[] = {1000000, 7};
    int64_t lb[] = {1000000, 8};
    Tensor* l0   = tensor_from_data(la, (int[]){2}, 1, &cfg_i64);
    Tensor* l1   = tensor_from_data(lb, (int[]){2}, 1, &cfg_i64);
    Tensor* lmul = uop_mul(l0, l1); /* [1e12, 56] */
    cml_llvm_execute(be, cml_ir_get_or_create_context());
    int iok = isum->data && ineg->data && lmul->data;
    if (iok) {
        const int32_t* s  = (const int32_t*)isum->data;
        const int32_t* n2 = (const int32_t*)ineg->data;
        const int64_t* m  = (const int64_t*)lmul->data;
        iok = s[0] == 16777217 && s[1] == 7 && s[2] == 0 && n2[0] == -16777216 && n2[2] == 5 &&
              m[0] == 1000000000000LL && m[1] == 56;
    }
    check("jit_integer_codegen", iok);

    cml_llvm_backend_destroy(be);
    return TEST_SUMMARY();
#endif /* CML_HAS_LLVM_BACKEND */
}
