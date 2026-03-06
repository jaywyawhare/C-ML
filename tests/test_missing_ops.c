/**
 * @file test_missing_ops.c
 * @brief Tests for sort, argsort, topk, cumprod, bitwise ops, nonzero, masked_fill
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tensor/tensor.h"
#include "ops/uops.h"

static int tests_passed = 0;
static int tests_total  = 0;

static const TensorConfig cpu_f32 = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

#define TEST(name)                                     \
    do {                                               \
        tests_total++;                                 \
        printf("  TEST: %s ... ", #name);              \
        if (test_##name()) {                           \
            tests_passed++;                            \
            printf("PASSED\n");                        \
        } else {                                       \
            printf("FAILED\n");                        \
        }                                              \
    } while (0)

#define APPROX(a, b) (fabsf((a) - (b)) < 1e-5f)

static int test_sort_1d(void) {
    float data[] = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 2.0f};
    int shape[] = {6};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* sorted = uop_sort(t, 0, false);
    tensor_ensure_executed(sorted);
    float* out = sorted->data;
    if (!APPROX(out[0], 1.0f)) return 0;
    if (!APPROX(out[1], 1.0f)) return 0;
    if (!APPROX(out[2], 2.0f)) return 0;
    if (!APPROX(out[3], 3.0f)) return 0;
    if (!APPROX(out[4], 4.0f)) return 0;
    if (!APPROX(out[5], 5.0f)) return 0;
    tensor_free(t);
    tensor_free(sorted);
    return 1;
}

static int test_sort_descending(void) {
    float data[] = {3.0f, 1.0f, 4.0f, 2.0f};
    int shape[] = {4};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* sorted = uop_sort(t, 0, true);
    tensor_ensure_executed(sorted);
    float* out = sorted->data;
    if (!APPROX(out[0], 4.0f)) return 0;
    if (!APPROX(out[1], 3.0f)) return 0;
    if (!APPROX(out[2], 2.0f)) return 0;
    if (!APPROX(out[3], 1.0f)) return 0;
    tensor_free(t);
    tensor_free(sorted);
    return 1;
}

static int test_argsort_1d(void) {
    float data[] = {3.0f, 1.0f, 4.0f, 2.0f};
    int shape[] = {4};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* indices = uop_argsort(t, 0, false);
    tensor_ensure_executed(indices);
    float* out = indices->data;
    // sorted order: 1.0(idx1), 2.0(idx3), 3.0(idx0), 4.0(idx2)
    if (!APPROX(out[0], 1.0f)) return 0;
    if (!APPROX(out[1], 3.0f)) return 0;
    if (!APPROX(out[2], 0.0f)) return 0;
    if (!APPROX(out[3], 2.0f)) return 0;
    tensor_free(t);
    tensor_free(indices);
    return 1;
}

static int test_topk(void) {
    float data[] = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f};
    int shape[] = {6};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* idx_out = NULL;
    Tensor* values = uop_topk(t, 3, 0, true, &idx_out);
    tensor_ensure_executed(values);
    float* out = values->data;
    // top 3 largest: 9, 5, 4
    if (!APPROX(out[0], 9.0f)) return 0;
    if (!APPROX(out[1], 5.0f)) return 0;
    if (!APPROX(out[2], 4.0f)) return 0;
    tensor_free(t);
    tensor_free(values);
    if (idx_out) tensor_free(idx_out);
    return 1;
}

static int test_cumprod(void) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    int shape[] = {4};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* result = uop_cumprod(t, 0);
    tensor_ensure_executed(result);
    float* out = result->data;
    // cumprod: 1, 1*2=2, 2*3=6, 6*4=24
    if (!APPROX(out[0], 1.0f)) return 0;
    if (!APPROX(out[1], 2.0f)) return 0;
    if (!APPROX(out[2], 6.0f)) return 0;
    if (!APPROX(out[3], 24.0f)) return 0;
    tensor_free(t);
    tensor_free(result);
    return 1;
}

static int test_bitwise_and(void) {
    float data_a[] = {7.0f, 5.0f, 3.0f, 15.0f};
    float data_b[] = {3.0f, 6.0f, 1.0f, 10.0f};
    int shape[] = {4};
    Tensor* a = tensor_from_data(data_a, shape, 1, &cpu_f32);
    Tensor* b = tensor_from_data(data_b, shape, 1, &cpu_f32);
    Tensor* result = uop_bitwise_and(a, b);
    tensor_ensure_executed(result);
    float* out = result->data;
    // 7&3=3, 5&6=4, 3&1=1, 15&10=10
    if (!APPROX(out[0], 3.0f)) return 0;
    if (!APPROX(out[1], 4.0f)) return 0;
    if (!APPROX(out[2], 1.0f)) return 0;
    if (!APPROX(out[3], 10.0f)) return 0;
    tensor_free(a);
    tensor_free(b);
    tensor_free(result);
    return 1;
}

static int test_bitwise_or(void) {
    float data_a[] = {5.0f, 3.0f};
    float data_b[] = {3.0f, 6.0f};
    int shape[] = {2};
    Tensor* a = tensor_from_data(data_a, shape, 1, &cpu_f32);
    Tensor* b = tensor_from_data(data_b, shape, 1, &cpu_f32);
    Tensor* result = uop_bitwise_or(a, b);
    tensor_ensure_executed(result);
    float* out = result->data;
    // 5|3=7, 3|6=7
    if (!APPROX(out[0], 7.0f)) return 0;
    if (!APPROX(out[1], 7.0f)) return 0;
    tensor_free(a);
    tensor_free(b);
    tensor_free(result);
    return 1;
}

static int test_bitwise_xor(void) {
    float data_a[] = {5.0f, 3.0f};
    float data_b[] = {3.0f, 6.0f};
    int shape[] = {2};
    Tensor* a = tensor_from_data(data_a, shape, 1, &cpu_f32);
    Tensor* b = tensor_from_data(data_b, shape, 1, &cpu_f32);
    Tensor* result = uop_bitwise_xor(a, b);
    tensor_ensure_executed(result);
    float* out = result->data;
    // 5^3=6, 3^6=5
    if (!APPROX(out[0], 6.0f)) return 0;
    if (!APPROX(out[1], 5.0f)) return 0;
    tensor_free(a);
    tensor_free(b);
    tensor_free(result);
    return 1;
}

static int test_bitwise_not(void) {
    float data[] = {0.0f, 5.0f};
    int shape[] = {2};
    Tensor* a = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* result = uop_bitwise_not(a);
    tensor_ensure_executed(result);
    float* out = result->data;
    // ~0 = -1, ~5 = -6
    if (!APPROX(out[0], -1.0f)) return 0;
    if (!APPROX(out[1], -6.0f)) return 0;
    tensor_free(a);
    tensor_free(result);
    return 1;
}

static int test_nonzero(void) {
    float data[] = {0.0f, 3.0f, 0.0f, 5.0f, 7.0f, 0.0f};
    int shape[] = {6};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* result = uop_nonzero(t);
    tensor_ensure_executed(result);
    float* out = result->data;
    // nonzero indices: 1, 3, 4
    if (result->shape[0] < 3) return 0;
    if (!APPROX(out[0], 1.0f)) return 0;
    if (!APPROX(out[1], 3.0f)) return 0;
    if (!APPROX(out[2], 4.0f)) return 0;
    tensor_free(t);
    tensor_free(result);
    return 1;
}

static int test_log10(void) {
    float data[] = {1.0f, 10.0f, 100.0f, 1000.0f};
    int shape[] = {4};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* r = uop_log10(t);
    tensor_ensure_executed(r);
    float* o = r->data;
    if (!APPROX(o[0], 0.0f)) return 0;
    if (!APPROX(o[1], 1.0f)) return 0;
    if (!APPROX(o[2], 2.0f)) return 0;
    if (!APPROX(o[3], 3.0f)) return 0;
    tensor_free(t); tensor_free(r);
    return 1;
}

static int test_sinh_cosh(void) {
    float data[] = {0.0f, 1.0f};
    int shape[] = {2};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* s = uop_sinh(t);
    Tensor* c = uop_cosh(t);
    tensor_ensure_executed(s);
    tensor_ensure_executed(c);
    if (!APPROX(((float*)s->data)[0], 0.0f)) return 0;
    if (!APPROX(((float*)s->data)[1], sinhf(1.0f))) return 0;
    if (!APPROX(((float*)c->data)[0], 1.0f)) return 0;
    if (!APPROX(((float*)c->data)[1], coshf(1.0f))) return 0;
    tensor_free(t); tensor_free(s); tensor_free(c);
    return 1;
}

static int test_asinh_acosh_atanh(void) {
    float data[] = {0.5f};
    int shape[] = {1};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* as = uop_asinh(t);
    Tensor* at = uop_atanh(t);
    tensor_ensure_executed(as);
    tensor_ensure_executed(at);
    if (!APPROX(((float*)as->data)[0], asinhf(0.5f))) return 0;
    if (!APPROX(((float*)at->data)[0], atanhf(0.5f))) return 0;
    tensor_free(t); tensor_free(as); tensor_free(at);
    // acosh requires input >= 1
    float data2[] = {2.0f};
    Tensor* t2 = tensor_from_data(data2, shape, 1, &cpu_f32);
    Tensor* ac = uop_acosh(t2);
    tensor_ensure_executed(ac);
    if (!APPROX(((float*)ac->data)[0], acoshf(2.0f))) return 0;
    tensor_free(t2); tensor_free(ac);
    return 1;
}

static int test_trunc(void) {
    float data[] = {1.7f, -2.3f, 3.0f, -0.9f};
    int shape[] = {4};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* r = uop_trunc(t);
    tensor_ensure_executed(r);
    float* o = r->data;
    if (!APPROX(o[0], 1.0f)) return 0;
    if (!APPROX(o[1], -2.0f)) return 0;
    if (!APPROX(o[2], 3.0f)) return 0;
    if (!APPROX(o[3], 0.0f)) return 0;
    tensor_free(t); tensor_free(r);
    return 1;
}

static int test_isinf_isnan_isfinite(void) {
    float inf = 1.0f / 0.0f;
    float nan_val = 0.0f / 0.0f;
    float data[] = {1.0f, inf, nan_val, -inf};
    int shape[] = {4};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* ri = uop_isinf(t);
    Tensor* rn = uop_isnan(t);
    Tensor* rf = uop_isfinite(t);
    tensor_ensure_executed(ri);
    tensor_ensure_executed(rn);
    tensor_ensure_executed(rf);
    float* oi = ri->data;
    float* on = rn->data;
    float* of_ = rf->data;
    // isinf: 0,1,0,1
    if (!APPROX(oi[0], 0.0f)) return 0;
    if (!APPROX(oi[1], 1.0f)) return 0;
    if (!APPROX(oi[2], 0.0f)) return 0;
    if (!APPROX(oi[3], 1.0f)) return 0;
    // isnan: 0,0,1,0
    if (!APPROX(on[0], 0.0f)) return 0;
    if (!APPROX(on[1], 0.0f)) return 0;
    if (!APPROX(on[2], 1.0f)) return 0;
    if (!APPROX(on[3], 0.0f)) return 0;
    // isfinite: 1,0,0,0
    if (!APPROX(of_[0], 1.0f)) return 0;
    if (!APPROX(of_[1], 0.0f)) return 0;
    if (!APPROX(of_[2], 0.0f)) return 0;
    if (!APPROX(of_[3], 0.0f)) return 0;
    tensor_free(t); tensor_free(ri); tensor_free(rn); tensor_free(rf);
    return 1;
}

static int test_logical_not(void) {
    float data[] = {0.0f, 1.0f, -2.0f, 0.0f};
    int shape[] = {4};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* r = uop_logical_not(t);
    tensor_ensure_executed(r);
    float* o = r->data;
    if (!APPROX(o[0], 1.0f)) return 0;
    if (!APPROX(o[1], 0.0f)) return 0;
    if (!APPROX(o[2], 0.0f)) return 0;
    if (!APPROX(o[3], 1.0f)) return 0;
    tensor_free(t); tensor_free(r);
    return 1;
}

static int test_idiv(void) {
    float data_a[] = {7.0f, -7.0f, 10.0f};
    float data_b[] = {2.0f, 2.0f, 3.0f};
    int shape[] = {3};
    Tensor* a = tensor_from_data(data_a, shape, 1, &cpu_f32);
    Tensor* b = tensor_from_data(data_b, shape, 1, &cpu_f32);
    Tensor* r = uop_idiv(a, b);
    tensor_ensure_executed(r);
    float* o = r->data;
    if (!APPROX(o[0], 3.0f)) return 0;
    if (!APPROX(o[1], -4.0f)) return 0;
    if (!APPROX(o[2], 3.0f)) return 0;
    tensor_free(a); tensor_free(b); tensor_free(r);
    return 1;
}

static int test_mod(void) {
    float data_a[] = {7.0f, 10.0f, 5.5f};
    float data_b[] = {3.0f, 4.0f, 2.0f};
    int shape[] = {3};
    Tensor* a = tensor_from_data(data_a, shape, 1, &cpu_f32);
    Tensor* b = tensor_from_data(data_b, shape, 1, &cpu_f32);
    Tensor* r = uop_mod(a, b);
    tensor_ensure_executed(r);
    float* o = r->data;
    if (!APPROX(o[0], 1.0f)) return 0;
    if (!APPROX(o[1], 2.0f)) return 0;
    if (!APPROX(o[2], 1.5f)) return 0;
    tensor_free(a); tensor_free(b); tensor_free(r);
    return 1;
}

static int test_minimum(void) {
    float data_a[] = {3.0f, 1.0f, 4.0f};
    float data_b[] = {2.0f, 5.0f, 1.0f};
    int shape[] = {3};
    Tensor* a = tensor_from_data(data_a, shape, 1, &cpu_f32);
    Tensor* b = tensor_from_data(data_b, shape, 1, &cpu_f32);
    Tensor* r = uop_minimum(a, b);
    tensor_ensure_executed(r);
    float* o = r->data;
    if (!APPROX(o[0], 2.0f)) return 0;
    if (!APPROX(o[1], 1.0f)) return 0;
    if (!APPROX(o[2], 1.0f)) return 0;
    tensor_free(a); tensor_free(b); tensor_free(r);
    return 1;
}

static int test_logaddexp(void) {
    float data_a[] = {1.0f, 2.0f};
    float data_b[] = {3.0f, 4.0f};
    int shape[] = {2};
    Tensor* a = tensor_from_data(data_a, shape, 1, &cpu_f32);
    Tensor* b = tensor_from_data(data_b, shape, 1, &cpu_f32);
    Tensor* r = uop_logaddexp(a, b);
    tensor_ensure_executed(r);
    float* o = r->data;
    float expected0 = logf(expf(1.0f) + expf(3.0f));
    float expected1 = logf(expf(2.0f) + expf(4.0f));
    if (!APPROX(o[0], expected0)) return 0;
    if (!APPROX(o[1], expected1)) return 0;
    tensor_free(a); tensor_free(b); tensor_free(r);
    return 1;
}

static int test_logical_and_or(void) {
    float data_a[] = {1.0f, 0.0f, 1.0f, 0.0f};
    float data_b[] = {1.0f, 1.0f, 0.0f, 0.0f};
    int shape[] = {4};
    Tensor* a = tensor_from_data(data_a, shape, 1, &cpu_f32);
    Tensor* b = tensor_from_data(data_b, shape, 1, &cpu_f32);
    Tensor* ra = uop_logical_and(a, b);
    Tensor* ro = uop_logical_or(a, b);
    tensor_ensure_executed(ra);
    tensor_ensure_executed(ro);
    float* oa = ra->data;
    float* oo = ro->data;
    // AND: 1,0,0,0
    if (!APPROX(oa[0], 1.0f) || !APPROX(oa[1], 0.0f) || !APPROX(oa[2], 0.0f) || !APPROX(oa[3], 0.0f)) return 0;
    // OR: 1,1,1,0
    if (!APPROX(oo[0], 1.0f) || !APPROX(oo[1], 1.0f) || !APPROX(oo[2], 1.0f) || !APPROX(oo[3], 0.0f)) return 0;
    tensor_free(a); tensor_free(b); tensor_free(ra); tensor_free(ro);
    return 1;
}

static int test_comparisons(void) {
    float data_a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float data_b[] = {2.0f, 2.0f, 1.0f, 4.0f};
    int shape[] = {4};
    Tensor* a = tensor_from_data(data_a, shape, 1, &cpu_f32);
    Tensor* b = tensor_from_data(data_b, shape, 1, &cpu_f32);

    Tensor* eq = uop_cmpeq(a, b);
    Tensor* ne = uop_cmpne(a, b);
    Tensor* le = uop_cmple(a, b);
    Tensor* gt = uop_cmpgt(a, b);
    Tensor* ge = uop_cmpge(a, b);
    tensor_ensure_executed(eq);
    tensor_ensure_executed(ne);
    tensor_ensure_executed(le);
    tensor_ensure_executed(gt);
    tensor_ensure_executed(ge);

    float *oeq=eq->data, *one=ne->data, *ole=le->data, *ogt=gt->data, *oge=ge->data;
    // eq: 0,1,0,1
    if (!APPROX(oeq[0],0) || !APPROX(oeq[1],1) || !APPROX(oeq[2],0) || !APPROX(oeq[3],1)) return 0;
    // ne: 1,0,1,0
    if (!APPROX(one[0],1) || !APPROX(one[1],0) || !APPROX(one[2],1) || !APPROX(one[3],0)) return 0;
    // le: 1,1,0,1
    if (!APPROX(ole[0],1) || !APPROX(ole[1],1) || !APPROX(ole[2],0) || !APPROX(ole[3],1)) return 0;
    // gt: 0,0,1,0
    if (!APPROX(ogt[0],0) || !APPROX(ogt[1],0) || !APPROX(ogt[2],1) || !APPROX(ogt[3],0)) return 0;
    // ge: 0,1,1,1
    if (!APPROX(oge[0],0) || !APPROX(oge[1],1) || !APPROX(oge[2],1) || !APPROX(oge[3],1)) return 0;

    tensor_free(a); tensor_free(b);
    tensor_free(eq); tensor_free(ne); tensor_free(le); tensor_free(gt); tensor_free(ge);
    return 1;
}

static int test_min_reduce(void) {
    float data[] = {3.0f, 1.0f, 4.0f, 2.0f};
    int shape[] = {4};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* r = uop_min_reduce(t, NULL);
    tensor_ensure_executed(r);
    if (!APPROX(((float*)r->data)[0], 1.0f)) return 0;
    tensor_free(t); tensor_free(r);
    return 1;
}

static int test_var_std(void) {
    float data[] = {2.0f, 4.0f, 4.0f, 4.0f, 5.0f, 5.0f, 7.0f, 9.0f};
    int shape[] = {8};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* v = uop_var(t, NULL);
    Tensor* s = uop_std(t, NULL);
    tensor_ensure_executed(v);
    tensor_ensure_executed(s);
    // mean = 5.0, var = 4.0, std = 2.0
    if (!APPROX(((float*)v->data)[0], 4.0f)) return 0;
    if (!APPROX(((float*)s->data)[0], 2.0f)) return 0;
    tensor_free(t); tensor_free(v); tensor_free(s);
    return 1;
}

static int test_any_all(void) {
    float data1[] = {0.0f, 0.0f, 1.0f, 0.0f};
    float data2[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float data3[] = {0.0f, 0.0f, 0.0f, 0.0f};
    int shape[] = {4};
    Tensor* t1 = tensor_from_data(data1, shape, 1, &cpu_f32);
    Tensor* t2 = tensor_from_data(data2, shape, 1, &cpu_f32);
    Tensor* t3 = tensor_from_data(data3, shape, 1, &cpu_f32);

    Tensor* any1 = uop_any(t1, NULL);
    Tensor* all1 = uop_all(t1, NULL);
    Tensor* any2 = uop_any(t2, NULL);
    Tensor* all2 = uop_all(t2, NULL);
    Tensor* any3 = uop_any(t3, NULL);
    Tensor* all3 = uop_all(t3, NULL);

    tensor_ensure_executed(any1); tensor_ensure_executed(all1);
    tensor_ensure_executed(any2); tensor_ensure_executed(all2);
    tensor_ensure_executed(any3); tensor_ensure_executed(all3);

    if (!APPROX(((float*)any1->data)[0], 1.0f)) return 0; // has non-zero
    if (!APPROX(((float*)all1->data)[0], 0.0f)) return 0; // not all non-zero
    if (!APPROX(((float*)any2->data)[0], 1.0f)) return 0;
    if (!APPROX(((float*)all2->data)[0], 1.0f)) return 0; // all non-zero
    if (!APPROX(((float*)any3->data)[0], 0.0f)) return 0; // no non-zero
    if (!APPROX(((float*)all3->data)[0], 0.0f)) return 0;

    tensor_free(t1); tensor_free(t2); tensor_free(t3);
    tensor_free(any1); tensor_free(all1); tensor_free(any2);
    tensor_free(all2); tensor_free(any3); tensor_free(all3);
    return 1;
}

static int test_logsumexp(void) {
    float data[] = {1.0f, 2.0f, 3.0f};
    int shape[] = {3};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* r = uop_logsumexp(t, NULL);
    tensor_ensure_executed(r);
    float expected = logf(expf(1.0f) + expf(2.0f) + expf(3.0f));
    if (!APPROX(((float*)r->data)[0], expected)) return 0;
    tensor_free(t); tensor_free(r);
    return 1;
}

static int test_cummax_cummin(void) {
    float data[] = {3.0f, 1.0f, 4.0f, 2.0f, 5.0f};
    int shape[] = {5};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* mx = uop_cummax(t, 0);
    Tensor* mn = uop_cummin(t, 0);
    tensor_ensure_executed(mx);
    tensor_ensure_executed(mn);
    float* omx = mx->data;
    float* omn = mn->data;
    // cummax: 3,3,4,4,5
    if (!APPROX(omx[0],3) || !APPROX(omx[1],3) || !APPROX(omx[2],4) || !APPROX(omx[3],4) || !APPROX(omx[4],5)) return 0;
    // cummin: 3,1,1,1,1
    if (!APPROX(omn[0],3) || !APPROX(omn[1],1) || !APPROX(omn[2],1) || !APPROX(omn[3],1) || !APPROX(omn[4],1)) return 0;
    tensor_free(t); tensor_free(mx); tensor_free(mn);
    return 1;
}

static int test_cat(void) {
    float data_a[] = {1.0f, 2.0f, 3.0f};
    float data_b[] = {4.0f, 5.0f};
    int sha[] = {3}, shb[] = {2};
    Tensor* a = tensor_from_data(data_a, sha, 1, &cpu_f32);
    Tensor* b = tensor_from_data(data_b, shb, 1, &cpu_f32);
    Tensor* tensors[] = {a, b};
    Tensor* r = uop_cat(tensors, 2, 0);
    tensor_ensure_executed(r);
    if (r->shape[0] != 5) return 0;
    float* o = r->data;
    if (!APPROX(o[0],1) || !APPROX(o[1],2) || !APPROX(o[2],3) || !APPROX(o[3],4) || !APPROX(o[4],5)) return 0;
    tensor_free(a); tensor_free(b); tensor_free(r);
    return 1;
}

static int test_stack(void) {
    float d1[] = {1.0f, 2.0f};
    float d2[] = {3.0f, 4.0f};
    int sh[] = {2};
    Tensor* t1 = tensor_from_data(d1, sh, 1, &cpu_f32);
    Tensor* t2 = tensor_from_data(d2, sh, 1, &cpu_f32);
    Tensor* tensors[] = {t1, t2};
    Tensor* r = uop_stack(tensors, 2, 0);
    tensor_ensure_executed(r);
    if (r->ndim != 2 || r->shape[0] != 2 || r->shape[1] != 2) return 0;
    float* o = r->data;
    // [[1,2],[3,4]]
    if (!APPROX(o[0],1) || !APPROX(o[1],2) || !APPROX(o[2],3) || !APPROX(o[3],4)) return 0;
    tensor_free(t1); tensor_free(t2); tensor_free(r);
    return 1;
}

static int test_roll(void) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    int shape[] = {5};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* r = uop_roll(t, 2, 0);
    tensor_ensure_executed(r);
    float* o = r->data;
    // roll by 2: [4,5,1,2,3]
    if (!APPROX(o[0],4) || !APPROX(o[1],5) || !APPROX(o[2],1) || !APPROX(o[3],2) || !APPROX(o[4],3)) return 0;
    tensor_free(t); tensor_free(r);
    return 1;
}

static int test_flatten(void) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    int shape[] = {2, 3};
    Tensor* t = tensor_from_data(data, shape, 2, &cpu_f32);
    Tensor* r = uop_flatten(t, 0, 1);
    tensor_ensure_executed(r);
    if (r->ndim != 1 || r->shape[0] != 6) return 0;
    float* o = r->data;
    for (int i = 0; i < 6; i++)
        if (!APPROX(o[i], (float)(i + 1))) return 0;
    tensor_free(t); tensor_free(r);
    return 1;
}

static int test_diag(void) {
    // 1D -> 2D diagonal matrix
    float data[] = {1.0f, 2.0f, 3.0f};
    int shape[] = {3};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* r = uop_diag(t, 0);
    tensor_ensure_executed(r);
    if (r->ndim != 2 || r->shape[0] != 3 || r->shape[1] != 3) return 0;
    float* o = r->data;
    // [[1,0,0],[0,2,0],[0,0,3]]
    if (!APPROX(o[0],1) || !APPROX(o[1],0) || !APPROX(o[2],0)) return 0;
    if (!APPROX(o[3],0) || !APPROX(o[4],2) || !APPROX(o[5],0)) return 0;
    if (!APPROX(o[6],0) || !APPROX(o[7],0) || !APPROX(o[8],3)) return 0;
    tensor_free(t); tensor_free(r);
    return 1;
}

static int test_one_hot(void) {
    float data[] = {0.0f, 2.0f, 1.0f};
    int shape[] = {3};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* r = uop_one_hot(t, 3);
    tensor_ensure_executed(r);
    if (r->ndim != 2 || r->shape[0] != 3 || r->shape[1] != 3) return 0;
    float* o = r->data;
    // [[1,0,0],[0,0,1],[0,1,0]]
    if (!APPROX(o[0],1) || !APPROX(o[1],0) || !APPROX(o[2],0)) return 0;
    if (!APPROX(o[3],0) || !APPROX(o[4],0) || !APPROX(o[5],1)) return 0;
    if (!APPROX(o[6],0) || !APPROX(o[7],1) || !APPROX(o[8],0)) return 0;
    tensor_free(t); tensor_free(r);
    return 1;
}

static int test_masked_fill(void) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float mask_data[] = {1.0f, 0.0f, 1.0f, 0.0f};
    int shape[] = {4};
    Tensor* a = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* mask = tensor_from_data(mask_data, shape, 1, &cpu_f32);
    Tensor* result = uop_masked_fill(a, mask, -999.0f);
    tensor_ensure_executed(result);
    float* out = result->data;
    // where mask=1: fill with -999, where mask=0: keep original
    if (!APPROX(out[0], -999.0f)) return 0;
    if (!APPROX(out[1], 2.0f)) return 0;
    if (!APPROX(out[2], -999.0f)) return 0;
    if (!APPROX(out[3], 4.0f)) return 0;
    tensor_free(a);
    tensor_free(mask);
    tensor_free(result);
    return 1;
}

static int test_erfc(void) {
    float data[] = {0.0f, 0.5f, 1.0f, 2.0f};
    int shape[] = {4};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* r = uop_erfc(t);
    tensor_ensure_executed(r);
    float* out = r->data;
    if (!APPROX(out[0], 1.0f)) return 0;      // erfc(0) = 1
    if (!APPROX(out[2], erfcf(1.0f))) return 0; // erfc(1) ~= 0.1573
    tensor_free(t); tensor_free(r);
    return 1;
}

static int test_lerp(void) {
    float d1[] = {0.0f, 2.0f, 4.0f};
    float d2[] = {10.0f, 20.0f, 40.0f};
    float dt[] = {0.5f, 0.5f, 0.25f};
    int shape[] = {3};
    Tensor* a = tensor_from_data(d1, shape, 1, &cpu_f32);
    Tensor* b = tensor_from_data(d2, shape, 1, &cpu_f32);
    Tensor* t = tensor_from_data(dt, shape, 1, &cpu_f32);
    Tensor* r = uop_lerp(a, b, t);
    tensor_ensure_executed(r);
    float* out = r->data;
    if (!APPROX(out[0], 5.0f)) return 0;   // 0 + 0.5*(10-0) = 5
    if (!APPROX(out[1], 11.0f)) return 0;  // 2 + 0.5*(20-2) = 11
    if (!APPROX(out[2], 13.0f)) return 0;  // 4 + 0.25*(40-4) = 13
    tensor_free(a); tensor_free(b); tensor_free(t); tensor_free(r);
    return 1;
}

static int test_tile(void) {
    float data[] = {1.0f, 2.0f, 3.0f};
    int shape[] = {3};
    int reps[] = {3};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* r = uop_tile(t, reps, 1);
    tensor_ensure_executed(r);
    float* out = r->data;
    if (r->shape[0] != 9) return 0;
    if (!APPROX(out[0], 1.0f)) return 0;
    if (!APPROX(out[3], 1.0f)) return 0;
    if (!APPROX(out[6], 1.0f)) return 0;
    if (!APPROX(out[8], 3.0f)) return 0;
    tensor_free(t); tensor_free(r);
    return 1;
}

static int test_repeat_interleave(void) {
    float data[] = {1.0f, 2.0f, 3.0f};
    int shape[] = {3};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* r = uop_repeat_interleave(t, 2, 0);
    tensor_ensure_executed(r);
    float* out = r->data;
    if (r->shape[0] != 6) return 0;
    if (!APPROX(out[0], 1.0f)) return 0;
    if (!APPROX(out[1], 1.0f)) return 0;
    if (!APPROX(out[2], 2.0f)) return 0;
    if (!APPROX(out[3], 2.0f)) return 0;
    if (!APPROX(out[4], 3.0f)) return 0;
    if (!APPROX(out[5], 3.0f)) return 0;
    tensor_free(t); tensor_free(r);
    return 1;
}

static int test_trace(void) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    int shape[] = {3, 3};
    Tensor* t = tensor_from_data(data, shape, 2, &cpu_f32);
    Tensor* r = uop_trace(t);
    tensor_ensure_executed(r);
    float* out = r->data;
    // trace = 1 + 5 + 9 = 15
    if (!APPROX(out[0], 15.0f)) return 0;
    tensor_free(t); tensor_free(r);
    return 1;
}

static int test_shrink(void) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    int shape[] = {2, 3};
    int starts[] = {0, 1};
    int ends[] = {2, 3};
    Tensor* t = tensor_from_data(data, shape, 2, &cpu_f32);
    Tensor* r = uop_shrink(t, starts, ends, 2);
    tensor_ensure_executed(r);
    float* out = r->data;
    if (r->shape[0] != 2 || r->shape[1] != 2) return 0;
    if (!APPROX(out[0], 2.0f)) return 0;  // [0,1]
    if (!APPROX(out[1], 3.0f)) return 0;  // [0,2]
    if (!APPROX(out[2], 5.0f)) return 0;  // [1,1]
    if (!APPROX(out[3], 6.0f)) return 0;  // [1,2]
    tensor_free(t); tensor_free(r);
    return 1;
}

static int test_logcumsumexp(void) {
    float data[] = {1.0f, 2.0f, 3.0f};
    int shape[] = {3};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* r = uop_logcumsumexp(t, 0);
    tensor_ensure_executed(r);
    float* out = r->data;
    // out[0] = 1.0 (just exp(1)/exp = log(exp(1)) = 1)
    if (!APPROX(out[0], 1.0f)) return 0;
    // out[1] = log(exp(1) + exp(2)) = 2 + log(exp(-1) + 1)
    float expected1 = logf(expf(1.0f) + expf(2.0f));
    if (!APPROX(out[1], expected1)) return 0;
    tensor_free(t); tensor_free(r);
    return 1;
}

static int test_relu6(void) {
    float data[] = {-1.0f, 0.0f, 3.0f, 7.0f, 10.0f};
    int shape[] = {5};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* r = uop_relu6(t);
    tensor_ensure_executed(r);
    float* out = r->data;
    if (!APPROX(out[0], 0.0f)) return 0;
    if (!APPROX(out[1], 0.0f)) return 0;
    if (!APPROX(out[2], 3.0f)) return 0;
    if (!APPROX(out[3], 6.0f)) return 0;
    if (!APPROX(out[4], 6.0f)) return 0;
    tensor_free(t); tensor_free(r);
    return 1;
}

static int test_hard_sigmoid(void) {
    float data[] = {-4.0f, 0.0f, 4.0f};
    int shape[] = {3};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* r = uop_hard_sigmoid(t);
    tensor_ensure_executed(r);
    float* out = r->data;
    if (!APPROX(out[0], 0.0f)) return 0;     // clamp((-4+3)/6, 0, 1) = 0
    if (!APPROX(out[1], 0.5f)) return 0;     // clamp((0+3)/6, 0, 1) = 0.5
    if (!APPROX(out[2], 1.0f)) return 0;     // clamp((4+3)/6, 0, 1) = 1
    tensor_free(t); tensor_free(r);
    return 1;
}

static int test_hard_tanh(void) {
    float data[] = {-2.0f, -0.5f, 0.5f, 2.0f};
    int shape[] = {4};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* r = uop_hard_tanh(t);
    tensor_ensure_executed(r);
    float* out = r->data;
    if (!APPROX(out[0], -1.0f)) return 0;
    if (!APPROX(out[1], -0.5f)) return 0;
    if (!APPROX(out[2], 0.5f)) return 0;
    if (!APPROX(out[3], 1.0f)) return 0;
    tensor_free(t); tensor_free(r);
    return 1;
}

static int test_celu(void) {
    float data[] = {-1.0f, 0.0f, 1.0f};
    int shape[] = {3};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* r = uop_celu(t, 1.0f);
    tensor_ensure_executed(r);
    float* out = r->data;
    // celu(-1, alpha=1) = max(0,-1) + min(0, 1*(exp(-1/1)-1)) = 0 + (exp(-1)-1) = -0.6321
    float expected = expf(-1.0f) - 1.0f;
    if (!APPROX(out[0], expected)) return 0;
    if (!APPROX(out[1], 0.0f)) return 0;
    if (!APPROX(out[2], 1.0f)) return 0;
    tensor_free(t); tensor_free(r);
    return 1;
}

static int test_quick_gelu(void) {
    float data[] = {0.0f, 1.0f, -1.0f};
    int shape[] = {3};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* r = uop_quick_gelu(t);
    tensor_ensure_executed(r);
    float* out = r->data;
    // quick_gelu(0) = 0 * sigmoid(0) = 0
    if (!APPROX(out[0], 0.0f)) return 0;
    // quick_gelu(1) = 1 * sigmoid(1.702) ~= 0.8455
    float expected_pos = 1.0f / (1.0f + expf(-1.702f));
    if (!APPROX(out[1], expected_pos)) return 0;
    tensor_free(t); tensor_free(r);
    return 1;
}

static int test_softplus(void) {
    float data[] = {0.0f, 1.0f, -1.0f, 10.0f};
    int shape[] = {4};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* r = uop_softplus(t);
    tensor_ensure_executed(r);
    float* out = r->data;
    if (!APPROX(out[0], logf(2.0f))) return 0;    // log(1 + exp(0)) = log(2)
    if (!APPROX(out[1], logf(1.0f + expf(1.0f)))) return 0;
    if (fabsf(out[3] - 10.0f) > 1e-4f) return 0;  // for large x, softplus(x) ~= x
    tensor_free(t); tensor_free(r);
    return 1;
}

static int test_softsign(void) {
    float data[] = {0.0f, 1.0f, -2.0f};
    int shape[] = {3};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* r = uop_softsign(t);
    tensor_ensure_executed(r);
    float* out = r->data;
    if (!APPROX(out[0], 0.0f)) return 0;
    if (!APPROX(out[1], 0.5f)) return 0;      // 1/(1+1) = 0.5
    if (!APPROX(out[2], -2.0f/3.0f)) return 0; // -2/(1+2) = -2/3
    tensor_free(t); tensor_free(r);
    return 1;
}

static int test_logsigmoid(void) {
    float data[] = {0.0f, 10.0f, -10.0f};
    int shape[] = {3};
    Tensor* t = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* r = uop_logsigmoid(t);
    tensor_ensure_executed(r);
    float* out = r->data;
    // logsigmoid(0) = log(0.5) = -log(2)
    if (!APPROX(out[0], -logf(2.0f))) return 0;
    // logsigmoid(10) ~= 0 (large positive)
    if (out[1] > -1e-3f && out[1] < 0.01f) { /* ok, close to 0 */ }
    else return 0;
    // logsigmoid(-10) ~= -10 (large negative)
    if (fabsf(out[2] - (-10.0f)) > 1e-4f) return 0;  // logsigmoid(-10) ~= -10
    tensor_free(t); tensor_free(r);
    return 1;
}

int main(void) {
    printf("=== Missing Ops Tests ===\n");
    TEST(sort_1d);
    TEST(sort_descending);
    TEST(argsort_1d);
    TEST(topk);
    TEST(cumprod);
    TEST(bitwise_and);
    TEST(bitwise_or);
    TEST(bitwise_xor);
    TEST(bitwise_not);
    TEST(nonzero);
    TEST(masked_fill);

    printf("\n=== New Unary Ops ===\n");
    TEST(log10);
    TEST(sinh_cosh);
    TEST(asinh_acosh_atanh);
    TEST(trunc);
    TEST(isinf_isnan_isfinite);
    TEST(logical_not);

    printf("\n=== New Binary Ops ===\n");
    TEST(idiv);
    TEST(mod);
    TEST(minimum);
    TEST(logaddexp);
    TEST(logical_and_or);

    printf("\n=== Comparison Ops ===\n");
    TEST(comparisons);

    printf("\n=== New Reduction Ops ===\n");
    TEST(min_reduce);
    TEST(var_std);
    TEST(any_all);
    TEST(logsumexp);
    TEST(cummax_cummin);

    printf("\n=== Movement Ops ===\n");
    TEST(cat);
    TEST(stack);
    TEST(roll);
    TEST(flatten);
    TEST(diag);
    TEST(one_hot);

    printf("\n=== Tinygrad Parity Round 2 ===\n");
    TEST(erfc);
    TEST(lerp);
    TEST(tile);
    TEST(repeat_interleave);
    TEST(trace);
    TEST(shrink);
    TEST(logcumsumexp);

    printf("\n=== Activation Ops ===\n");
    TEST(relu6);
    TEST(hard_sigmoid);
    TEST(hard_tanh);
    TEST(celu);
    TEST(quick_gelu);
    TEST(softplus);
    TEST(softsign);
    TEST(logsigmoid);

    printf("\nResults: %d/%d passed\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
