/*
 * Multi-dtype compute (increment 1): float64 and integer elementwise binary ops.
 *
 * The executor was formerly float32-only. These tests confirm the ops now run
 * in their native C type — verified by values that float32 CANNOT represent
 * (proving it isn't secretly going through the f32 path) — plus broadcasting.
 */
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "tensor/tensor.h"
#include "ops/uops.h"

static int g_pass = 0, g_total = 0;
static int check(const char* name, int ok) {
    g_total++;
    if (ok) { g_pass++; printf("  PASS: %s\n", name); }
    else    { printf("  FAIL: %s\n", name); }
    return ok;
}

static const TensorConfig cfg_f64 = {.dtype = DTYPE_FLOAT64, .device = DEVICE_CPU,
                                     .has_dtype = true, .has_device = true};
static const TensorConfig cfg_i32 = {.dtype = DTYPE_INT32, .device = DEVICE_CPU,
                                     .has_dtype = true, .has_device = true};
static const TensorConfig cfg_i64 = {.dtype = DTYPE_INT64, .device = DEVICE_CPU,
                                     .has_dtype = true, .has_device = true};

static int test_f64_arith(void) {
    double a[] = {1.0, 2.5, -3.0, 4.0};
    double b[] = {0.5, 0.5,  2.0, 8.0};
    Tensor* ta = tensor_from_data(a, (int[]){4}, 1, &cfg_f64);
    Tensor* tb = tensor_from_data(b, (int[]){4}, 1, &cfg_f64);

    Tensor* s = uop_add(ta, tb); tensor_ensure_executed(s);
    Tensor* d = uop_sub(ta, tb); tensor_ensure_executed(d);
    Tensor* m = uop_mul(ta, tb); tensor_ensure_executed(m);

    int ok = s->dtype == DTYPE_FLOAT64 && d->dtype == DTYPE_FLOAT64;
    const double* sd = (const double*)s->data;
    const double* dd = (const double*)d->data;
    const double* md = (const double*)m->data;
    for (int i = 0; i < 4; i++) {
        ok = ok && fabs(sd[i] - (a[i] + b[i])) < 1e-12;
        ok = ok && fabs(dd[i] - (a[i] - b[i])) < 1e-12;
        ok = ok && fabs(md[i] - (a[i] * b[i])) < 1e-12;
    }
    tensor_free(ta); tensor_free(tb); tensor_free(s); tensor_free(d); tensor_free(m);
    return ok;
}

/* 1e8 + 1 is exact in f64 but rounds to 1e8 in f32 — proves native f64 compute. */
static int test_f64_precision(void) {
    double a[] = {1e8};
    double b[] = {1.0};
    Tensor* ta = tensor_from_data(a, (int[]){1}, 1, &cfg_f64);
    Tensor* tb = tensor_from_data(b, (int[]){1}, 1, &cfg_f64);
    Tensor* s  = uop_add(ta, tb);
    tensor_ensure_executed(s);
    double got = ((const double*)s->data)[0];
    int ok = got == 100000001.0;   /* would be 1e8 if computed in f32 */
    tensor_free(ta); tensor_free(tb); tensor_free(s);
    return ok;
}

/* 2^24 + 1 = 16777217 is exact in int32 but NOT representable in f32. */
static int test_i32_exact(void) {
    int32_t a[] = {16777216, 100, 7};
    int32_t b[] = {1, 25, 6};
    Tensor* ta = tensor_from_data(a, (int[]){3}, 1, &cfg_i32);
    Tensor* tb = tensor_from_data(b, (int[]){3}, 1, &cfg_i32);
    Tensor* s = uop_add(ta, tb); tensor_ensure_executed(s);
    Tensor* p = uop_mul(ta, tb); tensor_ensure_executed(p);
    const int32_t* sd = (const int32_t*)s->data;
    const int32_t* pd = (const int32_t*)p->data;
    int ok = s->dtype == DTYPE_INT32 &&
             sd[0] == 16777217 && sd[1] == 125 && sd[2] == 13 &&
             pd[1] == 2500 && pd[2] == 42;
    tensor_free(ta); tensor_free(tb); tensor_free(s); tensor_free(p);
    return ok;
}

/* int64 product 1e6 * 1e6 = 1e12 is exact in int64, lossy as f32. */
static int test_i64_exact(void) {
    int64_t a[] = {1000000, 9};
    int64_t b[] = {1000000, 8};
    Tensor* ta = tensor_from_data(a, (int[]){2}, 1, &cfg_i64);
    Tensor* tb = tensor_from_data(b, (int[]){2}, 1, &cfg_i64);
    Tensor* p = uop_mul(ta, tb); tensor_ensure_executed(p);
    const int64_t* pd = (const int64_t*)p->data;
    int ok = p->dtype == DTYPE_INT64 && pd[0] == 1000000000000LL && pd[1] == 72;
    tensor_free(ta); tensor_free(tb); tensor_free(p);
    return ok;
}

/* f64 broadcasting: scalar * vector. */
static int test_f64_broadcast(void) {
    double v[] = {1.0, 2.0, 3.0, 4.0};
    double s[] = {2.5};
    Tensor* tv = tensor_from_data(v, (int[]){4}, 1, &cfg_f64);
    Tensor* ts = tensor_from_data(s, (int[]){1}, 1, &cfg_f64);
    Tensor* r = uop_mul(tv, ts); tensor_ensure_executed(r);
    const double* rd = (const double*)r->data;
    int ok = 1;
    for (int i = 0; i < 4; i++) ok = ok && fabs(rd[i] - v[i] * 2.5) < 1e-12;
    tensor_free(tv); tensor_free(ts); tensor_free(r);
    return ok;
}

/* f64 unary: neg/abs plus exp/log round-trip in double precision. */
static int test_f64_unary(void) {
    double a[] = {0.5, 1.0, 2.0, 4.0};
    Tensor* ta = tensor_from_data(a, (int[]){4}, 1, &cfg_f64);

    Tensor* ng = uop_neg(ta);  tensor_ensure_executed(ng);
    Tensor* ab = uop_abs(uop_neg(ta)); tensor_ensure_executed(ab);
    Tensor* rt = uop_exp(uop_log(ta)); tensor_ensure_executed(rt); /* exp(log(x)) ~ x */
    Tensor* sq = uop_sqrt(ta); tensor_ensure_executed(sq);

    const double* ngd = (const double*)ng->data;
    const double* abd = (const double*)ab->data;
    const double* rtd = (const double*)rt->data;
    const double* sqd = (const double*)sq->data;
    int ok = ng->dtype == DTYPE_FLOAT64;
    for (int i = 0; i < 4; i++) {
        ok = ok && fabs(ngd[i] + a[i]) < 1e-12;
        ok = ok && fabs(abd[i] - a[i]) < 1e-12;
        ok = ok && fabs(rtd[i] - a[i]) < 1e-6;      /* f64 round-trip is tight */
        ok = ok && fabs(sqd[i] - sqrt(a[i])) < 1e-12;
    }
    tensor_free(ta);
    return ok;
}

/* integer neg/abs. */
static int test_int_unary(void) {
    int32_t a[] = {-5, 3, -100, 42};
    Tensor* ta = tensor_from_data(a, (int[]){4}, 1, &cfg_i32);
    Tensor* ng = uop_neg(ta); tensor_ensure_executed(ng);
    Tensor* ab = uop_abs(ta); tensor_ensure_executed(ab);
    const int32_t* ngd = (const int32_t*)ng->data;
    const int32_t* abd = (const int32_t*)ab->data;
    int ok = ng->dtype == DTYPE_INT32 &&
             ngd[0] == 5 && ngd[2] == 100 &&
             abd[0] == 5 && abd[2] == 100 && abd[3] == 42;
    tensor_free(ta); tensor_free(ng); tensor_free(ab);
    return ok;
}

/* global reductions across dtypes. */
static int test_reductions_global(void) {
    double a[] = {1.0, 2.0, 3.0, 4.0, 5.0};   /* sum 15, mean 3 */
    Tensor* ta = tensor_from_data(a, (int[]){5}, 1, &cfg_f64);
    ReduceParams rp = {NULL, 0, false};
    Tensor* s = uop_sum(ta, &rp);  tensor_ensure_executed(s);
    Tensor* m = uop_mean(ta, &rp); tensor_ensure_executed(m);
    int ok = s->dtype == DTYPE_FLOAT64 &&
             fabs(((const double*)s->data)[0] - 15.0) < 1e-12 &&
             fabs(((const double*)m->data)[0] - 3.0) < 1e-12;

    int64_t b[] = {1000000000000LL, 2000000000000LL, 3};  /* exact in int64 */
    Tensor* tb = tensor_from_data(b, (int[]){3}, 1, &cfg_i64);
    Tensor* sb = uop_sum(tb, &rp); tensor_ensure_executed(sb);
    ok = ok && ((const int64_t*)sb->data)[0] == 3000000000003LL;

    int32_t c[] = {3, -7, 42, 10};
    Tensor* tc = tensor_from_data(c, (int[]){4}, 1, &cfg_i32);
    Tensor* mx = uop_max_reduce(tc, &rp); tensor_ensure_executed(mx);
    Tensor* mn = uop_min_reduce(tc, &rp); tensor_ensure_executed(mn);
    ok = ok && ((const int32_t*)mx->data)[0] == 42 && ((const int32_t*)mn->data)[0] == -7;

    tensor_free(ta); tensor_free(tb); tensor_free(tc);
    tensor_free(s); tensor_free(m); tensor_free(sb); tensor_free(mx); tensor_free(mn);
    return ok;
}

/* per-dimension reductions on an f64 [2,3] matrix. */
static int test_reductions_axis(void) {
    double a[] = {1, 2, 3, 4, 5, 6};   /* [[1,2,3],[4,5,6]] */
    Tensor* ta = tensor_from_data(a, (int[]){2, 3}, 2, &cfg_f64);

    int d1 = 1; ReduceParams rp1 = {&d1, 1, false};   /* sum cols -> [6,15] */
    Tensor* s1 = uop_sum(ta, &rp1); tensor_ensure_executed(s1);
    const double* s1d = (const double*)s1->data;
    int ok = s1->numel == 2 && fabs(s1d[0] - 6.0) < 1e-12 && fabs(s1d[1] - 15.0) < 1e-12;

    int d0 = 0; ReduceParams rp0 = {&d0, 1, false};   /* sum rows -> [5,7,9] */
    Tensor* s0 = uop_sum(ta, &rp0); tensor_ensure_executed(s0);
    const double* s0d = (const double*)s0->data;
    ok = ok && s0->numel == 3 &&
         fabs(s0d[0] - 5.0) < 1e-12 && fabs(s0d[1] - 7.0) < 1e-12 && fabs(s0d[2] - 9.0) < 1e-12;

    tensor_free(ta); tensor_free(s1); tensor_free(s0);
    return ok;
}

int main(void) {
    printf("=== multi-dtype compute: f64 + integers ===\n");
    check("f64_arith",     test_f64_arith());
    check("f64_precision", test_f64_precision());
    check("i32_exact",     test_i32_exact());
    check("i64_exact",     test_i64_exact());
    check("f64_broadcast", test_f64_broadcast());
    check("f64_unary",     test_f64_unary());
    check("int_unary",     test_int_unary());
    check("reductions_global", test_reductions_global());
    check("reductions_axis",   test_reductions_axis());
    printf("\nResults: %d/%d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
