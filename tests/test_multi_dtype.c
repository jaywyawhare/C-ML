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

static const TensorConfig cfg_f32 = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                                     .has_dtype = true, .has_device = true};
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

/* mixed-dtype binary ops promote to the wider type (numpy-style). */
static int test_promotion(void) {
    float  af[] = {1.0f, 2.0f, 3.0f};
    double bd[] = {0.5, 0.5, 0.5};
    Tensor* ta = tensor_from_data(af, (int[]){3}, 1, &cfg_f32);
    Tensor* tb = tensor_from_data(bd, (int[]){3}, 1, &cfg_f64);

    Tensor* s = uop_add(ta, tb);   /* f32 + f64 -> f64 */
    tensor_ensure_executed(s);
    int ok = s->dtype == DTYPE_FLOAT64;
    const double* sd = (const double*)s->data;
    ok = ok && fabs(sd[0] - 1.5) < 1e-12 && fabs(sd[1] - 2.5) < 1e-12 && fabs(sd[2] - 3.5) < 1e-12;

    /* int32 + int64 -> int64, computed exactly */
    int32_t ai[] = {1000000};
    int64_t bi[] = {1000000000000LL};
    Tensor* tia = tensor_from_data(ai, (int[]){1}, 1, &cfg_i32);
    Tensor* tib = tensor_from_data(bi, (int[]){1}, 1, &cfg_i64);
    Tensor* si = uop_add(tia, tib);
    tensor_ensure_executed(si);
    ok = ok && si->dtype == DTYPE_INT64 && ((const int64_t*)si->data)[0] == 1000001000000LL;

    tensor_free(ta); tensor_free(tb); tensor_free(s);
    tensor_free(tia); tensor_free(tib); tensor_free(si);
    return ok;
}

/* precision-preserving cast (not routed through f32). */
static int test_cast_precision(void) {
    int64_t a[] = {9007199254740993LL, 5};   /* 2^53 + 1, unrepresentable in f32/f64 exactly */
    Tensor* ta = tensor_from_data(a, (int[]){2}, 1, &cfg_i64);
    Tensor* tb = tensor_cast(ta, DTYPE_INT32);   /* int64 -> int32 truncates value, but low bits exact */
    Tensor* tc = tensor_cast(ta, DTYPE_INT64);   /* int64 -> int64 exact */
    int ok = tb->dtype == DTYPE_INT32 && tc->dtype == DTYPE_INT64 &&
             ((const int64_t*)tc->data)[0] == 9007199254740993LL &&
             ((const int32_t*)tb->data)[1] == 5;
    tensor_free(ta); tensor_free(tb); tensor_free(tc);
    return ok;
}

/* comparisons return DTYPE_BOOL (numpy-style), across input dtypes. */
static int test_comparisons(void) {
    int32_t a[] = {1, 5, 3, 2};
    int32_t b[] = {2, 2, 2, 2};
    Tensor* ta = tensor_from_data(a, (int[]){4}, 1, &cfg_i32);
    Tensor* tb = tensor_from_data(b, (int[]){4}, 1, &cfg_i32);

    Tensor* lt = uop_cmplt(ta, tb); tensor_ensure_executed(lt);  /* [1,0,0,0] */
    Tensor* gt = uop_cmpgt(ta, tb); tensor_ensure_executed(gt);  /* [0,1,1,0] */
    const uint8_t* ld = (const uint8_t*)lt->data;   /* bool storage is uint8 */
    const uint8_t* gd = (const uint8_t*)gt->data;
    int ok = lt->dtype == DTYPE_BOOL && gt->dtype == DTYPE_BOOL &&
             ld[0] == 1 && ld[1] == 0 && ld[2] == 0 && ld[3] == 0 &&
             gd[0] == 0 && gd[1] == 1 && gd[2] == 1 && gd[3] == 0;

    /* f64 equality also yields bool */
    double c[] = {1.0, 2.0, 3.0};
    double d[] = {1.0, 9.0, 3.0};
    Tensor* tc = tensor_from_data(c, (int[]){3}, 1, &cfg_f64);
    Tensor* td = tensor_from_data(d, (int[]){3}, 1, &cfg_f64);
    Tensor* eq = uop_cmpeq(tc, td); tensor_ensure_executed(eq);
    const uint8_t* ed = (const uint8_t*)eq->data;
    ok = ok && eq->dtype == DTYPE_BOOL && ed[0] == 1 && ed[1] == 0 && ed[2] == 1;

    /* a bool mask multiplied by a float promotes to float 0.0/1.0 */
    float fv[] = {10.0f, 20.0f, 30.0f, 40.0f};
    Tensor* tf = tensor_from_data(fv, (int[]){4}, 1, &cfg_f32);
    Tensor* masked = uop_mul(tf, lt);   /* f32 * bool -> f32 */
    tensor_ensure_executed(masked);
    const float* mk = (const float*)masked->data;
    ok = ok && masked->dtype == DTYPE_FLOAT32 &&
         mk[0] == 10.0f && mk[1] == 0.0f && mk[2] == 0.0f && mk[3] == 0.0f;

    tensor_free(ta); tensor_free(tb); tensor_free(lt); tensor_free(gt);
    tensor_free(tc); tensor_free(td); tensor_free(eq);
    tensor_free(tf); tensor_free(masked);
    return ok;
}

/* f64 and int matmul in native precision. */
static int test_matmul(void) {
    /* f64 [2,3] @ [3,2] */
    double a[] = {1, 2, 3, 4, 5, 6};          /* [[1,2,3],[4,5,6]] */
    double b[] = {7, 8, 9, 10, 11, 12};       /* [[7,8],[9,10],[11,12]] */
    Tensor* ta = tensor_from_data(a, (int[]){2, 3}, 2, &cfg_f64);
    Tensor* tb = tensor_from_data(b, (int[]){3, 2}, 2, &cfg_f64);
    Tensor* c  = uop_matmul(ta, tb);
    tensor_ensure_executed(c);
    /* [[1*7+2*9+3*11, 1*8+2*10+3*12],[...]] = [[58,64],[139,154]] */
    const double* cd = (const double*)c->data;
    int ok = c->dtype == DTYPE_FLOAT64 && c->numel == 4 &&
             fabs(cd[0] - 58) < 1e-9 && fabs(cd[1] - 64) < 1e-9 &&
             fabs(cd[2] - 139) < 1e-9 && fabs(cd[3] - 154) < 1e-9;

    /* int32 matmul, exact integer accumulation */
    int32_t ai[] = {1, 2, 3, 4};              /* [[1,2],[3,4]] */
    int32_t bi[] = {5, 6, 7, 8};              /* [[5,6],[7,8]] */
    Tensor* tia = tensor_from_data(ai, (int[]){2, 2}, 2, &cfg_i32);
    Tensor* tib = tensor_from_data(bi, (int[]){2, 2}, 2, &cfg_i32);
    Tensor* ci  = uop_matmul(tia, tib);
    tensor_ensure_executed(ci);
    /* [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]] */
    const int32_t* cid = (const int32_t*)ci->data;
    ok = ok && ci->dtype == DTYPE_INT32 &&
         cid[0] == 19 && cid[1] == 22 && cid[2] == 43 && cid[3] == 50;

    /* f64 precision: (1e8)·(1) + (1)·(1) accumulated exactly in f64 */
    double pa[] = {1e8, 1.0};                 /* [1,2] */
    double pb[] = {1.0, 1.0};                 /* [2,1] */
    Tensor* tpa = tensor_from_data(pa, (int[]){1, 2}, 2, &cfg_f64);
    Tensor* tpb = tensor_from_data(pb, (int[]){2, 1}, 2, &cfg_f64);
    Tensor* pc  = uop_matmul(tpa, tpb);
    tensor_ensure_executed(pc);
    ok = ok && ((const double*)pc->data)[0] == 100000001.0;

    tensor_free(ta); tensor_free(tb); tensor_free(c);
    tensor_free(tia); tensor_free(tib); tensor_free(ci);
    tensor_free(tpa); tensor_free(tpb); tensor_free(pc);
    return ok;
}

/* f64 direct conv2d: 3x3 input, 2x2 identity-ish kernel, stride 1, no pad. */
static int test_conv2d(void) {
    double in[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};   /* [1,1,3,3] */
    double w[]  = {1, 0, 0, 1};                   /* [1,1,2,2] */
    Tensor* ti = tensor_from_data(in, (int[]){1, 1, 3, 3}, 4, &cfg_f64);
    Tensor* tw = tensor_from_data(w,  (int[]){1, 1, 2, 2}, 4, &cfg_f64);

    int ks[] = {2, 2}, st[] = {1, 1}, pd[] = {0, 0}, dl[] = {1, 1};
    Conv2DParams p = {.kernel_size = ks, .stride = st, .padding = pd, .dilation = dl,
                      .groups = 1, .bias = false, .use_winograd = false};
    Tensor* c = uop_conv2d(ti, tw, NULL, &p);
    tensor_ensure_executed(c);
    /* out = [[1+5, 2+6],[4+8, 5+9]] = [[6,8],[12,14]] */
    const double* cd = (const double*)c->data;
    int ok = c && c->dtype == DTYPE_FLOAT64 && c->numel == 4 &&
             fabs(cd[0] - 6) < 1e-9 && fabs(cd[1] - 8) < 1e-9 &&
             fabs(cd[2] - 12) < 1e-9 && fabs(cd[3] - 14) < 1e-9;
    tensor_free(ti); tensor_free(tw); tensor_free(c);
    return ok;
}

/* f16/bf16 compute (in f32, stored as half). Values read back via
 * tensor_get_float; tolerances reflect half precision. */
static int test_half(void) {
    float av[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float bv[] = {0.5f, 1.5f, 2.5f, 0.25f};
    Tensor* af = tensor_from_data(av, (int[]){4}, 1, &cfg_f32);
    Tensor* bf = tensor_from_data(bv, (int[]){4}, 1, &cfg_f32);

    /* f16 binary + unary + reduction */
    Tensor* a = tensor_cast(af, DTYPE_FLOAT16);
    Tensor* b = tensor_cast(bf, DTYPE_FLOAT16);
    int ok = a && b && a->dtype == DTYPE_FLOAT16;

    Tensor* s = uop_add(a, b); tensor_ensure_executed(s);
    Tensor* m = uop_mul(a, b); tensor_ensure_executed(m);
    Tensor* ng = uop_neg(a);   tensor_ensure_executed(ng);
    ok = ok && s->dtype == DTYPE_FLOAT16 && m->dtype == DTYPE_FLOAT16;
    for (int i = 0; i < 4; i++) {
        ok = ok && fabsf(tensor_get_float(s, i) - (av[i] + bv[i])) < 1e-2f;
        ok = ok && fabsf(tensor_get_float(m, i) - (av[i] * bv[i])) < 1e-2f;
        ok = ok && fabsf(tensor_get_float(ng, i) - (-av[i])) < 1e-2f;
    }
    ReduceParams rp = {NULL, 0, false};
    Tensor* sm = uop_sum(a, &rp); tensor_ensure_executed(sm);   /* 1+2+3+4 = 10 */
    ok = ok && sm->dtype == DTYPE_FLOAT16 && fabsf(tensor_get_float(sm, 0) - 10.0f) < 1e-1f;

    /* bf16 add (wider tolerance — 8-bit mantissa) */
    Tensor* ba = tensor_cast(af, DTYPE_BFLOAT16);
    Tensor* bb = tensor_cast(bf, DTYPE_BFLOAT16);
    Tensor* bs = uop_add(ba, bb); tensor_ensure_executed(bs);
    ok = ok && bs->dtype == DTYPE_BFLOAT16;
    for (int i = 0; i < 4; i++)
        ok = ok && fabsf(tensor_get_float(bs, i) - (av[i] + bv[i])) < 1e-1f;

    tensor_free(af); tensor_free(bf); tensor_free(a); tensor_free(b);
    tensor_free(s); tensor_free(m); tensor_free(ng); tensor_free(sm);
    tensor_free(ba); tensor_free(bb); tensor_free(bs);
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
    check("promotion",         test_promotion());
    check("cast_precision",    test_cast_precision());
    check("comparisons",       test_comparisons());
    check("matmul",            test_matmul());
    check("conv2d",            test_conv2d());
    check("half_f16_bf16",     test_half());
    printf("\nResults: %d/%d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
