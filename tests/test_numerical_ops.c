
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tensor/tensor.h"
#include "ops/uops.h"

static int tests_passed = 0;
static int tests_total  = 0;

static const TensorConfig cpu_f32 = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
    .has_dtype = true, .has_device = true
};

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

#define APPROX(a, b)    (fabsf((a) - (b)) < 1e-4f)
#define APPROX_REL(a,b) (fabsf((a)-(b)) <= 1e-3f * (0.5f*(fabsf(a)+fabsf(b)) + 1e-8f))

#define MAKE1D(arr) tensor_from_data((arr), (int[]){(int)(sizeof(arr)/sizeof(arr[0]))}, 1, &cpu_f32)

static int test_sin_cos_identity(void) {
    float data[] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, -1.0f, -2.5f};
    int n = (int)(sizeof(data)/sizeof(data[0]));
    int shape[] = {n};
    Tensor* x = tensor_from_data(data, shape, 1, &cpu_f32);

    Tensor* s  = uop_sin(x);
    Tensor* c  = uop_cos(x);
    Tensor* s2 = uop_mul(s, s);
    Tensor* c2 = uop_mul(c, c);
    Tensor* r  = uop_add(s2, c2);
    tensor_ensure_executed(r);
    float* d = r->data;
    for (int i = 0; i < n; i++)
        if (!APPROX(d[i], 1.0f)) return 0;
    tensor_free(x); tensor_free(s); tensor_free(c);
    tensor_free(s2); tensor_free(c2); tensor_free(r);
    return 1;
}

static int test_exp_log_identity(void) {
    float data[] = {0.1f, 0.5f, 1.0f, 2.0f, 10.0f, 100.0f};
    int n = (int)(sizeof(data)/sizeof(data[0]));
    int shape[] = {n};
    Tensor* x   = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* lx  = uop_log(x);
    Tensor* out = uop_exp(lx);
    tensor_ensure_executed(out);
    float* d = out->data;
    for (int i = 0; i < n; i++)
        if (!APPROX_REL(d[i], data[i])) return 0;
    tensor_free(x); tensor_free(lx); tensor_free(out);
    return 1;
}

static int test_log2_exp2_identity(void) {
    float data[] = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f, 7.0f};
    int n = (int)(sizeof(data)/sizeof(data[0]));
    int shape[] = {n};
    Tensor* x   = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* e   = uop_exp2(x);
    Tensor* out = uop_log2(e);
    tensor_ensure_executed(out);
    float* d = out->data;
    for (int i = 0; i < n; i++)
        if (!APPROX(d[i], data[i])) return 0;
    tensor_free(x); tensor_free(e); tensor_free(out);
    return 1;
}

static int test_floor_ceil_bounds(void) {
    float data[] = {-2.7f, -0.1f, 0.0f, 0.9f, 1.5f, 3.99f};
    int n = (int)(sizeof(data)/sizeof(data[0]));
    int shape[] = {n};
    Tensor* x  = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* fl = uop_floor(x);
    Tensor* cl = uop_ceil(x);
    tensor_ensure_executed(fl);
    tensor_ensure_executed(cl);
    float* f = fl->data;
    float* c = cl->data;
    for (int i = 0; i < n; i++) {
        if (f[i] > data[i] + 1e-6f) return 0;  /* floor <= x */
        if (c[i] < data[i] - 1e-6f) return 0;  /* ceil  >= x */
        if (c[i] - f[i] > 1.0f + 1e-5f) return 0; /* diff <= 1 */
    }
    tensor_free(x); tensor_free(fl); tensor_free(cl);
    return 1;
}

static int test_round_nearest(void) {
    float in[]  = {0.4f, 0.6f, -0.4f, -0.6f, 1.5f, 2.5f};
    float exp[] = {0.0f, 1.0f,  0.0f, -1.0f, 2.0f, 2.0f}; /* banker's rounding */
    int n = (int)(sizeof(in)/sizeof(in[0]));
    int shape[] = {n};
    Tensor* x   = tensor_from_data(in, shape, 1, &cpu_f32);
    Tensor* out = uop_round(x);
    tensor_ensure_executed(out);
    float* d = out->data;
    for (int i = 0; i < n; i++)
        if (!APPROX(d[i], exp[i])) return 0;
    tensor_free(x); tensor_free(out);
    return 1;
}

static int test_trunc(void) {
    float in[]  = {1.7f, -1.7f, 2.0f, -2.0f, 0.9f};
    float exp[] = {1.0f, -1.0f, 2.0f, -2.0f, 0.0f};
    int n = (int)(sizeof(in)/sizeof(in[0]));
    int shape[] = {n};
    Tensor* x   = tensor_from_data(in, shape, 1, &cpu_f32);
    Tensor* out = uop_trunc(x);
    tensor_ensure_executed(out);
    float* d = out->data;
    for (int i = 0; i < n; i++)
        if (!APPROX(d[i], exp[i])) return 0;
    tensor_free(x); tensor_free(out);
    return 1;
}

static int test_sign_values(void) {
    float in[]  = {-5.0f, -0.0f, 0.0f, 0.001f, 3.14f};
    float exp[] = {-1.0f,  0.0f, 0.0f,  1.0f,   1.0f};
    int n = (int)(sizeof(in)/sizeof(in[0]));
    int shape[] = {n};
    Tensor* x   = tensor_from_data(in, shape, 1, &cpu_f32);
    Tensor* out = uop_sign(x);
    tensor_ensure_executed(out);
    float* d = out->data;
    for (int i = 0; i < n; i++)
        if (!APPROX(d[i], exp[i])) return 0;
    tensor_free(x); tensor_free(out);
    return 1;
}

static int test_square_vs_mul(void) {
    float data[] = {-3.0f, -1.0f, 0.0f, 2.0f, 5.0f};
    int n = (int)(sizeof(data)/sizeof(data[0]));
    int shape[] = {n};
    Tensor* x   = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* sq  = uop_square(x);
    Tensor* mul = uop_mul(x, x);
    tensor_ensure_executed(sq);
    tensor_ensure_executed(mul);
    float* ds = sq->data;
    float* dm = mul->data;
    for (int i = 0; i < n; i++)
        if (!APPROX(ds[i], dm[i])) return 0;
    tensor_free(x); tensor_free(sq); tensor_free(mul);
    return 1;
}

static int test_rsqrt(void) {
    float data[] = {1.0f, 4.0f, 9.0f, 0.25f};
    float exp[]  = {1.0f, 0.5f, 1.0f/3.0f, 2.0f};
    int n = (int)(sizeof(data)/sizeof(data[0]));
    int shape[] = {n};
    Tensor* x   = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* out = uop_rsqrt(x);
    tensor_ensure_executed(out);
    float* d = out->data;
    for (int i = 0; i < n; i++)
        if (!APPROX_REL(d[i], exp[i])) return 0;
    tensor_free(x); tensor_free(out);
    return 1;
}

static int test_erf_bounds(void) {
    float in[]  = {-3.0f, 0.0f, 3.0f};
    float exp[] = {-1.0f, 0.0f, 1.0f};  /* erf(±3) ≈ ±0.99998 */
    int n = 3;
    int shape[] = {n};
    Tensor* x   = tensor_from_data(in, shape, 1, &cpu_f32);
    Tensor* out = uop_erf(x);
    tensor_ensure_executed(out);
    float* d = out->data;
    /* erf(0) == 0 exactly */
    if (!APPROX(d[1], 0.0f)) return 0;
    /* erf(3) > 0.999 */
    if (d[2] < 0.999f) return 0;
    /* erf(-3) < -0.999 */
    if (d[0] > -0.999f) return 0;
    tensor_free(x); tensor_free(out);
    return 1;
}

static int test_asin_acos_identity(void) {
    /* sin(asin(x)) == x for x in [-1, 1] */
    float data[] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    int n = (int)(sizeof(data)/sizeof(data[0]));
    int shape[] = {n};
    Tensor* x     = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* asinx = uop_asin(x);
    Tensor* back  = uop_sin(asinx);
    tensor_ensure_executed(back);
    float* d = back->data;
    for (int i = 0; i < n; i++)
        if (!APPROX(d[i], data[i])) return 0;
    tensor_free(x); tensor_free(asinx); tensor_free(back);
    return 1;
}

static int test_atan_range(void) {
    /* atan(x) in (-pi/2, pi/2) */
    float data[] = {-1000.0f, -1.0f, 0.0f, 1.0f, 1000.0f};
    int n = (int)(sizeof(data)/sizeof(data[0]));
    int shape[] = {n};
    Tensor* x   = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* out = uop_atan(x);
    tensor_ensure_executed(out);
    float* d = out->data;
    float half_pi = 3.14159265f / 2.0f;
    for (int i = 0; i < n; i++) {
        if (d[i] <= -half_pi - 1e-4f) return 0;
        if (d[i] >=  half_pi + 1e-4f) return 0;
    }
    /* atan(0) == 0 */
    if (!APPROX(d[2], 0.0f)) return 0;
    tensor_free(x); tensor_free(out);
    return 1;
}

static int test_sinh_cosh_identity(void) {
    float data[] = {-2.0f, -0.5f, 0.0f, 0.5f, 2.0f};
    int n = (int)(sizeof(data)/sizeof(data[0]));
    int shape[] = {n};
    Tensor* x   = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* sh  = uop_sinh(x);
    Tensor* ch  = uop_cosh(x);
    Tensor* sh2 = uop_mul(sh, sh);
    Tensor* ch2 = uop_mul(ch, ch);
    Tensor* r   = uop_sub(ch2, sh2);  /* should be 1 */
    tensor_ensure_executed(r);
    float* d = r->data;
    for (int i = 0; i < n; i++)
        if (!APPROX(d[i], 1.0f)) return 0;
    tensor_free(x); tensor_free(sh); tensor_free(ch);
    tensor_free(sh2); tensor_free(ch2); tensor_free(r);
    return 1;
}

static int test_asinh_inverse(void) {
    float data[] = {-5.0f, -1.0f, 0.0f, 1.0f, 5.0f};
    int n = (int)(sizeof(data)/sizeof(data[0]));
    int shape[] = {n};
    Tensor* x    = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* ax   = uop_asinh(x);
    Tensor* back = uop_sinh(ax);
    tensor_ensure_executed(back);
    float* d = back->data;
    for (int i = 0; i < n; i++)
        if (!APPROX_REL(d[i], data[i])) return 0;
    tensor_free(x); tensor_free(ax); tensor_free(back);
    return 1;
}

static int test_log10_identity(void) {
    float data[] = {-2.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    int n = (int)(sizeof(data)/sizeof(data[0]));
    int shape[] = {n};
    Tensor* x    = tensor_from_data(data, shape, 1, &cpu_f32);
    /* 10^x = exp(x * ln10) */
    float ln10 = 2.302585f;
    float scaled[5];
    for (int i = 0; i < n; i++) scaled[i] = data[i] * ln10;
    Tensor* sc   = tensor_from_data(scaled, shape, 1, &cpu_f32);
    Tensor* pw   = uop_exp(sc);
    Tensor* out  = uop_log10(pw);
    tensor_ensure_executed(out);
    float* d = out->data;
    for (int i = 0; i < n; i++)
        if (!APPROX(d[i], data[i])) return 0;
    tensor_free(x); tensor_free(sc); tensor_free(pw); tensor_free(out);
    return 1;
}

static int test_isinf_isnan_isfinite(void) {
    float inf = 1.0f / 0.0f;
    float nan = 0.0f / 0.0f;
    float data[] = {1.0f, inf, -inf, nan, 0.0f};
    int n = 5;
    int shape[] = {n};
    Tensor* x = tensor_from_data(data, shape, 1, &cpu_f32);

    Tensor* is_inf    = uop_isinf(x);
    Tensor* is_nan    = uop_isnan(x);
    Tensor* is_finite = uop_isfinite(x);
    tensor_ensure_executed(is_inf);
    tensor_ensure_executed(is_nan);
    tensor_ensure_executed(is_finite);

    float* fi = is_inf->data;
    float* fn = is_nan->data;
    float* ff = is_finite->data;

    /* 1.0f: finite, not inf, not nan */
    if (fi[0] != 0.0f) return 0;
    if (fn[0] != 0.0f) return 0;
    if (ff[0] != 1.0f) return 0;

    /* +inf: is_inf, not nan, not finite */
    if (fi[1] != 1.0f) return 0;
    if (ff[1] != 0.0f) return 0;

    /* nan: is_nan, not inf, not finite */
    if (fn[3] != 1.0f) return 0;
    if (ff[3] != 0.0f) return 0;

    tensor_free(x); tensor_free(is_inf); tensor_free(is_nan); tensor_free(is_finite);
    return 1;
}

static int test_logical_not(void) {
    float data[] = {0.0f, 1.0f, -1.0f, 0.5f, 0.0f};
    float exp[]  = {1.0f, 0.0f,  0.0f, 0.0f, 1.0f};
    int n = (int)(sizeof(data)/sizeof(data[0]));
    int shape[] = {n};
    Tensor* x   = tensor_from_data(data, shape, 1, &cpu_f32);
    Tensor* out = uop_logical_not(x);
    tensor_ensure_executed(out);
    float* d = out->data;
    for (int i = 0; i < n; i++)
        if (!APPROX(d[i], exp[i])) return 0;
    tensor_free(x); tensor_free(out);
    return 1;
}

static int test_logical_and_or(void) {
    float a[]   = {0.0f, 1.0f, 0.0f, 1.0f};
    float b[]   = {0.0f, 0.0f, 1.0f, 1.0f};
    float eand[]= {0.0f, 0.0f, 0.0f, 1.0f};
    float eor[] = {0.0f, 1.0f, 1.0f, 1.0f};
    int n = 4;
    int shape[] = {n};
    Tensor* ta  = tensor_from_data(a, shape, 1, &cpu_f32);
    Tensor* tb  = tensor_from_data(b, shape, 1, &cpu_f32);
    Tensor* land = uop_logical_and(ta, tb);
    Tensor* lor  = uop_logical_or(ta, tb);
    tensor_ensure_executed(land);
    tensor_ensure_executed(lor);
    float* da = land->data;
    float* do_ = lor->data;
    for (int i = 0; i < n; i++) {
        if (!APPROX(da[i], eand[i])) return 0;
        if (!APPROX(do_[i], eor[i])) return 0;
    }
    tensor_free(ta); tensor_free(tb); tensor_free(land); tensor_free(lor);
    return 1;
}

static int test_comparison_ops(void) {
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[] = {1.0f, 1.0f, 4.0f, 3.0f};
    int n = 4;
    int shape[] = {n};
    Tensor* ta  = tensor_from_data(a, shape, 1, &cpu_f32);
    Tensor* tb  = tensor_from_data(b, shape, 1, &cpu_f32);

    Tensor* eq  = uop_cmpeq(ta, tb);
    Tensor* ne  = uop_cmpne(ta, tb);
    Tensor* le  = uop_cmple(ta, tb);
    Tensor* gt  = uop_cmpgt(ta, tb);
    Tensor* ge  = uop_cmpge(ta, tb);
    tensor_ensure_executed(eq); tensor_ensure_executed(ne);
    tensor_ensure_executed(le); tensor_ensure_executed(gt);
    tensor_ensure_executed(ge);

    float* deq = eq->data; float* dne = ne->data;
    float* dle = le->data; float* dgt = gt->data;
    float* dge = ge->data;

    /* a==b: [1,0,0,0] */
    float eeq[] = {1,0,0,0}, ene[] = {0,1,1,1};
    float ele[] = {1,0,1,0}, egt[] = {0,1,0,1}, ege[] = {1,1,0,1};
    for (int i = 0; i < n; i++) {
        if (!APPROX(deq[i], eeq[i])) return 0;
        if (!APPROX(dne[i], ene[i])) return 0;
        if (!APPROX(dle[i], ele[i])) return 0;
        if (!APPROX(dgt[i], egt[i])) return 0;
        if (!APPROX(dge[i], ege[i])) return 0;
    }
    tensor_free(ta); tensor_free(tb);
    tensor_free(eq); tensor_free(ne); tensor_free(le);
    tensor_free(gt); tensor_free(ge);
    return 1;
}

static int test_minimum(void) {
    float a[]  = {1.0f, 5.0f, -2.0f,  3.0f};
    float b[]  = {2.0f, 3.0f,  1.0f, -1.0f};
    float exp[]= {1.0f, 3.0f, -2.0f, -1.0f};
    int n = 4;
    int shape[] = {n};
    Tensor* ta  = tensor_from_data(a, shape, 1, &cpu_f32);
    Tensor* tb  = tensor_from_data(b, shape, 1, &cpu_f32);
    Tensor* out = uop_minimum(ta, tb);
    tensor_ensure_executed(out);
    float* d = out->data;
    for (int i = 0; i < n; i++)
        if (!APPROX(d[i], exp[i])) return 0;
    tensor_free(ta); tensor_free(tb); tensor_free(out);
    return 1;
}

static int test_mod(void) {
    float a[]  = {7.0f, -7.0f,  7.0f, -7.0f};
    float b[]  = {3.0f,  3.0f, -3.0f, -3.0f};
    /* fmodf semantics: sign follows dividend */
    float exp[]= {1.0f, -1.0f,  1.0f, -1.0f};
    int n = 4;
    int shape[] = {n};
    Tensor* ta  = tensor_from_data(a, shape, 1, &cpu_f32);
    Tensor* tb  = tensor_from_data(b, shape, 1, &cpu_f32);
    Tensor* out = uop_mod(ta, tb);
    tensor_ensure_executed(out);
    float* d = out->data;
    for (int i = 0; i < n; i++)
        if (!APPROX(d[i], exp[i])) return 0;
    tensor_free(ta); tensor_free(tb); tensor_free(out);
    return 1;
}

static int test_logaddexp(void) {
    float a[]  = {0.0f, 1.0f, 100.0f};
    float b[]  = {0.0f, 2.0f, 100.0f};
    int n = 3;
    int shape[] = {n};
    Tensor* ta  = tensor_from_data(a, shape, 1, &cpu_f32);
    Tensor* tb  = tensor_from_data(b, shape, 1, &cpu_f32);
    Tensor* out = uop_logaddexp(ta, tb);
    tensor_ensure_executed(out);
    float* d = out->data;
    /* logaddexp(0,0) = log(2) */
    if (!APPROX(d[0], logf(2.0f))) return 0;
    /* logaddexp(1,2) = log(e + e²) */
    if (!APPROX_REL(d[1], logf(expf(1.0f) + expf(2.0f)))) return 0;
    tensor_free(ta); tensor_free(tb); tensor_free(out);
    return 1;
}

static int test_copysign(void) {
    float a[]  = {3.0f, -3.0f, 3.0f, -3.0f};
    float b[]  = {1.0f,  1.0f,-1.0f, -1.0f};
    float exp[]= {3.0f,  3.0f,-3.0f, -3.0f};
    int n = 4;
    int shape[] = {n};
    Tensor* ta  = tensor_from_data(a, shape, 1, &cpu_f32);
    Tensor* tb  = tensor_from_data(b, shape, 1, &cpu_f32);
    Tensor* out = uop_copysign(ta, tb);
    tensor_ensure_executed(out);
    float* d = out->data;
    for (int i = 0; i < n; i++)
        if (!APPROX(d[i], exp[i])) return 0;
    tensor_free(ta); tensor_free(tb); tensor_free(out);
    return 1;
}

static int test_idiv(void) {
    float a[]  = {7.0f, -7.0f,  7.0f, -7.0f};
    float b[]  = {3.0f,  3.0f, -3.0f, -3.0f};
    float exp[]= {2.0f, -3.0f, -3.0f,  2.0f}; /* floor division */
    int n = 4;
    int shape[] = {n};
    Tensor* ta  = tensor_from_data(a, shape, 1, &cpu_f32);
    Tensor* tb  = tensor_from_data(b, shape, 1, &cpu_f32);
    Tensor* out = uop_idiv(ta, tb);
    tensor_ensure_executed(out);
    float* d = out->data;
    for (int i = 0; i < n; i++)
        if (!APPROX(d[i], exp[i])) return 0;
    tensor_free(ta); tensor_free(tb); tensor_free(out);
    return 1;
}

int main(void) {
    printf("Numerical Op Correctness Tests\n");
    TEST(sin_cos_identity);
    TEST(exp_log_identity);
    TEST(log2_exp2_identity);
    TEST(floor_ceil_bounds);
    TEST(round_nearest);
    TEST(trunc);
    TEST(sign_values);
    TEST(square_vs_mul);
    TEST(rsqrt);
    TEST(erf_bounds);
    TEST(asin_acos_identity);
    TEST(atan_range);
    TEST(sinh_cosh_identity);
    TEST(asinh_inverse);
    TEST(log10_identity);
    TEST(isinf_isnan_isfinite);
    TEST(logical_not);
    TEST(logical_and_or);
    TEST(comparison_ops);
    TEST(minimum);
    TEST(mod);
    TEST(logaddexp);
    TEST(copysign);
    TEST(idiv);

    printf("\nResults: %d/%d passed\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
