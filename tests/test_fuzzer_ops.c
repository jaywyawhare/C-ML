
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "cml.h"
#include "tensor/tensor.h"
#include "ops/uops.h"

static uint64_t rng_state;

static void rng_seed(uint64_t seed) { rng_state = seed ^ 0xdeadbeef1337ULL; }

static uint64_t rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

static int rng_int(int lo, int hi) { 
    return lo + (int)(rng_next() % (unsigned)(hi - lo));
}

static float rng_float(float lo, float hi) {
    return lo + (hi - lo) * ((float)(rng_next() & 0xFFFFFF) / (float)0x1000000);
}

static const TensorConfig cpu_f32 = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
    .has_dtype = true, .has_device = true
};

static int tests_passed = 0;
static int tests_total  = 0;

#define TEST(name)                                     \
    do {                                               \
        tests_total++;                                 \
        printf("  FUZZ: %s ... ", #name);              \
        fflush(stdout);                                \
        if (fuzz_##name()) {                           \
            tests_passed++;                            \
            printf("OK\n");                            \
        } else {                                       \
            printf("FAIL\n");                          \
        }                                              \
    } while (0)

static int has_nan_inf(Tensor* t) {
    if (!t || !t->data) return 0;
    float* d = t->data;
    for (size_t i = 0; i < t->numel; i++) {
        if (!isfinite(d[i])) return 1;
    }
    return 0;
}

static int make_shape(int* shape, int* ndim_out, int max_numel) {
    int ndim = rng_int(1, 5);
    *ndim_out = ndim;
    int numel = 1;
    for (int i = 0; i < ndim; i++) {
        int dim = rng_int(1, (int)cbrtf((float)max_numel) + 2);
        
        while (numel * dim > max_numel && dim > 1) dim--;
        shape[i] = dim;
        numel *= dim;
    }
    return numel;
}

static Tensor* make_positive_tensor(int* shape, int ndim, float range) {
    int n = 1;
    for (int i = 0; i < ndim; i++) n *= shape[i];
    float* data = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++)
        data[i] = rng_float(0.01f, range);
    Tensor* t = tensor_from_data(data, shape, ndim, &cpu_f32);
    free(data);
    return t;
}

static Tensor* make_signed_tensor(int* shape, int ndim, float range) {
    int n = 1;
    for (int i = 0; i < ndim; i++) n *= shape[i];
    float* data = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++)
        data[i] = rng_float(-range, range);
    Tensor* t = tensor_from_data(data, shape, ndim, &cpu_f32);
    free(data);
    return t;
}

static Tensor* make_asin_safe_tensor(int* shape, int ndim) {
    int n = 1;
    for (int i = 0; i < ndim; i++) n *= shape[i];
    float* data = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        data[i] = (float)sin((double)rng_int(0, 1000) / 100.0);
    }
    Tensor* t = tensor_from_data(data, shape, ndim, &cpu_f32);
    free(data);
    return t;
}
typedef Tensor* (*UnaryOp)(Tensor*);

static int fuzz_unary_op_no_crash(UnaryOp op, const char* name,
                                   int use_positive, int n_trials) {
    int failures = 0;
    for (int trial = 0; trial < n_trials; trial++) {
        int shape[8], ndim;
        make_shape(shape, &ndim, 512);
        Tensor* x = use_positive
            ? make_positive_tensor(shape, ndim, 5.0f)
            : make_signed_tensor(shape, ndim, 3.0f);
        if (!x) { failures++; continue; }

        Tensor* out = op(x);
        if (!out) {
            fprintf(stderr, "    [%s] NULL output for shape [", name);
            for (int i = 0; i < ndim; i++) fprintf(stderr, "%d%s", shape[i], i<ndim-1?",":"");
            fprintf(stderr, "]\n");
            failures++;
        } else {
            if (tensor_ensure_executed(out) != 0) failures++;
            else if (has_nan_inf(out)) {
                
                fprintf(stderr, "    [%s] NaN/Inf in output\n", name);
                failures++;
            }
            tensor_free(out);
        }
        tensor_free(x);
    }
    return (failures == 0);
}

#define FUZZ_UNARY(opfn, positive) \
    fuzz_unary_op_no_crash(opfn, #opfn, positive, 20)

static int fuzz_sin(void)         { return FUZZ_UNARY(uop_sin,  0); }
static int fuzz_cos(void)         { return FUZZ_UNARY(uop_cos,  0); }
static int fuzz_tanh(void)        { return FUZZ_UNARY(uop_tanh, 0); }
static int fuzz_sigmoid(void)     { return FUZZ_UNARY(uop_sigmoid, 0); }
static int fuzz_exp(void)         { return fuzz_unary_op_no_crash(uop_exp, "uop_exp", 0, 20); }
static int fuzz_neg(void)         { return FUZZ_UNARY(uop_neg,  0); }
static int fuzz_abs(void)         { return FUZZ_UNARY(uop_abs,  0); }
static int fuzz_floor(void)       { return FUZZ_UNARY(uop_floor, 0); }
static int fuzz_ceil(void)        { return FUZZ_UNARY(uop_ceil,  0); }
static int fuzz_round(void)       { return FUZZ_UNARY(uop_round, 0); }
static int fuzz_trunc(void)       { return FUZZ_UNARY(uop_trunc, 0); }
static int fuzz_sign(void)        { return FUZZ_UNARY(uop_sign,  0); }
static int fuzz_square(void)      { return FUZZ_UNARY(uop_square, 0); }
static int fuzz_log(void)         { return FUZZ_UNARY(uop_log,   1); }
static int fuzz_sqrt(void)        { return FUZZ_UNARY(uop_sqrt,  1); }
static int fuzz_rsqrt(void)       { return FUZZ_UNARY(uop_rsqrt, 1); }
static int fuzz_recip(void)       { return FUZZ_UNARY(uop_recip, 1); }
static int fuzz_log2(void)        { return FUZZ_UNARY(uop_log2,  1); }
static int fuzz_log10(void)       { return FUZZ_UNARY(uop_log10, 1); }
static int fuzz_exp2(void)        { return FUZZ_UNARY(uop_exp2,  0); }
static int fuzz_erf(void)         { return FUZZ_UNARY(uop_erf,   0); }
static int fuzz_sinh(void)        { return fuzz_unary_op_no_crash(uop_sinh, "uop_sinh", 0, 20); }
static int fuzz_cosh(void)        { return fuzz_unary_op_no_crash(uop_cosh, "uop_cosh", 0, 20); }
static int fuzz_asin(void) {
    int failures = 0;
    for (int trial = 0; trial < 20; trial++) {
        int shape[8], ndim;
        make_shape(shape, &ndim, 512);
        Tensor* x = make_asin_safe_tensor(shape, ndim);
        if (!x) { failures++; continue; }
        Tensor* out = uop_asin(x);
        if (!out) {
            fprintf(stderr, "    [uop_asin] NULL output\n");
            failures++;
        } else {
            if (tensor_ensure_executed(out) != 0) failures++;
            tensor_free(out);
        }
        tensor_free(x);
    }
    return (failures == 0);
}
static int fuzz_logical_not(void) { return FUZZ_UNARY(uop_logical_not, 0); }
typedef Tensor* (*BinaryOp)(Tensor*, Tensor*);

static int fuzz_binary_op_no_crash(BinaryOp op, const char* name,
                                    int use_positive, int n_trials) {
    int failures = 0;
    for (int trial = 0; trial < n_trials; trial++) {
        int shape[8], ndim;
        make_shape(shape, &ndim, 256);
        Tensor* a = use_positive
            ? make_positive_tensor(shape, ndim, 5.0f)
            : make_signed_tensor(shape, ndim, 3.0f);
        Tensor* b = use_positive
            ? make_positive_tensor(shape, ndim, 5.0f)
            : make_signed_tensor(shape, ndim, 3.0f);
        if (!a || !b) {
            if (a) tensor_free(a);
            if (b) tensor_free(b);
            failures++;
            continue;
        }

        Tensor* out = op(a, b);
        if (!out) {
            fprintf(stderr, "    [%s] NULL output\n", name);
            failures++;
        } else {
            if (tensor_ensure_executed(out) != 0) failures++;
            tensor_free(out);
        }
        tensor_free(a); tensor_free(b);
    }
    return (failures == 0);
}

#define FUZZ_BINARY(opfn, positive) \
    fuzz_binary_op_no_crash(opfn, #opfn, positive, 20)

static int fuzz_add(void)          { return FUZZ_BINARY(uop_add, 0); }
static int fuzz_sub(void)          { return FUZZ_BINARY(uop_sub, 0); }
static int fuzz_mul(void)          { return FUZZ_BINARY(uop_mul, 0); }
static int fuzz_div(void)          { return fuzz_binary_op_no_crash(uop_div, "uop_div", 1, 20); }
static int fuzz_maximum(void)      { return FUZZ_BINARY(uop_max, 0); }
static int fuzz_minimum(void)      { return FUZZ_BINARY(uop_minimum, 0); }
static int fuzz_cmplt(void)        { return FUZZ_BINARY(uop_cmplt, 0); }
static int fuzz_cmpeq(void)        { return FUZZ_BINARY(uop_cmpeq, 0); }
static int fuzz_cmpne(void)        { return FUZZ_BINARY(uop_cmpne, 0); }
static int fuzz_cmple(void)        { return FUZZ_BINARY(uop_cmple, 0); }
static int fuzz_cmpgt(void)        { return FUZZ_BINARY(uop_cmpgt, 0); }
static int fuzz_cmpge(void)        { return FUZZ_BINARY(uop_cmpge, 0); }
static int fuzz_logical_and(void)  { return FUZZ_BINARY(uop_logical_and, 0); }
static int fuzz_logical_or(void)   { return FUZZ_BINARY(uop_logical_or, 0); }
static int fuzz_logaddexp(void)    { return FUZZ_BINARY(uop_logaddexp, 0); }
static int fuzz_copysign(void)     { return FUZZ_BINARY(uop_copysign, 0); }
static int fuzz_reduce_shape(void) {
    int failures = 0;
    for (int trial = 0; trial < 30; trial++) {
        int shape[8], ndim;
        make_shape(shape, &ndim, 256);
        Tensor* x = make_signed_tensor(shape, ndim, 3.0f);
        if (!x) { failures++; continue; }

        int dim = rng_int(0, ndim);
        int dims[1] = {dim};
        ReduceParams rp = {dims, 1, false};
        Tensor* s = uop_sum(x, &rp);
        if (!s) { tensor_free(x); failures++; continue; }

        
        int expected_ndim = ndim - 1;
        if (expected_ndim < 1) expected_ndim = 1;
        if (s->ndim != expected_ndim && ndim > 1) {
            fprintf(stderr, "    [sum] expected ndim %d got %d\n", expected_ndim, s->ndim);
            failures++;
        }
        tensor_free(s);
        tensor_free(x);
    }
    return (failures == 0);
}

static int fuzz_mean_shape(void) {
    int failures = 0;
    for (int trial = 0; trial < 30; trial++) {
        int shape[8], ndim;
        make_shape(shape, &ndim, 256);
        Tensor* x = make_signed_tensor(shape, ndim, 3.0f);
        if (!x) { failures++; continue; }
        int dim = rng_int(0, ndim);
        int dims[1] = {dim};
        ReduceParams rp = {dims, 1, false};
        Tensor* m = uop_mean(x, &rp);
        if (!m) { tensor_free(x); failures++; continue; }
        tensor_free(m);
        tensor_free(x);
    }
    return (failures == 0);
}

static int fuzz_reshape_numel(void) {
    int failures = 0;
    for (int trial = 0; trial < 50; trial++) {
        
        int n = rng_int(2, 64);
        int shape1[] = {n};
        Tensor* x = make_signed_tensor(shape1, 1, 2.0f);
        if (!x) { failures++; continue; }

        
        int a = 1;
        for (int f = 2; f <= n; f++) {
            if (n % f == 0) { a = f; break; }
        }
        int b = n / a;
        int shape2[] = {a, b};
        ReshapeParams rparams = {shape2, 2};
        Tensor* r = uop_reshape(x, &rparams);
        if (!r) { tensor_free(x); failures++; continue; }

        if ((size_t)(a * b) != r->numel) {
            fprintf(stderr, "    [reshape] numel %zu != %d*%d=%d\n", r->numel, a, b, a*b);
            failures++;
        }
        tensor_free(r);
        tensor_free(x);
    }
    return (failures == 0);
}

static int fuzz_random_op_chain(void) {
    
    int failures = 0;
    UnaryOp ops[] = {uop_sin, uop_cos, uop_tanh, uop_sigmoid, uop_abs,
                     uop_neg, uop_floor, uop_ceil, uop_round, uop_square};
    int nops = (int)(sizeof(ops)/sizeof(ops[0]));

    for (int trial = 0; trial < 30; trial++) {
        int shape[] = {rng_int(1, 32)};
        Tensor* cur = make_signed_tensor(shape, 1, 1.0f);
        if (!cur) { failures++; continue; }

        int chain_len = rng_int(2, 6);
        for (int step = 0; step < chain_len; step++) {
            int op_idx = rng_int(0, nops);
            Tensor* next = ops[op_idx](cur);
            if (!next) {
                fprintf(stderr, "    [chain] NULL at step %d\n", step);
                failures++;
                tensor_free(cur);
                cur = NULL;
                break;
            }
            if (tensor_ensure_executed(next) != 0)
                failures++;
            tensor_free(cur);
            cur = next;
        }
        if (cur) {
            tensor_ensure_executed(cur);
            tensor_free(cur);
        }
        cml_reset_ir_context();
    }
    return (failures == 0);
}

static int fuzz_matmul_shapes(void) {
    int failures = 0;
    for (int trial = 0; trial < 20; trial++) {
        int M = rng_int(1, 16);
        int K = rng_int(1, 16);
        int N = rng_int(1, 16);
        int sa[] = {M, K};
        int sb[] = {K, N};
        Tensor* A = make_signed_tensor(sa, 2, 1.0f);
        Tensor* B = make_signed_tensor(sb, 2, 1.0f);
        if (!A || !B) {
            if (A) tensor_free(A);
            if (B) tensor_free(B);
            failures++;
            continue;
        }
        Tensor* C = uop_matmul(A, B);
        if (!C) {
            fprintf(stderr, "    [matmul] NULL for (%d,%d)x(%d,%d)\n", M, K, K, N);
            failures++;
        } else {
            if (C->shape[0] != M || C->shape[1] != N) {
                fprintf(stderr, "    [matmul] wrong shape: got %dx%d expected %dx%d\n",
                        C->shape[0], C->shape[1], M, N);
                failures++;
            }
            tensor_free(C);
        }
        tensor_free(A); tensor_free(B);
    }
    return (failures == 0);
}

static int fuzz_empty_tensor_safety(void) {
    
    int shape[] = {0};
    Tensor* x = tensor_empty(shape, 1, &cpu_f32);
    if (!x) return 1; 
    Tensor* out = uop_neg(x);
    if (out) tensor_free(out);
    tensor_free(x);
    return 1;
}

static int fuzz_where(void) {
    int failures = 0;
    for (int trial = 0; trial < 20; trial++) {
        int n = rng_int(2, 64);
        int shape[] = {n};
        
        float* cd = malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) cd[i] = (float)(rng_int(0, 2));
        Tensor* cond = tensor_from_data(cd, shape, 1, &cpu_f32);
        Tensor* a = make_signed_tensor(shape, 1, 3.0f);
        Tensor* b = make_signed_tensor(shape, 1, 3.0f);
        free(cd);
        if (!cond || !a || !b) {
            if (cond) tensor_free(cond);
            if (a)    tensor_free(a);
            if (b)    tensor_free(b);
            failures++;
            continue;
        }
        WhereParams wparams = {cond, a, b};
        Tensor* out = uop_where(&wparams);
        if (!out) {
            failures++;
        } else {
            tensor_ensure_executed(out);
            
            float* dc = cond->data; float* da = a->data;
            float* db = b->data;   float* do_ = out->data;
            if (tensor_ensure_executed(cond) == 0 &&
                tensor_ensure_executed(a)    == 0 &&
                tensor_ensure_executed(b)    == 0) {
                for (int i = 0; i < n; i++) {
                    float expected = (dc[i] != 0.0f) ? da[i] : db[i];
                    if (fabsf(do_[i] - expected) > 1e-5f) { failures++; break; }
                }
            }
            tensor_free(out);
        }
        tensor_free(cond); tensor_free(a); tensor_free(b);
    }
    return (failures == 0);
}

static int fuzz_fill_value(void) {
    for (int trial = 0; trial < 20; trial++) {
        int n = rng_int(1, 128);
        int shape[] = {n};
        float val = rng_float(-10.0f, 10.0f);
        Tensor* x = tensor_full(shape, 1, &cpu_f32, val);
        if (!x) return 0;
        tensor_ensure_executed(x);
        float* d = x->data;
        for (int i = 0; i < n; i++) {
            if (fabsf(d[i] - val) > 1e-6f) { tensor_free(x); return 0; }
        }
        tensor_free(x);
    }
    return 1;
}

static int fuzz_dtype_stress(void) {
    
    DType dtypes[] = {DTYPE_FLOAT32, DTYPE_INT32};
    int failures = 0;

    for (int dt = 0; dt < (int)(sizeof(dtypes)/sizeof(dtypes[0])); dt++) {
        TensorConfig cfg = {.dtype = dtypes[dt], .device = DEVICE_CPU,
                            .has_dtype = true, .has_device = true};
        int n = rng_int(4, 32);
        int shape[] = {n};
        Tensor* x = tensor_zeros(shape, 1, &cfg);
        Tensor* y = tensor_ones(shape, 1, &cfg);
        if (!x || !y) {
            if (x) tensor_free(x);
            if (y) tensor_free(y);
            failures++;
            continue;
        }
        Tensor* out = uop_add(x, y);
        if (!out) failures++;
        else tensor_free(out);
        tensor_free(x); tensor_free(y);
    }
    return (failures == 0);
}

int main(int argc, char* argv[]) {
    uint64_t seed = (argc > 1) ? (uint64_t)atoll(argv[1]) : (uint64_t)time(NULL);
    printf("Fuzzer seed: %llu\n", (unsigned long long)seed);
    rng_seed(seed);

    printf("\nUnary Op Fuzz Tests\n");
    TEST(sin); TEST(cos); TEST(tanh); TEST(sigmoid);
    TEST(exp); TEST(neg); TEST(abs);
    TEST(floor); TEST(ceil); TEST(round); TEST(trunc);
    TEST(sign); TEST(square);
    TEST(log); TEST(sqrt); TEST(rsqrt); TEST(recip);
    TEST(log2); TEST(log10); TEST(exp2);
    TEST(erf); TEST(sinh); TEST(cosh); TEST(asin);
    TEST(logical_not);

    printf("\nBinary Op Fuzz Tests\n");
    TEST(add); TEST(sub); TEST(mul); TEST(div);
    TEST(maximum); TEST(minimum);
    TEST(cmplt); TEST(cmpeq); TEST(cmpne);
    TEST(cmple); TEST(cmpgt); TEST(cmpge);
    TEST(logical_and); TEST(logical_or);
    TEST(logaddexp); TEST(copysign);

    printf("\nShape Invariant Tests\n");
    TEST(reduce_shape);
    TEST(mean_shape);
    TEST(reshape_numel);
    TEST(matmul_shapes);

    printf("\nOp Chain & Edge Case Tests\n");
    TEST(random_op_chain);
    TEST(empty_tensor_safety);
    TEST(where);
    TEST(fill_value);
    TEST(dtype_stress);

    cml_reset_ir_context();

    printf("\nResults: %d/%d passed\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
