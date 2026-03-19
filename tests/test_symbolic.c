#include "symbolic/symbolic.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    do { printf("  TEST: %-50s ", #name); } while(0)

#define PASS() \
    do { printf("[PASS]\n"); tests_passed++; } while(0)

#define FAIL(msg) \
    do { printf("[FAIL] %s\n", msg); tests_failed++; } while(0)

#define ASSERT_EQ(a, b) \
    do { if ((a) != (b)) { \
        char _msg[256]; \
        snprintf(_msg, sizeof(_msg), "Expected %lld, got %lld", (long long)(b), (long long)(a)); \
        FAIL(_msg); return; \
    }} while(0)

#define ASSERT_TRUE(cond) \
    do { if (!(cond)) { FAIL(#cond " is false"); return; } } while(0)

#define ASSERT_NOT_NULL(ptr) \
    do { if ((ptr) == NULL) { FAIL(#ptr " is NULL"); return; } } while(0)

static void test_const_creation(void) {
    TEST(const_creation);
    SymExpr* c = sym_const(42);
    ASSERT_NOT_NULL(c);
    ASSERT_EQ(c->type, SYM_CONST);
    ASSERT_EQ(c->const_val, 42);
    ASSERT_EQ(c->ref_count, 1);
    sym_expr_release(c);
    PASS();
}

static void test_var_creation(void) {
    TEST(var_creation);
    SymExpr* v = sym_var("batch", 1, 128);
    ASSERT_NOT_NULL(v);
    ASSERT_EQ(v->type, SYM_VAR);
    ASSERT_TRUE(strcmp(v->var.name, "batch") == 0);
    ASSERT_EQ(v->var.vmin, 1);
    ASSERT_EQ(v->var.vmax, 128);
    sym_expr_release(v);
    PASS();
}

static void test_const_folding_add(void) {
    TEST(const_folding_add);
    SymExpr* a = sym_const(10);
    SymExpr* b = sym_const(20);
    SymExpr* c = sym_add(a, b);
    ASSERT_NOT_NULL(c);
    ASSERT_EQ(c->type, SYM_CONST);
    ASSERT_EQ(c->const_val, 30);
    sym_expr_release(a);
    sym_expr_release(b);
    sym_expr_release(c);
    PASS();
}

static void test_const_folding_mul(void) {
    TEST(const_folding_mul);
    SymExpr* a = sym_const(5);
    SymExpr* b = sym_const(7);
    SymExpr* c = sym_mul(a, b);
    ASSERT_NOT_NULL(c);
    ASSERT_EQ(c->type, SYM_CONST);
    ASSERT_EQ(c->const_val, 35);
    sym_expr_release(a);
    sym_expr_release(b);
    sym_expr_release(c);
    PASS();
}

static void test_const_folding_div(void) {
    TEST(const_folding_div);
    SymExpr* a = sym_const(20);
    SymExpr* b = sym_const(4);
    SymExpr* c = sym_div(a, b);
    ASSERT_NOT_NULL(c);
    ASSERT_EQ(c->type, SYM_CONST);
    ASSERT_EQ(c->const_val, 5);
    sym_expr_release(a);
    sym_expr_release(b);
    sym_expr_release(c);
    PASS();
}

static void test_const_folding_mod(void) {
    TEST(const_folding_mod);
    SymExpr* a = sym_const(17);
    SymExpr* b = sym_const(5);
    SymExpr* c = sym_mod(a, b);
    ASSERT_NOT_NULL(c);
    ASSERT_EQ(c->type, SYM_CONST);
    ASSERT_EQ(c->const_val, 2);
    sym_expr_release(a);
    sym_expr_release(b);
    sym_expr_release(c);
    PASS();
}

static void test_const_folding_min(void) {
    TEST(const_folding_min);
    SymExpr* a = sym_const(10);
    SymExpr* b = sym_const(3);
    SymExpr* c = sym_min_expr(a, b);
    ASSERT_NOT_NULL(c);
    ASSERT_EQ(c->type, SYM_CONST);
    ASSERT_EQ(c->const_val, 3);
    sym_expr_release(a);
    sym_expr_release(b);
    sym_expr_release(c);
    PASS();
}

static void test_const_folding_max(void) {
    TEST(const_folding_max);
    SymExpr* a = sym_const(10);
    SymExpr* b = sym_const(3);
    SymExpr* c = sym_max_expr(a, b);
    ASSERT_NOT_NULL(c);
    ASSERT_EQ(c->type, SYM_CONST);
    ASSERT_EQ(c->const_val, 10);
    sym_expr_release(a);
    sym_expr_release(b);
    sym_expr_release(c);
    PASS();
}

static void test_symbolic_add_no_fold(void) {
    TEST(symbolic_add_no_fold);
    SymExpr* v = sym_var("N", 1, 100);
    SymExpr* c = sym_const(5);
    SymExpr* sum = sym_add(v, c);
    ASSERT_NOT_NULL(sum);
    ASSERT_EQ(sum->type, SYM_ADD); // Not folded since one is var
    sym_expr_release(v);
    sym_expr_release(c);
    sym_expr_release(sum);
    PASS();
}

static void test_bounds_var(void) {
    TEST(bounds_var);
    SymExpr* v = sym_var("batch", 1, 64);
    ASSERT_EQ(sym_expr_min(v), 1);
    ASSERT_EQ(sym_expr_max(v), 64);
    sym_expr_release(v);
    PASS();
}

static void test_bounds_add(void) {
    TEST(bounds_add);
    SymExpr* v = sym_var("N", 1, 100);
    SymExpr* c = sym_const(10);
    SymExpr* e = sym_add(v, c);
    ASSERT_EQ(sym_expr_min(e), 11);
    ASSERT_EQ(sym_expr_max(e), 110);
    sym_expr_release(v);
    sym_expr_release(c);
    sym_expr_release(e);
    PASS();
}

static void test_bounds_mul_positive(void) {
    TEST(bounds_mul_positive);
    SymExpr* v = sym_var("N", 2, 10);
    SymExpr* c = sym_const(3);
    SymExpr* e = sym_mul(v, c);
    ASSERT_EQ(sym_expr_min(e), 6);
    ASSERT_EQ(sym_expr_max(e), 30);
    sym_expr_release(v);
    sym_expr_release(c);
    sym_expr_release(e);
    PASS();
}

static void test_bounds_mul_negative(void) {
    TEST(bounds_mul_negative);
    SymExpr* v = sym_var("x", -5, 5);
    SymExpr* c = sym_const(-2);
    SymExpr* e = sym_mul(v, c);
    // Products: (-5)*(-2)=10, 5*(-2)=-10
    ASSERT_EQ(sym_expr_min(e), -10);
    ASSERT_EQ(sym_expr_max(e), 10);
    sym_expr_release(v);
    sym_expr_release(c);
    sym_expr_release(e);
    PASS();
}

static void test_bounds_div(void) {
    TEST(bounds_div);
    SymExpr* v = sym_var("N", 10, 100);
    SymExpr* c = sym_const(5);
    SymExpr* e = sym_div(v, c);
    ASSERT_EQ(sym_expr_min(e), 2);
    ASSERT_EQ(sym_expr_max(e), 20);
    sym_expr_release(v);
    sym_expr_release(c);
    sym_expr_release(e);
    PASS();
}

static void test_bounds_mod(void) {
    TEST(bounds_mod);
    SymExpr* v = sym_var("N", 0, 100);
    SymExpr* c = sym_const(8);
    SymExpr* e = sym_mod(v, c);
    ASSERT_EQ(sym_expr_min(e), 0);
    ASSERT_EQ(sym_expr_max(e), 7);
    sym_expr_release(v);
    sym_expr_release(c);
    sym_expr_release(e);
    PASS();
}

static void test_bounds_min(void) {
    TEST(bounds_min);
    SymExpr* a = sym_var("A", 5, 20);
    SymExpr* b = sym_var("B", 10, 30);
    SymExpr* e = sym_min_expr(a, b);
    // min's min = min(5, 10) = 5
    // min's max = min(20, 30) = 20
    ASSERT_EQ(sym_expr_min(e), 5);
    ASSERT_EQ(sym_expr_max(e), 20);
    sym_expr_release(a);
    sym_expr_release(b);
    sym_expr_release(e);
    PASS();
}

static void test_bounds_max(void) {
    TEST(bounds_max);
    SymExpr* a = sym_var("A", 5, 20);
    SymExpr* b = sym_var("B", 10, 30);
    SymExpr* e = sym_max_expr(a, b);
    // max's min = max(5, 10) = 10
    // max's max = max(20, 30) = 30
    ASSERT_EQ(sym_expr_min(e), 10);
    ASSERT_EQ(sym_expr_max(e), 30);
    sym_expr_release(a);
    sym_expr_release(b);
    sym_expr_release(e);
    PASS();
}

static void test_eval_var(void) {
    TEST(eval_var);
    SymExpr* v = sym_var("N", 1, 100);
    const char* names[] = { "N" };
    int64_t vals[] = { 42 };
    ASSERT_EQ(sym_eval(v, names, vals, 1), 42);
    sym_expr_release(v);
    PASS();
}

static void test_eval_complex(void) {
    TEST(eval_complex);
    // (N + 1) * 2
    SymExpr* n = sym_var("N", 0, 100);
    SymExpr* one = sym_const(1);
    SymExpr* two = sym_const(2);
    SymExpr* np1 = sym_add(n, one);
    SymExpr* result = sym_mul(np1, two);
    const char* names[] = { "N" };
    int64_t vals[] = { 10 };
    ASSERT_EQ(sym_eval(result, names, vals, 1), 22);
    sym_expr_release(n);
    sym_expr_release(one);
    sym_expr_release(two);
    sym_expr_release(np1);
    sym_expr_release(result);
    PASS();
}

static void test_eval_multi_var(void) {
    TEST(eval_multi_var);
    // A + B * 2
    SymExpr* a = sym_var("A", 0, 100);
    SymExpr* b = sym_var("B", 0, 50);
    SymExpr* two = sym_const(2);
    SymExpr* b2 = sym_mul(b, two);
    SymExpr* e = sym_add(a, b2);
    const char* names[] = { "A", "B" };
    int64_t vals[] = { 10, 5 };
    ASSERT_EQ(sym_eval(e, names, vals, 2), 20);
    sym_expr_release(a);
    sym_expr_release(b);
    sym_expr_release(two);
    sym_expr_release(b2);
    sym_expr_release(e);
    PASS();
}

static void test_simplify_add_zero(void) {
    TEST(simplify_add_zero);
    SymExpr* v = sym_var("x", 1, 10);
    SymExpr* zero = sym_const(0);
    SymExpr* e = sym_add(v, zero);
    SymExpr* s = sym_simplify(e);
    ASSERT_NOT_NULL(s);
    ASSERT_EQ(s->type, SYM_VAR);
    ASSERT_TRUE(strcmp(s->var.name, "x") == 0);
    sym_expr_release(v);
    sym_expr_release(zero);
    sym_expr_release(e);
    sym_expr_release(s);
    PASS();
}

static void test_simplify_mul_one(void) {
    TEST(simplify_mul_one);
    SymExpr* v = sym_var("x", 1, 10);
    SymExpr* one = sym_const(1);
    SymExpr* e = sym_mul(v, one);
    SymExpr* s = sym_simplify(e);
    ASSERT_NOT_NULL(s);
    ASSERT_EQ(s->type, SYM_VAR);
    sym_expr_release(v);
    sym_expr_release(one);
    sym_expr_release(e);
    sym_expr_release(s);
    PASS();
}

static void test_simplify_mul_zero(void) {
    TEST(simplify_mul_zero);
    SymExpr* v = sym_var("x", 1, 10);
    SymExpr* zero = sym_const(0);
    SymExpr* e = sym_mul(v, zero);
    SymExpr* s = sym_simplify(e);
    ASSERT_NOT_NULL(s);
    ASSERT_EQ(s->type, SYM_CONST);
    ASSERT_EQ(s->const_val, 0);
    sym_expr_release(v);
    sym_expr_release(zero);
    sym_expr_release(e);
    sym_expr_release(s);
    PASS();
}

static void test_simplify_div_one(void) {
    TEST(simplify_div_one);
    SymExpr* v = sym_var("x", 1, 10);
    SymExpr* one = sym_const(1);
    SymExpr* e = sym_div(v, one);
    SymExpr* s = sym_simplify(e);
    ASSERT_NOT_NULL(s);
    ASSERT_EQ(s->type, SYM_VAR);
    sym_expr_release(v);
    sym_expr_release(one);
    sym_expr_release(e);
    sym_expr_release(s);
    PASS();
}

static void test_simplify_mod_one(void) {
    TEST(simplify_mod_one);
    SymExpr* v = sym_var("x", 1, 10);
    SymExpr* one = sym_const(1);
    SymExpr* e = sym_mod(v, one);
    SymExpr* s = sym_simplify(e);
    ASSERT_NOT_NULL(s);
    ASSERT_EQ(s->type, SYM_CONST);
    ASSERT_EQ(s->const_val, 0);
    sym_expr_release(v);
    sym_expr_release(one);
    sym_expr_release(e);
    sym_expr_release(s);
    PASS();
}

static void test_simplify_nested(void) {
    TEST(simplify_nested);
    // (x + 0) * 1 should simplify to x
    SymExpr* x = sym_var("x", 1, 10);
    SymExpr* zero = sym_const(0);
    SymExpr* one = sym_const(1);
    SymExpr* xp0 = sym_add(x, zero);
    SymExpr* e = sym_mul(xp0, one);
    SymExpr* s = sym_simplify(e);
    ASSERT_NOT_NULL(s);
    ASSERT_EQ(s->type, SYM_VAR);
    ASSERT_TRUE(strcmp(s->var.name, "x") == 0);
    sym_expr_release(x);
    sym_expr_release(zero);
    sym_expr_release(one);
    sym_expr_release(xp0);
    sym_expr_release(e);
    sym_expr_release(s);
    PASS();
}

static void test_to_string(void) {
    TEST(to_string);
    SymExpr* n = sym_var("N", 1, 100);
    SymExpr* c = sym_const(2);
    SymExpr* e = sym_mul(n, c);
    char buf[256];
    sym_expr_to_string(e, buf, sizeof(buf));
    ASSERT_TRUE(strstr(buf, "N") != NULL);
    ASSERT_TRUE(strstr(buf, "*") != NULL);
    ASSERT_TRUE(strstr(buf, "2") != NULL);
    sym_expr_release(n);
    sym_expr_release(c);
    sym_expr_release(e);
    PASS();
}

static void test_shape_from_concrete(void) {
    TEST(shape_from_concrete);
    int dims[] = { 3, 32, 32 };
    SymShape* s = sym_shape_from_concrete(dims, 3);
    ASSERT_NOT_NULL(s);
    ASSERT_EQ(s->ndim, 3);
    ASSERT_TRUE(!s->dims[0].is_symbolic);
    ASSERT_EQ(s->dims[0].concrete, 3);
    ASSERT_EQ(s->dims[1].concrete, 32);
    ASSERT_EQ(s->dims[2].concrete, 32);
    sym_shape_release(s);
    PASS();
}

static void test_shape_broadcast_concrete(void) {
    TEST(shape_broadcast_concrete);
    int a_dims[] = { 3, 1, 5 };
    int b_dims[] = { 1, 4, 5 };
    SymShape* a = sym_shape_from_concrete(a_dims, 3);
    SymShape* b = sym_shape_from_concrete(b_dims, 3);
    SymShape* c = sym_shape_broadcast(a, b);
    ASSERT_NOT_NULL(c);
    ASSERT_EQ(c->ndim, 3);
    ASSERT_EQ(c->dims[0].concrete, 3);
    ASSERT_EQ(c->dims[1].concrete, 4);
    ASSERT_EQ(c->dims[2].concrete, 5);
    sym_shape_release(a);
    sym_shape_release(b);
    sym_shape_release(c);
    PASS();
}

static void test_shape_broadcast_symbolic(void) {
    TEST(shape_broadcast_symbolic);
    // Shape (N, 1) broadcast with (1, 5) -> (N, 5)
    SymExpr* n = sym_var("N", 1, 64);

    SymShape* a = (SymShape*)calloc(1, sizeof(SymShape));
    a->ndim = 2;
    a->ref_count = 1;
    a->dims = (SymDim*)calloc(2, sizeof(SymDim));
    a->dims[0] = sym_dim_symbolic(n);
    a->dims[1] = sym_dim_concrete(1);

    int b_dims[] = { 1, 5 };
    SymShape* b = sym_shape_from_concrete(b_dims, 2);

    SymShape* c = sym_shape_broadcast(a, b);
    ASSERT_NOT_NULL(c);
    ASSERT_EQ(c->ndim, 2);
    ASSERT_TRUE(c->dims[0].is_symbolic); // N
    ASSERT_TRUE(!c->dims[1].is_symbolic);
    ASSERT_EQ(c->dims[1].concrete, 5);

    sym_expr_release(n);
    sym_shape_release(a);
    sym_shape_release(b);
    sym_shape_release(c);
    PASS();
}

static void test_shape_eval(void) {
    TEST(shape_eval);
    SymExpr* n = sym_var("N", 1, 64);

    SymShape* s = (SymShape*)calloc(1, sizeof(SymShape));
    s->ndim = 2;
    s->ref_count = 1;
    s->dims = (SymDim*)calloc(2, sizeof(SymDim));
    s->dims[0] = sym_dim_symbolic(n);
    s->dims[1] = sym_dim_concrete(10);

    const char* names[] = { "N" };
    int64_t vals[] = { 32 };
    int out[2];
    int rc = sym_shape_eval(s, names, vals, 1, out);
    ASSERT_EQ(rc, 0);
    ASSERT_EQ(out[0], 32);
    ASSERT_EQ(out[1], 10);

    sym_expr_release(n);
    sym_shape_release(s);
    PASS();
}

static void test_shape_to_string(void) {
    TEST(shape_to_string);
    SymExpr* n = sym_var("N", 1, 64);

    SymShape* s = (SymShape*)calloc(1, sizeof(SymShape));
    s->ndim = 2;
    s->ref_count = 1;
    s->dims = (SymDim*)calloc(2, sizeof(SymDim));
    s->dims[0] = sym_dim_symbolic(n);
    s->dims[1] = sym_dim_concrete(10);

    char buf[256];
    sym_shape_to_string(s, buf, sizeof(buf));
    ASSERT_TRUE(strstr(buf, "N") != NULL);
    ASSERT_TRUE(strstr(buf, "10") != NULL);

    sym_expr_release(n);
    sym_shape_release(s);
    PASS();
}

static void test_ref_counting(void) {
    TEST(ref_counting);
    SymExpr* v = sym_var("x", 0, 10);
    ASSERT_EQ(v->ref_count, 1);
    sym_expr_retain(v);
    ASSERT_EQ(v->ref_count, 2);
    sym_expr_release(v);
    ASSERT_EQ(v->ref_count, 1);
    sym_expr_release(v);
    // v is freed here; no crash = pass
    PASS();
}

static void test_shape_broadcast_incompatible(void) {
    TEST(shape_broadcast_incompatible);
    int a_dims[] = { 3, 4 };
    int b_dims[] = { 5, 4 };
    SymShape* a = sym_shape_from_concrete(a_dims, 2);
    SymShape* b = sym_shape_from_concrete(b_dims, 2);
    SymShape* c = sym_shape_broadcast(a, b);
    ASSERT_TRUE(c == NULL); // Should fail - 3 vs 5 not broadcastable
    sym_shape_release(a);
    sym_shape_release(b);
    PASS();
}

static void test_div_by_zero_returns_null(void) {
    TEST(div_by_zero_returns_null);
    SymExpr* a = sym_const(10);
    SymExpr* b = sym_const(0);
    SymExpr* c = sym_div(a, b);
    ASSERT_TRUE(c == NULL);
    sym_expr_release(a);
    sym_expr_release(b);
    PASS();
}

int main(void) {
    printf("\nSymbolic Shapes Tests\n\n");

    test_const_creation();
    test_var_creation();

    test_const_folding_add();
    test_const_folding_mul();
    test_const_folding_div();
    test_const_folding_mod();
    test_const_folding_min();
    test_const_folding_max();
    test_symbolic_add_no_fold();

    test_bounds_var();
    test_bounds_add();
    test_bounds_mul_positive();
    test_bounds_mul_negative();
    test_bounds_div();
    test_bounds_mod();
    test_bounds_min();
    test_bounds_max();

    test_eval_var();
    test_eval_complex();
    test_eval_multi_var();

    test_simplify_add_zero();
    test_simplify_mul_one();
    test_simplify_mul_zero();
    test_simplify_div_one();
    test_simplify_mod_one();
    test_simplify_nested();

    test_to_string();

    test_shape_from_concrete();
    test_shape_broadcast_concrete();
    test_shape_broadcast_symbolic();
    test_shape_eval();
    test_shape_to_string();

    test_ref_counting();

    test_shape_broadcast_incompatible();
    test_div_by_zero_returns_null();

    printf("\nResults: %d passed, %d failed\n\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
