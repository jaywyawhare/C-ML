#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "ops/ir/pattern_matcher.h"

static void test_pattern_builder_op(void) {
    printf("  test_pattern_builder_op...");

    /* Create a pattern node that matches UOP_ADD with no inputs */
    CMLPatternNode* pat = cml_pattern_op(UOP_ADD, NULL, 0);
    assert(pat != NULL);
    assert(pat->kind == CML_PAT_OP);
    assert(pat->op_type == UOP_ADD);
    assert(pat->num_inputs == 0);

    cml_pattern_free(pat);
    printf(" PASS\n");
}

static void test_pattern_builder_capture(void) {
    printf("  test_pattern_builder_capture...");

    CMLPatternNode* cap = cml_pattern_capture("x");
    assert(cap != NULL);
    assert(cap->kind == CML_PAT_CAPTURE);
    assert(strcmp(cap->capture_name, "x") == 0);

    cml_pattern_free(cap);
    printf(" PASS\n");
}

static void test_pattern_builder_any(void) {
    printf("  test_pattern_builder_any...");

    CMLPatternNode* any = cml_pattern_any();
    assert(any != NULL);
    assert(any->kind == CML_PAT_ANY);

    cml_pattern_free(any);
    printf(" PASS\n");
}

static void test_pattern_with_inputs(void) {
    printf("  test_pattern_with_inputs...");

    /* Build pattern: ADD(capture("a"), capture("b")) */
    CMLPatternNode* cap_a = cml_pattern_capture("a");
    CMLPatternNode* cap_b = cml_pattern_capture("b");
    assert(cap_a != NULL);
    assert(cap_b != NULL);

    CMLPatternNode* inputs[2] = { cap_a, cap_b };
    CMLPatternNode* add_pat = cml_pattern_op(UOP_ADD, inputs, 2);
    assert(add_pat != NULL);
    assert(add_pat->kind == CML_PAT_OP);
    assert(add_pat->op_type == UOP_ADD);
    assert(add_pat->num_inputs == 2);
    assert(add_pat->inputs[0] == cap_a);
    assert(add_pat->inputs[1] == cap_b);

    /* Free the root; it should handle freeing children */
    cml_pattern_free(add_pat);
    printf(" PASS\n");
}

static void test_registry_create_free(void) {
    printf("  test_registry_create_free...");

    CMLRewriteRegistry* reg = cml_rewrite_registry_create();
    assert(reg != NULL);
    assert(reg->num_rules == 0);

    cml_rewrite_registry_free(reg);
    printf(" PASS\n");
}

static void test_builtin_rules(void) {
    printf("  test_builtin_rules...");

    CMLRewriteRegistry* reg = cml_rewrite_builtin_rules();
    assert(reg != NULL);
    assert(reg->num_rules > 0);
    printf(" (found %d builtin rules) ", reg->num_rules);

    /* Verify each rule has a pattern and a name */
    for (int i = 0; i < reg->num_rules; i++) {
        assert(reg->rules[i].pattern != NULL);
        assert(reg->rules[i].name != NULL);
        assert(strlen(reg->rules[i].name) > 0);
    }

    cml_rewrite_registry_free(reg);
    printf(" PASS\n");
}

static void test_register_rule(void) {
    printf("  test_register_rule...");

    CMLRewriteRegistry* reg = cml_rewrite_registry_create();
    assert(reg != NULL);

    /* Create a dummy pattern: MUL(any, any) */
    CMLPatternNode* any1 = cml_pattern_any();
    CMLPatternNode* any2 = cml_pattern_any();
    CMLPatternNode* inputs[2] = { any1, any2 };
    CMLPatternNode* pat = cml_pattern_op(UOP_MUL, inputs, 2);
    assert(pat != NULL);

    int ret = cml_rewrite_register(reg, pat, NULL, 10, "test_rule");
    assert(ret == 0);
    assert(reg->num_rules == 1);
    assert(strcmp(reg->rules[0].name, "test_rule") == 0);
    assert(reg->rules[0].priority == 10);

    cml_rewrite_registry_free(reg);
    printf(" PASS\n");
}

int main(void) {
    printf("Pattern Matcher Tests\n");

    test_pattern_builder_op();
    test_pattern_builder_capture();
    test_pattern_builder_any();
    test_pattern_with_inputs();
    test_registry_create_free();
    test_builtin_rules();
    test_register_rule();

    printf("All pattern matcher tests passed.\n");
    return 0;
}
