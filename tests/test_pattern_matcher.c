#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "test_require.h"

#include "ops/ir/pattern_matcher.h"

static void test_pattern_builder_op(void) {
    printf("  test_pattern_builder_op...");

    /* Create a pattern node that matches UOP_ADD with no inputs */
    CMLPatternNode* pat = cml_pattern_op(UOP_ADD, NULL, 0);
    REQUIRE(pat != NULL);
    REQUIRE(pat->kind == CML_PAT_OP);
    REQUIRE(pat->op_type == UOP_ADD);
    REQUIRE(pat->num_inputs == 0);

    cml_pattern_free(pat);
    printf(" PASS\n");
}

static void test_pattern_builder_capture(void) {
    printf("  test_pattern_builder_capture...");

    CMLPatternNode* cap = cml_pattern_capture("x");
    REQUIRE(cap != NULL);
    REQUIRE(cap->kind == CML_PAT_CAPTURE);
    REQUIRE(strcmp(cap->capture_name, "x") == 0);

    cml_pattern_free(cap);
    printf(" PASS\n");
}

static void test_pattern_builder_any(void) {
    printf("  test_pattern_builder_any...");

    CMLPatternNode* any = cml_pattern_any();
    REQUIRE(any != NULL);
    REQUIRE(any->kind == CML_PAT_ANY);

    cml_pattern_free(any);
    printf(" PASS\n");
}

static void test_pattern_with_inputs(void) {
    printf("  test_pattern_with_inputs...");

    /* Build pattern: ADD(capture("a"), capture("b")) */
    CMLPatternNode* cap_a = cml_pattern_capture("a");
    CMLPatternNode* cap_b = cml_pattern_capture("b");
    REQUIRE(cap_a != NULL);
    REQUIRE(cap_b != NULL);

    CMLPatternNode* inputs[2] = {cap_a, cap_b};
    CMLPatternNode* add_pat   = cml_pattern_op(UOP_ADD, inputs, 2);
    REQUIRE(add_pat != NULL);
    REQUIRE(add_pat->kind == CML_PAT_OP);
    REQUIRE(add_pat->op_type == UOP_ADD);
    REQUIRE(add_pat->num_inputs == 2);
    REQUIRE(add_pat->inputs[0] == cap_a);
    REQUIRE(add_pat->inputs[1] == cap_b);

    /* Free the root; it should handle freeing children */
    cml_pattern_free(add_pat);
    printf(" PASS\n");
}

static void test_registry_create_free(void) {
    printf("  test_registry_create_free...");

    CMLRewriteRegistry* reg = cml_rewrite_registry_create();
    REQUIRE(reg != NULL);
    REQUIRE(reg->num_rules == 0);

    cml_rewrite_registry_free(reg);
    printf(" PASS\n");
}

static void test_builtin_rules(void) {
    printf("  test_builtin_rules...");

    CMLRewriteRegistry* reg = cml_rewrite_builtin_rules();
    REQUIRE(reg != NULL);
    REQUIRE(reg->num_rules > 0);
    printf(" (found %d builtin rules) ", reg->num_rules);

    /* Verify each rule has a pattern and a name */
    for (int i = 0; i < reg->num_rules; i++) {
        REQUIRE(reg->rules[i].pattern != NULL);
        REQUIRE(reg->rules[i].name != NULL);
        REQUIRE(strlen(reg->rules[i].name) > 0);
    }

    cml_rewrite_registry_free(reg);
    printf(" PASS\n");
}

static void test_register_rule(void) {
    printf("  test_register_rule...");

    CMLRewriteRegistry* reg = cml_rewrite_registry_create();
    REQUIRE(reg != NULL);

    /* Create a dummy pattern: MUL(any, any) */
    CMLPatternNode* any1      = cml_pattern_any();
    CMLPatternNode* any2      = cml_pattern_any();
    CMLPatternNode* inputs[2] = {any1, any2};
    CMLPatternNode* pat       = cml_pattern_op(UOP_MUL, inputs, 2);
    REQUIRE(pat != NULL);

    int ret = cml_rewrite_register(reg, pat, NULL, 10, "test_rule");
    REQUIRE(ret == 0);
    REQUIRE(reg->num_rules == 1);
    REQUIRE(strcmp(reg->rules[0].name, "test_rule") == 0);
    REQUIRE(reg->rules[0].priority == 10);

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
