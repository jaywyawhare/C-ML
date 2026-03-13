/**
 * @file test_nir_compiler.c
 * @brief Tests for NIR/Mesa multi-vendor GPU compiler
 */

#include "ops/ir/nir_compiler.h"
#include "ops/uops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    printf("  Testing: %s... ", #name); \
    tests_run++; \
    if (test_##name()) { \
        printf("PASS\n"); \
        tests_passed++; \
    } else { \
        printf("FAIL\n"); \
    } \
} while(0)

/* ── Test: cml_nir_available() never crashes ─────────────────────────── */

static int test_nir_available(void) {
    /* Must not crash regardless of whether Mesa is installed. */
    bool avail = cml_nir_available();
    printf("(nir=%s) ", avail ? "yes" : "no");
    return 1; /* Always passes -- just reports availability */
}

/* ── Test: target name table ─────────────────────────────────────────── */

static int test_target_names(void) {
    /* Verify every enum value maps to a non-NULL name */
    for (int i = 0; i < NIR_TARGET_COUNT; i++) {
        const char* name = cml_nir_target_name((CMLNIRTarget)i);
        if (!name) return 0;
        if (strlen(name) == 0) return 0;
    }
    /* Out-of-range should return "unknown" */
    const char* unk = cml_nir_target_name(NIR_TARGET_COUNT);
    if (!unk) return 0;
    if (strcmp(unk, "unknown") != 0) return 0;

    const char* unk2 = cml_nir_target_name((CMLNIRTarget)-1);
    if (!unk2) return 0;
    if (strcmp(unk2, "unknown") != 0) return 0;

    return 1;
}

/* ── Test: new enum values exist ─────────────────────────────────────── */

static int test_new_targets(void) {
    /* Verify the three new targets are accessible */
    const char* nvk = cml_nir_target_name(NIR_TARGET_NVK);
    if (!nvk || strcmp(nvk, "nvk") != 0) return 0;

    const char* radv = cml_nir_target_name(NIR_TARGET_RADV);
    if (!radv || strcmp(radv, "radv") != 0) return 0;

    const char* llvmpipe = cml_nir_target_name(NIR_TARGET_LLVMPIPE);
    if (!llvmpipe || strcmp(llvmpipe, "llvmpipe") != 0) return 0;

    /* Verify count includes all 8 targets */
    if (NIR_TARGET_COUNT != 8) return 0;

    return 1;
}

/* ── Test: create / free lifecycle ───────────────────────────────────── */

static int test_create_free(void) {
    /* Create a compiler for each target -- must not crash even without Mesa */
    for (int i = 0; i < NIR_TARGET_COUNT; i++) {
        CMLNIRCompiler* c = cml_nir_compiler_create((CMLNIRTarget)i);
        if (!c) return 0;

        /* Target should match what we asked for */
        if (c->target != (CMLNIRTarget)i) {
            cml_nir_compiler_free(c);
            return 0;
        }

        cml_nir_compiler_free(c);
    }

    /* Invalid target should return NULL */
    CMLNIRCompiler* bad = cml_nir_compiler_create(NIR_TARGET_COUNT);
    if (bad != NULL) {
        cml_nir_compiler_free(bad);
        return 0;
    }

    return 1;
}

/* ── Test: create / free sets fields correctly ───────────────────────── */

static int test_create_fields(void) {
    CMLNIRCompiler* c = cml_nir_compiler_create(NIR_TARGET_RADEONSI);
    if (!c) return 0;

    if (c->target != NIR_TARGET_RADEONSI) { cml_nir_compiler_free(c); return 0; }

    /* SPIR-V output should be empty before compilation */
    if (c->spirv_output != NULL) { cml_nir_compiler_free(c); return 0; }
    if (c->spirv_size != 0)      { cml_nir_compiler_free(c); return 0; }

    /* Binary access should reflect empty state */
    if (cml_nir_binary_size(c) != 0)   { cml_nir_compiler_free(c); return 0; }
    if (cml_nir_binary_data(c) != NULL) { cml_nir_compiler_free(c); return 0; }

    cml_nir_compiler_free(c);
    return 1;
}

/* ── Test: compile without Mesa gracefully fails ─────────────────────── */

static int test_compile_unavailable(void) {
    if (cml_nir_available()) {
        printf("(skipped: Mesa available) ");
        return 1;
    }

    CMLNIRCompiler* c = cml_nir_compiler_create(NIR_TARGET_RADEONSI);
    if (!c) return 0;

    /* Compilation should fail gracefully (not crash) */
    int rc = cml_nir_compile(c, NULL);
    if (rc == 0) {
        /* Should not succeed without Mesa */
        cml_nir_compiler_free(c);
        return 0;
    }

    cml_nir_compiler_free(c);
    return 1;
}

/* ── Test: emit_uop without init returns error ───────────────────────── */

static int test_emit_uop_uninit(void) {
    /* NULL compiler */
    if (cml_nir_emit_uop(NULL, UOP_ADD, 2) == 0) return 0;

    /* Uninitialized compiler (no Mesa) */
    CMLNIRCompiler* c = cml_nir_compiler_create(NIR_TARGET_IRIS);
    if (!c) return 0;

    if (!c->initialized) {
        /* Without Mesa, emit should fail */
        if (cml_nir_emit_uop(c, UOP_ADD, 2) == 0) {
            cml_nir_compiler_free(c);
            return 0;
        }
    }

    cml_nir_compiler_free(c);
    return 1;
}

/* ── Test: binary accessors with NULL ─────────────────────────────────── */

static int test_binary_null(void) {
    if (cml_nir_binary_size(NULL) != 0)   return 0;
    if (cml_nir_binary_data(NULL) != NULL) return 0;
    return 1;
}

/* ── Test: double free safety ─────────────────────────────────────────── */

static int test_double_free(void) {
    /* cml_nir_compiler_free(NULL) should be safe */
    cml_nir_compiler_free(NULL);
    return 1;
}

/* ══════════════════════════════════════════════════════════════════════════ */

int main(void) {
    printf("=== NIR Compiler Tests ===\n\n");

    printf("Availability:\n");
    TEST(nir_available);

    printf("\nTarget names:\n");
    TEST(target_names);
    TEST(new_targets);

    printf("\nLifecycle:\n");
    TEST(create_free);
    TEST(create_fields);
    TEST(double_free);

    printf("\nCompilation:\n");
    TEST(compile_unavailable);
    TEST(emit_uop_uninit);
    TEST(binary_null);

    printf("\nTests passed: %d / %d\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
