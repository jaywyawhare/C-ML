#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ops/ir/memory_planner.h"

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  %-50s ", #name); \
    fflush(stdout); \
    if (test_##name()) { \
        tests_passed++; \
        printf("[PASS]\n"); \
    } else { \
        printf("[FAIL]\n"); \
    } \
} while(0)

/* Two buffers with non-overlapping lifetimes should share a slot. */
static int test_basic_reuse(void) {
    size_t sizes[]    = {1024, 1024};
    int first_use[]   = {0, 2};
    int last_use[]    = {1, 3};

    CMLMemoryPlan* plan = cml_memory_plan_create(2, sizes, first_use, last_use);
    if (!plan) return 0;

    int ok = (plan->total_memory == 1024);
    ok = ok && (plan->saved_memory == 1024);
    ok = ok && (plan->buffer_offsets[0] == plan->buffer_offsets[1]);

    cml_memory_plan_free(plan);
    return ok;
}

/* Two buffers with overlapping lifetimes cannot share. */
static int test_no_reuse_overlap(void) {
    size_t sizes[]    = {512, 512};
    int first_use[]   = {0, 1};
    int last_use[]    = {2, 3};

    CMLMemoryPlan* plan = cml_memory_plan_create(2, sizes, first_use, last_use);
    if (!plan) return 0;

    int ok = (plan->total_memory == 1024);
    ok = ok && (plan->saved_memory == 0);
    ok = ok && (plan->buffer_offsets[0] != plan->buffer_offsets[1]);

    cml_memory_plan_free(plan);
    return ok;
}

/* A small buffer can reuse a larger slot. */
static int test_size_fitting(void) {
    size_t sizes[]    = {2048, 512};
    int first_use[]   = {0, 2};
    int last_use[]    = {1, 3};

    CMLMemoryPlan* plan = cml_memory_plan_create(2, sizes, first_use, last_use);
    if (!plan) return 0;

    /* The 512 buffer should reuse the 2048 slot */
    int ok = (plan->total_memory == 2048);
    ok = ok && (plan->saved_memory == 512);
    ok = ok && (plan->buffer_offsets[0] == plan->buffer_offsets[1]);

    cml_memory_plan_free(plan);
    return ok;
}

/* A large buffer reuses a smaller buffer's slot (slot grows to max). */
static int test_small_then_large(void) {
    size_t sizes[]    = {512, 2048};
    int first_use[]   = {0, 2};
    int last_use[]    = {1, 3};

    CMLMemoryPlan* plan = cml_memory_plan_create(2, sizes, first_use, last_use);
    if (!plan) return 0;

    /* No overlap, so the greedy coloring (largest first) creates slot=2048
     * and the 512 buffer reuses it. Total = 2048. */
    int ok = (plan->total_memory == 2048);
    ok = ok && (plan->saved_memory == 512);

    cml_memory_plan_free(plan);
    return ok;
}

/* Pipeline pattern: A B C D where each is used only during its step. */
static int test_pipeline_reuse(void) {
    size_t sizes[]    = {1024, 1024, 1024, 1024};
    int first_use[]   = {0, 1, 2, 3};
    int last_use[]    = {0, 1, 2, 3};

    CMLMemoryPlan* plan = cml_memory_plan_create(4, sizes, first_use, last_use);
    if (!plan) return 0;

    int ok = (plan->total_memory == 1024);
    ok = ok && (plan->saved_memory == 3 * 1024);

    cml_memory_plan_free(plan);
    return ok;
}

/* All buffers alive simultaneously: no reuse possible. */
static int test_all_alive(void) {
    size_t sizes[]    = {100, 200, 300};
    int first_use[]   = {0, 0, 0};
    int last_use[]    = {5, 5, 5};

    CMLMemoryPlan* plan = cml_memory_plan_create(3, sizes, first_use, last_use);
    if (!plan) return 0;

    int ok = (plan->total_memory == 600);
    ok = ok && (plan->saved_memory == 0);
    ok = ok && (plan->peak_memory == 600);

    cml_memory_plan_free(plan);
    return ok;
}

/* Peak memory tracks the maximum concurrent live size. */
static int test_peak_memory(void) {
    /* Step 0: buf0(1000) alive
     * Step 1: buf0(1000) + buf1(2000) alive -> peak = 3000
     * Step 2: buf1(2000) alive
     * Step 3: buf2(500) alive */
    size_t sizes[]    = {1000, 2000, 500};
    int first_use[]   = {0, 1, 3};
    int last_use[]    = {1, 2, 3};

    CMLMemoryPlan* plan = cml_memory_plan_create(3, sizes, first_use, last_use);
    if (!plan) return 0;

    int ok = (plan->peak_memory == 3000);

    cml_memory_plan_free(plan);
    return ok;
}

/* Single buffer: trivial case. */
static int test_single_buffer(void) {
    size_t sizes[]    = {4096};
    int first_use[]   = {0};
    int last_use[]    = {10};

    CMLMemoryPlan* plan = cml_memory_plan_create(1, sizes, first_use, last_use);
    if (!plan) return 0;

    int ok = (plan->total_memory == 4096);
    ok = ok && (plan->saved_memory == 0);
    ok = ok && (plan->buffer_offsets[0] == 0);
    ok = ok && (plan->buffer_reuse_map[0] == -1);

    cml_memory_plan_free(plan);
    return ok;
}

/* NULL inputs return NULL. */
static int test_null_inputs(void) {
    int ok = (cml_memory_plan_create(0, NULL, NULL, NULL) == NULL);
    ok = ok && (cml_memory_plan_create(-1, NULL, NULL, NULL) == NULL);

    size_t s = 100;
    int f = 0, l = 1;
    ok = ok && (cml_memory_plan_create(1, NULL, &f, &l) == NULL);
    ok = ok && (cml_memory_plan_create(1, &s, NULL, &l) == NULL);
    ok = ok && (cml_memory_plan_create(1, &s, &f, NULL) == NULL);

    return ok;
}

/* Interleaved lifetimes with mixed sizes. */
static int test_interleaved(void) {
    /* buf0: [0,2] 1024
     * buf1: [1,3] 512
     * buf2: [3,5] 1024
     * buf3: [4,6] 256
     *
     * buf0 and buf1 overlap -> separate.
     * buf2 can't reuse buf0 (buf0 ends at 2, buf2 starts at 3 -> no overlap, can reuse).
     * buf3 starts at 4, buf1 ends at 3 -> no overlap, buf3(256) fits in buf1(512). */
    size_t sizes[]    = {1024, 512, 1024, 256};
    int first_use[]   = {0, 1, 3, 4};
    int last_use[]    = {2, 3, 5, 6};

    CMLMemoryPlan* plan = cml_memory_plan_create(4, sizes, first_use, last_use);
    if (!plan) return 0;

    /* Optimal: slot0=1024 (buf0,buf2), slot1=512 (buf1,buf3) -> total=1536 */
    int ok = (plan->total_memory == 1536);
    ok = ok && (plan->saved_memory == (1024 + 512 + 1024 + 256) - 1536);

    cml_memory_plan_free(plan);
    return ok;
}

int main(void) {
    printf("Memory Planner Tests\n");

    TEST(basic_reuse);
    TEST(no_reuse_overlap);
    TEST(size_fitting);
    TEST(small_then_large);
    TEST(pipeline_reuse);
    TEST(all_alive);
    TEST(peak_memory);
    TEST(single_buffer);
    TEST(null_inputs);
    TEST(interleaved);

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
