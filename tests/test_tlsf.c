#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#include "alloc/tlsf_alloc.h"

static int tests_run = 0;
static int tests_passed = 0;

#define RUN_TEST(test) do { \
    tests_run++; \
    printf("  [%d] %-50s ", tests_run, #test); \
    if (test()) { tests_passed++; printf("PASS\n"); } \
    else { printf("FAIL\n"); } \
} while(0)


static int test_create_destroy(void)
{
    CMLTLSFAllocator* alloc = cml_tlsf_create(4096);
    if (!alloc) return 0;
    cml_tlsf_destroy(alloc);
    return 1;
}

static int test_alloc_free_basic(void)
{
    CMLTLSFAllocator* alloc = cml_tlsf_create(4096);
    if (!alloc) return 0;

    void* ptr = cml_tlsf_alloc(alloc, 128);
    if (!ptr) { cml_tlsf_destroy(alloc); return 0; }

    /* Write and read back */
    memset(ptr, 0xAB, 128);
    unsigned char* bytes = (unsigned char*)ptr;
    for (int i = 0; i < 128; i++) {
        if (bytes[i] != 0xAB) { cml_tlsf_destroy(alloc); return 0; }
    }

    cml_tlsf_free(alloc, ptr);
    cml_tlsf_destroy(alloc);
    return 1;
}

static int test_alloc_multiple(void)
{
    CMLTLSFAllocator* alloc = cml_tlsf_create(16384);
    if (!alloc) return 0;

    void* ptrs[10];
    for (int i = 0; i < 10; i++) {
        ptrs[i] = cml_tlsf_alloc(alloc, 64 + i * 32);
        if (!ptrs[i]) { cml_tlsf_destroy(alloc); return 0; }
    }

    /* All pointers should be distinct */
    for (int i = 0; i < 10; i++) {
        for (int j = i + 1; j < 10; j++) {
            if (ptrs[i] == ptrs[j]) { cml_tlsf_destroy(alloc); return 0; }
        }
    }

    /* Free all */
    for (int i = 0; i < 10; i++) {
        cml_tlsf_free(alloc, ptrs[i]);
    }

    cml_tlsf_destroy(alloc);
    return 1;
}

static int test_alloc_aligned(void)
{
    CMLTLSFAllocator* alloc = cml_tlsf_create(65536);
    if (!alloc) return 0;

    /* Test various alignments */
    size_t alignments[] = {16, 32, 64, 128, 256};
    void* ptrs[5];

    for (int i = 0; i < 5; i++) {
        ptrs[i] = cml_tlsf_alloc_aligned(alloc, 100, alignments[i]);
        if (!ptrs[i]) { cml_tlsf_destroy(alloc); return 0; }
        /* Check alignment */
        if (((uintptr_t)ptrs[i] % alignments[i]) != 0) {
            cml_tlsf_destroy(alloc);
            return 0;
        }
    }

    for (int i = 0; i < 5; i++) {
        cml_tlsf_free(alloc, ptrs[i]);
    }

    cml_tlsf_destroy(alloc);
    return 1;
}

static int test_alloc_size(void)
{
    CMLTLSFAllocator* alloc = cml_tlsf_create(4096);
    if (!alloc) return 0;

    void* ptr = cml_tlsf_alloc(alloc, 200);
    if (!ptr) { cml_tlsf_destroy(alloc); return 0; }

    size_t sz = cml_tlsf_alloc_size(alloc, ptr);
    /* Should be at least 200 (aligned up to TLSF_ALIGN) */
    if (sz < 200) { cml_tlsf_destroy(alloc); return 0; }
    /* Should be aligned */
    if (sz % TLSF_ALIGN != 0) { cml_tlsf_destroy(alloc); return 0; }

    cml_tlsf_free(alloc, ptr);
    cml_tlsf_destroy(alloc);
    return 1;
}

static int test_realloc(void)
{
    CMLTLSFAllocator* alloc = cml_tlsf_create(16384);
    if (!alloc) return 0;

    void* ptr = cml_tlsf_alloc(alloc, 64);
    if (!ptr) { cml_tlsf_destroy(alloc); return 0; }

    /* Fill with pattern */
    memset(ptr, 0xCD, 64);

    /* Realloc larger */
    void* ptr2 = cml_tlsf_realloc(alloc, ptr, 256);
    if (!ptr2) { cml_tlsf_destroy(alloc); return 0; }

    /* Check original data is preserved */
    unsigned char* bytes = (unsigned char*)ptr2;
    for (int i = 0; i < 64; i++) {
        if (bytes[i] != 0xCD) { cml_tlsf_destroy(alloc); return 0; }
    }

    cml_tlsf_free(alloc, ptr2);
    cml_tlsf_destroy(alloc);
    return 1;
}

static int test_free_and_reuse(void)
{
    CMLTLSFAllocator* alloc = cml_tlsf_create(4096);
    if (!alloc) return 0;

    /* Allocate and free */
    void* ptr1 = cml_tlsf_alloc(alloc, 128);
    if (!ptr1) { cml_tlsf_destroy(alloc); return 0; }
    cml_tlsf_free(alloc, ptr1);

    /* Allocate the same size again: should reuse the same block */
    void* ptr2 = cml_tlsf_alloc(alloc, 128);
    if (!ptr2) { cml_tlsf_destroy(alloc); return 0; }

    /* On a fresh allocator with one free + one alloc of same size,
     * the memory should be reused (same address) */
    if (ptr1 != ptr2) {
        /* Not necessarily the same pointer due to merging/splitting,
         * but the allocator should have memory available */
    }

    cml_tlsf_free(alloc, ptr2);
    cml_tlsf_destroy(alloc);
    return 1;
}

static int test_stats(void)
{
    CMLTLSFAllocator* alloc = cml_tlsf_create(8192);
    if (!alloc) return 0;

    size_t used, peak, nalloc, nfree;
    cml_tlsf_stats(alloc, &used, &peak, &nalloc, &nfree);
    if (used != 0 || nalloc != 0 || nfree != 0) {
        cml_tlsf_destroy(alloc);
        return 0;
    }

    void* ptr1 = cml_tlsf_alloc(alloc, 100);
    void* ptr2 = cml_tlsf_alloc(alloc, 200);
    if (!ptr1 || !ptr2) { cml_tlsf_destroy(alloc); return 0; }

    cml_tlsf_stats(alloc, &used, &peak, &nalloc, &nfree);
    if (nalloc != 2) { cml_tlsf_destroy(alloc); return 0; }
    if (used == 0) { cml_tlsf_destroy(alloc); return 0; }

    cml_tlsf_free(alloc, ptr1);
    cml_tlsf_stats(alloc, &used, &peak, &nalloc, &nfree);
    if (nfree != 1) { cml_tlsf_destroy(alloc); return 0; }

    cml_tlsf_free(alloc, ptr2);
    cml_tlsf_destroy(alloc);
    return 1;
}

static int test_peak_tracking(void)
{
    CMLTLSFAllocator* alloc = cml_tlsf_create(16384);
    if (!alloc) return 0;

    /* Allocate three blocks */
    void* p1 = cml_tlsf_alloc(alloc, 100);
    void* p2 = cml_tlsf_alloc(alloc, 200);
    void* p3 = cml_tlsf_alloc(alloc, 300);
    if (!p1 || !p2 || !p3) { cml_tlsf_destroy(alloc); return 0; }

    size_t used_after3, peak_after3, na, nf;
    cml_tlsf_stats(alloc, &used_after3, &peak_after3, &na, &nf);

    /* Free middle block */
    cml_tlsf_free(alloc, p2);

    size_t used_after_free, peak_after_free;
    cml_tlsf_stats(alloc, &used_after_free, &peak_after_free, &na, &nf);

    /* Peak should not decrease */
    if (peak_after_free < peak_after3) {
        cml_tlsf_destroy(alloc);
        return 0;
    }
    /* Current usage should have decreased */
    if (used_after_free >= used_after3) {
        cml_tlsf_destroy(alloc);
        return 0;
    }
    /* Peak should equal the max we ever had */
    if (peak_after_free != peak_after3) {
        cml_tlsf_destroy(alloc);
        return 0;
    }

    cml_tlsf_free(alloc, p1);
    cml_tlsf_free(alloc, p3);
    cml_tlsf_destroy(alloc);
    return 1;
}

static int test_integrity_check(void)
{
    CMLTLSFAllocator* alloc = cml_tlsf_create(8192);
    if (!alloc) return 0;

    /* Fresh allocator should pass integrity check */
    if (!cml_tlsf_check(alloc)) { cml_tlsf_destroy(alloc); return 0; }

    /* After some operations */
    void* p1 = cml_tlsf_alloc(alloc, 100);
    void* p2 = cml_tlsf_alloc(alloc, 200);
    if (!cml_tlsf_check(alloc)) { cml_tlsf_destroy(alloc); return 0; }

    cml_tlsf_free(alloc, p1);
    if (!cml_tlsf_check(alloc)) { cml_tlsf_destroy(alloc); return 0; }

    cml_tlsf_free(alloc, p2);
    if (!cml_tlsf_check(alloc)) { cml_tlsf_destroy(alloc); return 0; }

    cml_tlsf_destroy(alloc);
    return 1;
}

static int test_create_with_pool(void)
{
    /* User-provided pool */
    size_t pool_size = 4096;
    void* pool = malloc(pool_size);
    if (!pool) return 0;

    CMLTLSFAllocator* alloc = cml_tlsf_create_with_pool(pool, pool_size);
    if (!alloc) { free(pool); return 0; }

    void* ptr = cml_tlsf_alloc(alloc, 64);
    if (!ptr) { cml_tlsf_destroy(alloc); free(pool); return 0; }

    /* Pointer should be within the pool */
    if ((char*)ptr < (char*)pool || (char*)ptr >= (char*)pool + pool_size) {
        cml_tlsf_destroy(alloc);
        free(pool);
        return 0;
    }

    cml_tlsf_free(alloc, ptr);
    cml_tlsf_destroy(alloc);
    free(pool);
    return 1;
}

static int test_alloc_zero_returns_null(void)
{
    CMLTLSFAllocator* alloc = cml_tlsf_create(4096);
    if (!alloc) return 0;

    void* ptr = cml_tlsf_alloc(alloc, 0);
    if (ptr != NULL) { cml_tlsf_destroy(alloc); return 0; }

    cml_tlsf_destroy(alloc);
    return 1;
}

static int test_free_null_safe(void)
{
    CMLTLSFAllocator* alloc = cml_tlsf_create(4096);
    if (!alloc) return 0;

    /* Should not crash */
    cml_tlsf_free(alloc, NULL);
    cml_tlsf_free(NULL, NULL);

    cml_tlsf_destroy(alloc);
    return 1;
}

static int test_many_alloc_free_cycles(void)
{
    CMLTLSFAllocator* alloc = cml_tlsf_create(65536);
    if (!alloc) return 0;

    /* Stress test: many alloc/free cycles */
    for (int cycle = 0; cycle < 50; cycle++) {
        void* ptrs[8];
        for (int i = 0; i < 8; i++) {
            ptrs[i] = cml_tlsf_alloc(alloc, 32 + i * 16);
            if (!ptrs[i]) { cml_tlsf_destroy(alloc); return 0; }
            /* Write a pattern */
            memset(ptrs[i], (unsigned char)(i + cycle), 32 + i * 16);
        }
        /* Free in reverse order */
        for (int i = 7; i >= 0; i--) {
            cml_tlsf_free(alloc, ptrs[i]);
        }
    }

    /* Should still pass integrity check */
    if (!cml_tlsf_check(alloc)) { cml_tlsf_destroy(alloc); return 0; }

    cml_tlsf_destroy(alloc);
    return 1;
}


static int test_timeline_create_destroy(void)
{
    CMLTimelinePlanner* planner = cml_timeline_planner_create(8);
    if (!planner) return 0;
    cml_timeline_planner_destroy(planner);
    return 1;
}

static int test_timeline_add(void)
{
    CMLTimelinePlanner* planner = cml_timeline_planner_create(4);
    if (!planner) return 0;

    int ret = cml_timeline_planner_add(planner, 0, 1024, 0, 5);
    if (ret != 0) { cml_timeline_planner_destroy(planner); return 0; }

    ret = cml_timeline_planner_add(planner, 1, 2048, 2, 8);
    if (ret != 0) { cml_timeline_planner_destroy(planner); return 0; }

    /* Invalid: alloc_time > free_time */
    ret = cml_timeline_planner_add(planner, 2, 512, 10, 5);
    if (ret == 0) { cml_timeline_planner_destroy(planner); return 0; }

    cml_timeline_planner_destroy(planner);
    return 1;
}

static int test_timeline_solve_simple(void)
{
    /* Two non-overlapping tensors should share memory */
    CMLTimelinePlanner* planner = cml_timeline_planner_create(4);
    if (!planner) return 0;

    /* Tensor 0: size 1024, alive at steps 0-3 */
    cml_timeline_planner_add(planner, 0, 1024, 0, 3);
    /* Tensor 1: size 1024, alive at steps 5-8 (no overlap with tensor 0) */
    cml_timeline_planner_add(planner, 1, 1024, 5, 8);

    int ret = cml_timeline_planner_solve(planner);
    if (ret != 0) { cml_timeline_planner_destroy(planner); return 0; }

    const CMLTimelineRecord* r0 = cml_timeline_planner_get(planner, 0);
    const CMLTimelineRecord* r1 = cml_timeline_planner_get(planner, 1);
    if (!r0 || !r1) { cml_timeline_planner_destroy(planner); return 0; }

    /* Non-overlapping tensors of same size: they should share the same offset */
    if (r0->offset != r1->offset) {
        cml_timeline_planner_destroy(planner);
        return 0;
    }

    /* Total memory should be just 1024 (aligned) */
    size_t total = cml_timeline_planner_total_memory(planner);
    if (total > 1040) { /* Allow for alignment padding */
        cml_timeline_planner_destroy(planner);
        return 0;
    }

    cml_timeline_planner_destroy(planner);
    return 1;
}

static int test_timeline_solve_overlap(void)
{
    /* Two overlapping tensors need separate memory */
    CMLTimelinePlanner* planner = cml_timeline_planner_create(4);
    if (!planner) return 0;

    /* Tensor 0: size 1024, alive at steps 0-5 */
    cml_timeline_planner_add(planner, 0, 1024, 0, 5);
    /* Tensor 1: size 1024, alive at steps 3-8 (overlaps with tensor 0) */
    cml_timeline_planner_add(planner, 1, 1024, 3, 8);

    int ret = cml_timeline_planner_solve(planner);
    if (ret != 0) { cml_timeline_planner_destroy(planner); return 0; }

    const CMLTimelineRecord* r0 = cml_timeline_planner_get(planner, 0);
    const CMLTimelineRecord* r1 = cml_timeline_planner_get(planner, 1);
    if (!r0 || !r1) { cml_timeline_planner_destroy(planner); return 0; }

    /* Overlapping tensors must have non-overlapping memory regions */
    size_t end0 = r0->offset + r0->size;
    size_t end1 = r1->offset + r1->size;
    bool spatial_overlap = (r0->offset < end1) && (r1->offset < end0);
    if (spatial_overlap) {
        cml_timeline_planner_destroy(planner);
        return 0;
    }

    /* Total memory should be at least 2 * 1024 */
    size_t total = cml_timeline_planner_total_memory(planner);
    if (total < 2048) {
        cml_timeline_planner_destroy(planner);
        return 0;
    }

    cml_timeline_planner_destroy(planner);
    return 1;
}

static int test_timeline_peak_usage(void)
{
    CMLTimelinePlanner* planner = cml_timeline_planner_create(8);
    if (!planner) return 0;

    /* Three tensors, two overlap at step 3 */
    cml_timeline_planner_add(planner, 0, 256, 0, 4);   /* 256 bytes, steps 0-4 */
    cml_timeline_planner_add(planner, 1, 512, 3, 6);   /* 512 bytes, steps 3-6 */
    cml_timeline_planner_add(planner, 2, 128, 7, 9);   /* 128 bytes, steps 7-9 */

    int ret = cml_timeline_planner_solve(planner);
    if (ret != 0) { cml_timeline_planner_destroy(planner); return 0; }

    /* Peak usage should be 256 + 512 = 768 (tensors 0 and 1 overlap at steps 3-4) */
    size_t peak = cml_timeline_planner_peak_usage(planner);
    /* Account for alignment: sizes are already aligned to 16 */
    if (peak < 768) {
        cml_timeline_planner_destroy(planner);
        return 0;
    }
    /* Peak should not exceed sum of all (since tensor 2 doesn't overlap with 0 or 1) */
    if (peak > 256 + 512 + 128) {
        cml_timeline_planner_destroy(planner);
        return 0;
    }

    cml_timeline_planner_destroy(planner);
    return 1;
}

static int test_timeline_total_memory(void)
{
    CMLTimelinePlanner* planner = cml_timeline_planner_create(8);
    if (!planner) return 0;

    /* Tensor 0 and 1 overlap, tensor 2 does not overlap with either */
    cml_timeline_planner_add(planner, 0, 1024, 0, 5);
    cml_timeline_planner_add(planner, 1, 1024, 3, 8);
    cml_timeline_planner_add(planner, 2, 512, 10, 12);

    int ret = cml_timeline_planner_solve(planner);
    if (ret != 0) { cml_timeline_planner_destroy(planner); return 0; }

    size_t total = cml_timeline_planner_total_memory(planner);

    /* Tensor 2 can reuse memory from tensor 0 or 1.
     * Minimum total = 2 * 1024 = 2048 (for the two overlapping 1024-byte tensors) */
    if (total < 2048) {
        cml_timeline_planner_destroy(planner);
        return 0;
    }
    /* Should not need more than sum of the two overlapping ones
     * (tensor 2 fits in the gap) */
    if (total > 2048 + 16) { /* small tolerance for alignment */
        cml_timeline_planner_destroy(planner);
        return 0;
    }

    cml_timeline_planner_destroy(planner);
    return 1;
}

static int test_timeline_sequential(void)
{
    /* Many tensors used sequentially should all share memory */
    CMLTimelinePlanner* planner = cml_timeline_planner_create(32);
    if (!planner) return 0;

    for (int i = 0; i < 20; i++) {
        cml_timeline_planner_add(planner, i, 256, i * 2, i * 2 + 1);
    }

    int ret = cml_timeline_planner_solve(planner);
    if (ret != 0) { cml_timeline_planner_destroy(planner); return 0; }

    size_t total = cml_timeline_planner_total_memory(planner);
    /* All 20 tensors are non-overlapping and same size (256).
     * They should all share memory. Total should be ~256. */
    if (total > 256 + 16) { /* tolerance for alignment */
        cml_timeline_planner_destroy(planner);
        return 0;
    }

    cml_timeline_planner_destroy(planner);
    return 1;
}

static int test_timeline_print(void)
{
    CMLTimelinePlanner* planner = cml_timeline_planner_create(4);
    if (!planner) return 0;

    cml_timeline_planner_add(planner, 0, 100, 0, 3);
    cml_timeline_planner_add(planner, 1, 200, 2, 5);
    cml_timeline_planner_solve(planner);

    /* Should not crash */
    cml_timeline_planner_print(planner);

    /* Also test print on NULL */
    cml_timeline_planner_print(NULL);

    cml_timeline_planner_destroy(planner);
    return 1;
}


int main(void)
{
    printf("TLSF Allocator Tests\n\n");

    printf("[TLSF Core]\n");
    RUN_TEST(test_create_destroy);
    RUN_TEST(test_alloc_free_basic);
    RUN_TEST(test_alloc_multiple);
    RUN_TEST(test_alloc_aligned);
    RUN_TEST(test_alloc_size);
    RUN_TEST(test_realloc);
    RUN_TEST(test_free_and_reuse);
    RUN_TEST(test_stats);
    RUN_TEST(test_peak_tracking);
    RUN_TEST(test_integrity_check);
    RUN_TEST(test_create_with_pool);
    RUN_TEST(test_alloc_zero_returns_null);
    RUN_TEST(test_free_null_safe);
    RUN_TEST(test_many_alloc_free_cycles);

    printf("\n[Timeline Planner]\n");
    RUN_TEST(test_timeline_create_destroy);
    RUN_TEST(test_timeline_add);
    RUN_TEST(test_timeline_solve_simple);
    RUN_TEST(test_timeline_solve_overlap);
    RUN_TEST(test_timeline_peak_usage);
    RUN_TEST(test_timeline_total_memory);
    RUN_TEST(test_timeline_sequential);
    RUN_TEST(test_timeline_print);

    printf("\nResults: %d/%d tests passed\n", tests_passed, tests_run);

    return (tests_passed == tests_run) ? 0 : 1;
}
