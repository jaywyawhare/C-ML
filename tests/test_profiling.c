/* Profiler scopes: aggregation, nested self vs inclusive time, memory
 * attribution, the disabled fast path, and JSON export. Timing assertions are
 * structural (self <= total, min <= max), never absolute wall-clock. */
#include "backend/profiling.h"
#include "alloc/cml_allocator.h"
#include "cml.h"
#include "test_harness.h"

#include <stdio.h>
#include <stdlib.h>

/** Burn a little CPU so a scope has non-trivial, measurable duration. */
static double burn(void) {
    volatile double acc = 0.0;
    for (int i = 0; i < 200000; i++)
        acc += (double)i * 1.000001;
    return acc;
}

static int test_aggregation(void) {
    Profiler* p = profiler_create();
    for (int i = 0; i < 5; i++) {
        profiler_scope_begin(p, "step");
        burn();
        profiler_scope_end(p);
    }
    const ProfileStat* s = profiler_find_stat(p, "step");
    int ok               = s && s->count == 5 && s->total_ms >= 0.0 && s->min_ms <= s->max_ms &&
             s->self_ms <= s->total_ms + 1e-9;
    profiler_free(p);
    return ok;
}

static int test_nested_self_time(void) {
    Profiler* p = profiler_create();
    profiler_scope_begin(p, "outer");
    burn();
    profiler_scope_begin(p, "inner");
    burn();
    profiler_scope_end(p); /* inner */
    burn();
    profiler_scope_end(p); /* outer */

    const ProfileStat* outer = profiler_find_stat(p, "outer");
    const ProfileStat* inner = profiler_find_stat(p, "inner");
    int ok                   = outer && inner && outer->count == 1 && inner->count == 1 &&
             /* outer includes inner, so inclusive >= inner inclusive ... */
             outer->total_ms >= inner->total_ms - 1e-9 &&
             /* ... and outer's self time excludes inner, so it is strictly less
                than its inclusive time once a child has run. */
             outer->self_ms < outer->total_ms &&
             /* inner has no children: self == inclusive. */
             inner->self_ms <= inner->total_ms + 1e-9;
    profiler_free(p);
    return ok;
}

static int test_memory_attribution(void) {
    cml_init();
    Profiler* p    = profiler_create();
    void* keep[8]  = {0};
    const int K    = 8;
    const size_t N = 4096;

    profiler_scope_begin(p, "allocs");
    for (int i = 0; i < K; i++)
        keep[i] = cml_malloc(N);
    profiler_scope_end(p);

    const ProfileStat* s = profiler_find_stat(p, "allocs");
    int ok               = s && s->alloc_count >= K && s->net_bytes >= (long long)(K * N);

    for (int i = 0; i < K; i++)
        cml_free(keep[i]);
    profiler_free(p);
    return ok;
}

static int test_disabled_is_noop(void) {
    Profiler* p = profiler_create();
    profiler_set_enabled(p, false);
    int rc = profiler_scope_begin(p, "x");
    profiler_scope_end(p);
    int ok = rc == -1 && profiler_find_stat(p, "x") == NULL && p->num_stats == 0;
    profiler_free(p);
    return ok;
}

static int test_json_export(void) {
    Profiler* p = profiler_create();
    profiler_scope_begin(p, "a");
    burn();
    profiler_scope_end(p);
    const char* path = "/tmp/cml_prof_test.json";
    int rc           = profiler_write_json(p, path);
    FILE* f          = fopen(path, "r");
    int ok           = rc == 0 && f != NULL;
    if (f)
        fclose(f);
    remove(path);
    profiler_free(p);
    return ok;
}

static int test_unbalanced_end_is_safe(void) {
    Profiler* p = profiler_create();
    profiler_scope_end(p); /* no open scope: must not crash or underflow */
    int ok = p->stack_depth == 0;
    profiler_free(p);
    return ok;
}

int main(void) {
    printf("=== profiler scope tests ===\n");
    TEST(aggregation);
    TEST(nested_self_time);
    TEST(memory_attribution);
    TEST(disabled_is_noop);
    TEST(json_export);
    TEST(unbalanced_end_is_safe);
    return TEST_SUMMARY();
}
