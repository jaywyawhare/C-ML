/* Flamegraph capture aggregates per kernel signature rather than per execution.
 * The properties that matter: repeated executions of one signature collapse to a
 * single entry carrying the summed time and an occurrence count, distinct
 * signatures stay apart, and the totals survive the folding. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cml.h"
#include "ops/ir/internal.h"
#include "ops/ir/flamegraph.h"
#include "test_harness.h"


/* Minimal node: the classifiers only read type, the fusion markers, the
 * forward_node link and the output shape. */
static struct IRNode* make_node(UOpType type, int numel) {
    struct IRNode* n = (struct IRNode*)calloc(1, sizeof(struct IRNode));
    if (!n) return NULL;
    n->type        = type;
    n->output_ndim = 1;
    n->output_shape = (int*)malloc(sizeof(int));
    if (n->output_shape) n->output_shape[0] = numel;
    return n;
}

static void free_node(struct IRNode* n) {
    if (!n) return;
    free(n->output_shape);
    free(n);
}

int main(void) {
    setenv("FLAMEGRAPH", "1", 1);
    cml_init();
    printf("=== flamegraph aggregation ===\n");

    CHECK("capture enabled by FLAMEGRAPH=1", cml_flame_enabled() != 0);
    cml_flame_reset();
    CHECK("reset clears the buffer", cml_flame_num_spans() == 0);

    struct IRNode* mm  = make_node(UOP_MATMUL, 4096);
    struct IRNode* add = make_node(UOP_ADD, 4096);
    struct IRNode* mm2 = make_node(UOP_MATMUL, 64); /* same op, different work size */
    CHECK("nodes built", mm && add && mm2);
    if (!mm || !add || !mm2) return 1;

    /* 100 executions across 3 signatures. */
    for (int i = 0; i < 100; i++) cml_flame_record(mm, 1.0);
    for (int i = 0; i < 50;  i++) cml_flame_record(add, 0.5);
    cml_flame_record(mm2, 7.0);
    cml_flame_record(mm, 9.0); /* an outlier execution of an existing signature */

    CHECK("every execution counted", cml_flame_num_spans() == 152);

    const char* path = "/tmp/cml_flamegraph_test.json";
    remove(path);
    CHECK("export succeeds", cml_flame_export(path) == 0);

    FILE* f = fopen(path, "r");
    CHECK("export produced a file", f != NULL);
    if (f) {
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        char* buf = (char*)malloc((size_t)sz + 1);
        size_t got = fread(buf, 1, (size_t)sz, f); buf[got] = '\0';
        fclose(f);

        CHECK("marked as aggregated", strstr(buf, "\"aggregated\":true") != NULL);
        CHECK("num_spans is total executions", strstr(buf, "\"num_spans\":152") != NULL);

        /* 3 signatures -> 3 entries, however many executions there were. */
        int entries = 0;
        for (const char* p = buf; (p = strstr(p, "\"op\":")); p++) entries++;
        CHECK("one entry per signature, not per execution", entries == 3);

        /* MATMUL@4096: 100 x 1.0 + one 9.0 = 109.0 over 101 executions, max 9.0. */
        CHECK("summed time per signature", strstr(buf, "\"ms\":109.0000") != NULL);
        CHECK("occurrence count per signature", strstr(buf, "\"count\":101") != NULL);
        CHECK("slowest execution retained", strstr(buf, "\"max_ms\":9.0000") != NULL);

        /* Same op at a different work size must not be folded in. */
        CHECK("work size separates signatures", strstr(buf, "\"count\":1") != NULL);

        /* Two sections with two different contracts. The aggregate is bounded by
         * signature count however long the run is -- that is what makes it cheap
         * to keep on by default. The timeline cannot be: preserving "what ran
         * when" means preserving every execution, so it is capped instead and
         * reports what it dropped. */
        const char* tl = strstr(buf, "\"timeline\"");
        CHECK("timeline section present", tl != NULL);
        CHECK("aggregate stays bounded by signatures",
              tl != NULL && (size_t)(tl - buf) < 2048);
        CHECK("timeline reports its drop count", strstr(buf, "\"dropped\":0") != NULL);

        /* One triple per execution, in order. */
        int triples = 0;
        for (const char* p2 = tl ? tl : buf; (p2 = strchr(p2, '[')); p2++) triples++;
        CHECK("timeline holds every execution", triples >= 152);

        free(buf);
        remove(path);
    }

    cml_flame_reset();
    CHECK("reset after export", cml_flame_num_spans() == 0);

    free_node(mm); free_node(add); free_node(mm2);
    return TEST_SUMMARY();
}
