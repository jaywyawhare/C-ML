/* The Chrome Trace export reuses the timeline spans the flamegraph records and
 * reshapes them into complete ("X") events. What matters here: the file is
 * well-formed JSON, every event carries the keys the trace viewers require,
 * ts/dur are numeric microseconds, and one event survives per recorded
 * execution -- verified with a hand-rolled scanner, no JSON library. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cml.h"
#include "ops/ir/internal.h"
#include "ops/ir/flamegraph.h"
#include "test_harness.h"

#define NUM_RECORDS 15

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

static int braces_balanced(const char* s) {
    int depth = 0, in_str = 0;
    for (const char* p = s; *p; p++) {
        if (in_str) {
            if (*p == '\\') { if (!p[1]) return 0; p++; }
            else if (*p == '"') in_str = 0;
        } else if (*p == '"') in_str = 1;
        else if (*p == '{' || *p == '[') depth++;
        else if (*p == '}' || *p == ']') depth--;
        if (depth < 0) return 0;
    }
    return depth == 0 && !in_str;
}

/* Value of `key` parsed as a number; 1 when present and fully numeric. */
static int numeric_field(const char* obj, const char* key, double* out) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\":", key);
    const char* p = strstr(obj, pat);
    if (!p) return 0;
    p += strlen(pat);
    char* end;
    double v = strtod(p, &end);
    if (end == p) return 0;
    while (*end == ' ') end++;
    if (*end != ',' && *end != '}') return 0;
    if (out) *out = v;
    return 1;
}

int main(void) {
    setenv("FLAMEGRAPH", "1", 1);
    cml_init();
    printf("=== chrome trace export ===\n");

    CHECK("capture enabled by FLAMEGRAPH=1", cml_flame_enabled() != 0);
    cml_flame_reset();

    struct IRNode* mm  = make_node(UOP_MATMUL, 4096);
    struct IRNode* add = make_node(UOP_ADD, 4096);
    CHECK("nodes built", mm && add);
    if (!mm || !add) return 1;

    /* Durations stay under the real time between records: the recorded start
     * is (now - duration), so an oversized fake duration would place spans
     * before the timeline origin and turn ts negative -- an artifact no real
     * profile has, since a span cannot outlive the dispatch that measured it. */
    for (int i = 0; i < 10; i++) cml_flame_record(mm, 0.05);
    for (int i = 0; i < 5;  i++) cml_flame_record(add, 0.02);
    CHECK("every execution counted", cml_flame_num_spans() == NUM_RECORDS);

    const char* path = "/tmp/cml_chrome_trace_test.json";
    remove(path);
    CHECK("export succeeds", cml_flame_export_chrome_trace(path) == 0);

    FILE* f = fopen(path, "r");
    CHECK("export produced a file", f != NULL);
    if (f) {
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        char* buf = (char*)malloc((size_t)sz + 1);
        size_t got = fread(buf, 1, (size_t)sz, f); buf[got] = '\0';
        fclose(f);

        CHECK("balanced braces", braces_balanced(buf));
        CHECK("traceEvents array present", strstr(buf, "{\"traceEvents\":[") != NULL);
        CHECK("displayTimeUnit is ns", strstr(buf, "\"displayTimeUnit\":\"ns\"") != NULL);

        /* Walk the event objects inside traceEvents. */
        int events = 0, ok_keys = 1, ok_numeric = 1, ok_order = 1;
        double prev_ts = -1.0;
        for (const char* p = strstr(buf, "\"traceEvents\":["); p && *p; ) {
            p = strchr(p + 1, '{');
            if (!p) break;
            const char* end = strchr(p, '}');
            if (!end) { ok_keys = 0; break; }
            size_t len = (size_t)(end - p + 1);
            char* ev = (char*)malloc(len + 1);
            memcpy(ev, p, len); ev[len] = '\0';

            events++;
            /* Required keys of a complete event on the trace viewer axis. */
            if (!strstr(ev, "\"ph\":\"X\"") ||
                !strstr(ev, "\"name\":\"") || !strstr(ev, "\"cat\":\"") ||
                !strstr(ev, "\"args\":{\"phase\":\""))
                ok_keys = 0;
            double ts = 0.0, dur = 0.0;
            if (!numeric_field(ev, "ts", &ts) || !numeric_field(ev, "dur", &dur))
                ok_numeric = 0;
            else {
                if (ts < 0.0 || dur <= 0.0) ok_numeric = 0;
                if (ts < prev_ts) ok_order = 0;   /* spans kept in execution order */
                prev_ts = ts;
            }
            free(ev);
            p = end;
        }
        CHECK("one event per recorded execution", events == NUM_RECORDS);
        CHECK("every event carries required keys", ok_keys);
        CHECK("ts/dur numeric and in range", ok_numeric);
        CHECK("events in chronological order", ok_order);

        /* Durations survive the ms -> us conversion (MATMUL ran 0.05ms). */
        CHECK("durations converted to microseconds",
              strstr(buf, "\"dur\":50.000") != NULL);

        free(buf);
        remove(path);
    }

    cml_flame_reset();
    CHECK("no spans after reset blocks export",
          cml_flame_export_chrome_trace(path) == -1);

    free_node(mm); free_node(add);
    return TEST_SUMMARY();
}
