#include "ops/ir/flamegraph.h"
#include "ops/ir/internal.h"
#include "alloc/cml_allocator.h"
#include "core/cml_flags.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* One timed kernel execution. Op/kind/phase strings are static literals (from
 * uop_type_to_string / the classifiers below), so we store the pointers. */
typedef struct {
    const char* op;     /* UOP name, e.g. "MATMUL"           */
    const char* kind;   /* coarse class, e.g. "matmul"       */
    const char* phase;  /* "forward" | "backward"            */
    long long   numel;  /* output element count (work size)  */
    double      ms;     /* measured wall time                */
} FlameSpan;

static FlameSpan* g_spans   = NULL;
static int        g_count   = 0;
static int        g_cap     = 0;
static double     g_total   = 0.0;

double cml_flame_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

int cml_flame_enabled(void) {
    static int checked = 0, on = 0;
    if (!checked) {
        const char* v = getenv("FLAMEGRAPH");
        on      = v && v[0] != '\0' && strcmp(v, "0") != 0 && strcmp(v, "false") != 0;
        /* PROFILE is the general profiling switch; it implies flamegraph capture. */
        if (cml_flag_enabled(CML_FLAG_PROFILE))
            on = 1;
        checked = 1;
    }
    return on;
}

/* Coarse kernel class. Fusion wins first: a fused node is what we most want to
 * see as a single wide bar. Otherwise classify by op family. */
static const char* flame_kind(const struct IRNode* n) {
    /* A fused chain is emitted as a UOP_FUSED_ELEMENTWISE node (its identity is
     * in the type); some paths also set the is_fused/fusion_type markers. */
    if (n->type == UOP_FUSED_ELEMENTWISE ||
        n->is_fused || n->fused_kernel || n->fusion_type != 0) return "fused";
    switch (n->type) {
        case UOP_MATMUL:
            return "matmul";
        case UOP_CONV2D: case UOP_CONV3D:
        case UOP_CONV_TRANSPOSE2D: case UOP_CONV_TRANSPOSE3D:
        case UOP_MAXPOOL2D: case UOP_AVGPOOL2D:
            return "conv";
        case UOP_SUM: case UOP_MEAN: case UOP_MAX_REDUCE:
        case UOP_PROD: case UOP_ARGMAX: case UOP_ARGMIN:
            return "reduce";
        case UOP_RESHAPE: case UOP_PERMUTE: case UOP_EXPAND:
        case UOP_SLICE: case UOP_STRIDE:
            return "movement";
        case UOP_ADD: case UOP_SUB: case UOP_MUL: case UOP_DIV:
        case UOP_NEG: case UOP_EXP: case UOP_LOG: case UOP_SQRT:
        case UOP_SIGMOID: case UOP_TANH: case UOP_ABS:
        case UOP_RELU: case UOP_SILU: case UOP_GELU:
            return "elemwise";
        default:
            return "other";
    }
}

/* A node produced by autodiff carries a link back to its forward node. */
static const char* flame_phase(const struct IRNode* n) {
    return n->forward_node ? "backward" : "forward";
}

static long long flame_numel(const struct IRNode* n) {
    if (n->output && n->output->numel > 0) return (long long)n->output->numel;
    if (n->output_shape && n->output_ndim > 0) {
        long long p = 1;
        for (int i = 0; i < n->output_ndim; i++) p *= n->output_shape[i];
        return p;
    }
    return 0;
}

void cml_flame_reset(void) {
    g_count = 0;
    g_total = 0.0;
}

/* Flush accumulated spans to flamegraph.json when the process exits, so any
 * program that executed a graph under FLAMEGRAPH=1 produces a profile without
 * needing an explicit export call. Registered once, lazily, on first record. */
static void flame_atexit_flush(void) {
    if (g_count > 0) cml_flame_export("flamegraph.json");
}

void cml_flame_record(const struct IRNode* node, double ms) {
    if (!cml_flame_enabled() || !node) return;

    static int registered = 0;
    if (!registered) { atexit(flame_atexit_flush); registered = 1; }

    /* Cap total spans so a long training run can't grow the buffer without
     * bound; beyond the cap we keep the earliest (warmup + first steps). */
    if (g_count >= 20000) return;

    if (g_count >= g_cap) {
        int ncap = g_cap ? g_cap * 2 : 256;
        FlameSpan* ns = (FlameSpan*)cml_realloc(g_spans, (size_t)ncap * sizeof(FlameSpan));
        if (!ns) return;   /* drop the span rather than corrupt state */
        g_spans = ns;
        g_cap   = ncap;
    }

    g_spans[g_count].op    = uop_type_to_string(node->type);
    g_spans[g_count].kind  = flame_kind(node);
    g_spans[g_count].phase = flame_phase(node);
    g_spans[g_count].numel = flame_numel(node);
    g_spans[g_count].ms    = ms;
    g_count++;
    g_total += ms;
}

int cml_flame_num_spans(void) { return g_count; }

int cml_flame_export(const char* path) {
    if (!cml_flame_enabled() || g_count == 0 || !path) return -1;

    FILE* f = fopen(path, "w");
    if (!f) return -1;

    fprintf(f, "{\"total_ms\":%.4f,\"num_spans\":%d,\"spans\":[", g_total, g_count);
    for (int i = 0; i < g_count; i++) {
        FlameSpan* s = &g_spans[i];
        fprintf(f, "%s{\"i\":%d,\"op\":\"%s\",\"kind\":\"%s\",\"phase\":\"%s\","
                   "\"numel\":%lld,\"ms\":%.4f}",
                i ? "," : "", i,
                s->op ? s->op : "?",
                s->kind ? s->kind : "other",
                s->phase ? s->phase : "forward",
                s->numel, s->ms);
    }
    fprintf(f, "]}\n");
    fclose(f);
    return 0;
}
