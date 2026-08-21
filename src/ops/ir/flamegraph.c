#include "ops/ir/flamegraph.h"
#include "ops/ir/internal.h"
#include "alloc/cml_allocator.h"
#include "core/cml_flags.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Timings for one kernel signature, accumulated over the whole run.
 *
 * The dashboard folds every span into a (phase -> kind -> op:numel) tree and
 * sums ms/count; it never reads a span individually and never uses their order.
 * Storing one record per execution therefore bought nothing and cost a great
 * deal: a 300-step run wrote 1.2 MB, and the 20k-span cap kept the EARLIEST
 * spans, so a long run ended up showing warmup and cold-cache iterations while
 * the steady state it was meant to profile was discarded.
 *
 * Accumulating into the same key the consumer groups by is bounded by the
 * number of distinct kernel signatures (tens), not by steps x nodes -- so the
 * whole run is covered with no cap and no truncation bias.
 *
 * Op/kind/phase are static string literals (uop_type_to_string / the
 * classifiers below), so identity comparison on the pointers is exact and
 * avoids strcmp on every executed node. */
typedef struct {
    const char* op;     /* UOP name, e.g. "MATMUL"           */
    const char* kind;   /* coarse class, e.g. "matmul"       */
    const char* phase;  /* "forward" | "backward"            */
    /* Module path the node was built under, e.g. "Sequential/Linear". This is
     * the only real hierarchy in a profile of a lazy graph -- there is no call
     * stack to sample -- so it is what gives the flame graph its depth.
     *
     * Owned, not borrowed: the graph is reset between batches, which frees every
     * node and its scope string, while these entries live until process exit.
     * NULL outside a module forward. */
    char* scope;
    /* Folded creation stack, borrowed idea from `scope` but far deeper: this is
     * what gives the flame graph its shape. Owned, for the same lifetime reason. */
    char* stack;
    long long   numel;  /* output element count (work size)  */
    double      total_ms;
    double      max_ms;   /* slowest single execution: catches outlier steps */
    /* The first execution of a signature, kept apart from the rest.
     *
     * The timing bracket in the executor wraps JIT compilation, which only
     * happens once per signature, so that first sample is compile + first-touch
     * allocation rather than kernel time -- routinely 99% of an entry's total
     * across hundreds of calls. Left mixed in, it makes every width in the flame
     * graph a measure of compilation. Kept here so a reader can subtract it and
     * see steady state, without losing the number itself. */
    double      first_ms;
    long long   count;  /* executions folded into this entry */
} FlameEntry;

/* Distinct signatures in a real graph number in the tens; this is a ceiling
 * against pathological graphs, not an expected limit. */
#define FLAME_MAX_ENTRIES 4096

static FlameEntry* g_entries = NULL;
static int         g_nentries = 0;
static int         g_cap     = 0;
static long long   g_count   = 0;   /* total executions recorded */
static double      g_total   = 0.0;

/* One record per execution, in the order they happened.
 *
 * The aggregate above answers "where did the time go"; it cannot answer "what
 * ran when", because collapsing 14k executions into 39 signatures is exactly
 * the act of discarding the time axis. A timeline needs each dispatch kept
 * separately with its start, so it is kept here -- 16 bytes each, referring
 * back to the aggregate for the op name, kind and stack rather than repeating
 * strings 14k times. */
typedef struct {
    float t0;      /* ms since the first record */
    float dur;
    int   entry;   /* index into g_entries */
} FlameSpan;

/* Caps the timeline at roughly 3MB of JSON. The aggregate stays bounded by
 * signature count no matter how long the run is; the timeline cannot, because
 * keeping the time axis means keeping every execution. Past this it truncates
 * and reports how many it dropped rather than growing without limit. */
#define FLAME_MAX_SPANS 150000
static FlameSpan* g_spans   = NULL;
static int        g_nspans  = 0;
static int        g_scap    = 0;
static int        g_dropped = 0;
static double     g_t_origin = -1.0;

double cml_flame_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

int cml_flame_enabled(void) {
    static int checked = 0, on = 0;
    /* Not cached: NO_EXPORT is checked every call so it still holds if capture
     * was enabled before the flag was read. VIZ+NO_EXPORT is rejected at init. */
    if (cml_flag_enabled(CML_FLAG_NO_EXPORT))
        return 0;
    if (!checked) {
        const char* v = getenv("FLAMEGRAPH");
        bool explicit_off = v && (strcmp(v, "0") == 0 || strcmp(v, "false") == 0);
        on      = v && v[0] != '\0' && !explicit_off;
        /* PROFILE is the general profiling switch; it implies flamegraph capture. */
        if (cml_flag_enabled(CML_FLAG_PROFILE))
            on = 1;
        /* VIZ implies it too: the dashboard has a flamegraph panel, and needing a
         * second undiscoverable env var to fill it meant most dashboard runs
         * showed an empty Kernel Studio. A VIZ run already pays graph export and
         * per-epoch metrics, so one clock_gettime per node is proportionate --
         * the zero-overhead guarantee that matters is for runs with neither set. */
        /* ...unless FLAMEGRAPH=0 says otherwise: capture costs a clock read per
         * executed node and writes a span buffer at exit, so a VIZ run that only
         * wants the graph/metrics panels must be able to decline it. */
        if (!on && !explicit_off) {
            const char* viz = getenv("VIZ");
            if (viz && viz[0] != '\0' && strcmp(viz, "0") != 0)
                on = 1;
        }
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

    /* Every uop is classified. This used to name about thirty and let the other
     * hundred and twenty fall through to "other", which put a quarter of a real
     * training profile -- LINEAR, ADAM_STEP, FILL, the comparisons -- into a
     * bucket that says nothing. "other" is now genuinely unreachable, so if it
     * ever appears it means a new uop was added without being classified. */
    switch (n->type) {
        case UOP_MATMUL: case UOP_LINEAR:
            return "matmul";
        case UOP_CONV2D: case UOP_MAXPOOL2D: case UOP_AVGPOOL2D: case UOP_CONV3D:
        case UOP_CONV_TRANSPOSE2D: case UOP_CONV_TRANSPOSE3D:
            return "conv";
        case UOP_SUM: case UOP_MAX_REDUCE: case UOP_MEAN: case UOP_PROD:
        case UOP_ARGMAX: case UOP_ARGMIN: case UOP_CUMSUM: case UOP_CUMPROD:
        case UOP_MIN_REDUCE: case UOP_VAR: case UOP_STD: case UOP_ANY:
        case UOP_ALL: case UOP_LOGSUMEXP: case UOP_CUMMAX: case UOP_CUMMIN:
        case UOP_LOGCUMSUMEXP: case UOP_TRACE:
            return "reduce";
        case UOP_RESHAPE: case UOP_PERMUTE: case UOP_EXPAND: case UOP_STRIDE:
        case UOP_SLICE: case UOP_TRIU: case UOP_TRIL: case UOP_PAD:
        case UOP_CAT: case UOP_STACK: case UOP_ROLL: case UOP_FLATTEN:
        case UOP_UNFLATTEN: case UOP_DIAG: case UOP_TILE: case UOP_REPEAT_INTERLEAVE:
        case UOP_SHRINK: case UOP_UNFOLD: case UOP_SPLIT: case UOP_CHUNK:
        case UOP_DIAGONAL: case UOP_FOLD: case UOP_IM2COL: case UOP_COL2IM:
            return "movement";
        case UOP_GATHER: case UOP_SORT: case UOP_ARGSORT: case UOP_TOPK:
        case UOP_NONZERO: case UOP_SCATTER: case UOP_ONE_HOT: case UOP_MASKED_SELECT:
        case UOP_MESHGRID: case UOP_SCATTER_ADD:
            return "index";
        case UOP_FILL: case UOP_CONST: case UOP_RAND_UNIFORM: case UOP_RAND_NORMAL:
        case UOP_ARANGE_OP: case UOP_EYE_OP: case UOP_RAND_INT: case UOP_ALLOC:
            return "init";
        case UOP_SGD_STEP: case UOP_ADAM_STEP:
            return "optim";
        case UOP_ADD: case UOP_SUB: case UOP_MUL: case UOP_DIV:
        case UOP_MAX: case UOP_CMPLT: case UOP_NEG: case UOP_EXP:
        case UOP_LOG: case UOP_SQRT: case UOP_RECIP: case UOP_ABS:
        case UOP_SIN: case UOP_COS: case UOP_TAN: case UOP_TANH:
        case UOP_SIGMOID: case UOP_POW: case UOP_SIGN: case UOP_FLOOR:
        case UOP_CEIL: case UOP_ROUND: case UOP_LOG2: case UOP_EXP2:
        case UOP_ASIN: case UOP_ACOS: case UOP_ATAN: case UOP_SQUARE:
        case UOP_RSQRT: case UOP_ERF: case UOP_CLAMP: case UOP_WHERE:
        case UOP_BITWISE_AND: case UOP_BITWISE_OR: case UOP_BITWISE_XOR: case UOP_BITWISE_NOT:
        case UOP_MASKED_FILL: case UOP_LOG10: case UOP_SINH: case UOP_COSH:
        case UOP_ASINH: case UOP_ACOSH: case UOP_ATANH: case UOP_TRUNC:
        case UOP_ISINF: case UOP_ISNAN: case UOP_ISFINITE: case UOP_LOGICAL_NOT:
        case UOP_IDIV: case UOP_MOD: case UOP_MINIMUM: case UOP_COPYSIGN:
        case UOP_LOGADDEXP: case UOP_LSHIFT: case UOP_RSHIFT: case UOP_LOGICAL_AND:
        case UOP_LOGICAL_OR: case UOP_CMPEQ: case UOP_CMPNE: case UOP_CMPLE:
        case UOP_CMPGT: case UOP_CMPGE: case UOP_ERFC: case UOP_LERP:
        case UOP_RELU: case UOP_RELU6: case UOP_HARD_SIGMOID: case UOP_HARD_TANH:
        case UOP_CELU: case UOP_QUICK_GELU: case UOP_SOFTPLUS: case UOP_SOFTSIGN:
        case UOP_LOGSIGMOID: case UOP_ELU: case UOP_SELU: case UOP_GELU:
        case UOP_LEAKY_RELU: case UOP_MISH: case UOP_SILU: case UOP_HARDSWISH:
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
    for (int i = 0; i < g_nentries; i++) {
        cml_free(g_entries[i].scope);
        cml_free(g_entries[i].stack);
        g_entries[i].scope = NULL;
        g_entries[i].stack = NULL;
    }
    g_nentries = 0;
    g_count = 0;
    g_total = 0.0;
    cml_free(g_spans);
    g_spans = NULL;
    g_nspans = g_scap = g_dropped = 0;
    g_t_origin = -1.0;
}

/* Flush accumulated spans to flamegraph.json when the process exits, so any
 * program that executed a graph under FLAMEGRAPH=1 produces a profile without
 * needing an explicit export call. Registered once, lazily, on first record. */
static void flame_atexit_flush(void) {
    if (g_count > 0) cml_flame_export("flamegraph.json");
}

/* Linear scan over the signature table. It stays short (tens of entries) and is
 * walked once per executed node, so it beats hashing on both simplicity and
 * cache behaviour at this size. */
static FlameEntry* flame_find_or_add(const char* op, const char* kind, const char* phase,
                                     long long numel, const char* scope, const char* stack) {
    for (int i = 0; i < g_nentries; i++) {
        FlameEntry* e = &g_entries[i];
        /* Scope is compared by value: two nodes under the same module path are
         * the same frame even though each owns its own copy of the string. */
        if (e->op == op && e->kind == kind && e->phase == phase && e->numel == numel &&
            ((e->scope == scope) ||
             (e->scope && scope && strcmp(e->scope, scope) == 0)) &&
            ((e->stack == stack) ||
             (e->stack && stack && strcmp(e->stack, stack) == 0)))
            return e;
    }

    if (g_nentries >= FLAME_MAX_ENTRIES)
        return NULL;

    if (g_nentries >= g_cap) {
        int ncap = g_cap ? g_cap * 2 : 64;
        FlameEntry* ne = (FlameEntry*)cml_realloc(g_entries, (size_t)ncap * sizeof(FlameEntry));
        if (!ne)
            return NULL;
        g_entries = ne;
        g_cap     = ncap;
    }

    FlameEntry* e = &g_entries[g_nentries++];
    e->op       = op;
    e->kind     = kind;
    e->phase    = phase;
    e->numel    = numel;
    e->scope    = scope ? cml_strdup(scope) : NULL;
    e->stack    = stack ? cml_strdup(stack) : NULL;
    e->total_ms = 0.0;
    e->max_ms   = 0.0;
    e->first_ms = -1.0;   /* < 0 until the first execution lands */
    e->count    = 0;
    return e;
}

void cml_flame_record(const struct IRNode* node, double ms) {
    if (!cml_flame_enabled() || !node) return;

    static int registered = 0;
    if (!registered) { atexit(flame_atexit_flush); registered = 1; }

    FlameEntry* e = flame_find_or_add(uop_type_to_string(node->type), flame_kind(node),
                                      flame_phase(node), flame_numel(node), node->scope,
                                      node->build_stack);
    if (!e)
        return;   /* signature table full: drop rather than corrupt state */

    /* Record time is the end of the dispatch, so the start is one duration back.
     * The executor's own t0 is a few hundred nanoseconds earlier; not worth
     * widening the API for. */
    const double now = cml_flame_now_ms();
    if (g_t_origin < 0.0) g_t_origin = now - ms;
    if (g_nspans < FLAME_MAX_SPANS) {
        if (g_nspans >= g_scap) {
            int ncap = g_scap ? g_scap * 2 : 4096;
            if (ncap > FLAME_MAX_SPANS) ncap = FLAME_MAX_SPANS;
            FlameSpan* ns = (FlameSpan*)cml_realloc(g_spans, (size_t)ncap * sizeof(FlameSpan));
            if (ns) { g_spans = ns; g_scap = ncap; }
        }
        if (g_nspans < g_scap) {
            FlameSpan* sp = &g_spans[g_nspans++];
            sp->t0    = (float)((now - ms) - g_t_origin);
            sp->dur   = (float)ms;
            sp->entry = (int)(e - g_entries);
        } else {
            g_dropped++;
        }
    } else {
        g_dropped++;
    }

    if (e->first_ms < 0.0)
        e->first_ms = ms;
    e->total_ms += ms;
    e->count++;
    if (ms > e->max_ms)
        e->max_ms = ms;

    g_count++;
    g_total += ms;
}

int cml_flame_num_spans(void) { return g_count; }

int cml_flame_export(const char* path) {
    if (!cml_flame_enabled() || g_count == 0 || !path) return -1;

    FILE* f = fopen(path, "w");
    if (!f) return -1;

    /* Emitted as "spans" with a count per entry so the dashboard's existing
     * fold still works; readers that predate `count` treat each entry as one
     * occurrence, which degrades to the old shape rather than breaking. */
    fprintf(f, "{\"total_ms\":%.4f,\"num_spans\":%lld,\"aggregated\":true,\"spans\":[",
            g_total, g_count);
    for (int i = 0; i < g_nentries; i++) {
        FlameEntry* e = &g_entries[i];
        fprintf(f, "%s{\"i\":%d,\"op\":\"%s\",\"kind\":\"%s\",\"phase\":\"%s\","
                   "\"scope\":\"%s\",\"stack\":\"%s\","
                   "\"numel\":%lld,\"ms\":%.4f,\"count\":%lld,\"max_ms\":%.4f,"
                   "\"first_ms\":%.4f}",
                i ? "," : "", i,
                e->op ? e->op : "?",
                e->kind ? e->kind : "other",
                e->phase ? e->phase : "forward",
                e->scope ? e->scope : "",
                e->stack ? e->stack : "",
                e->numel, e->total_ms, e->count, e->max_ms,
                e->first_ms < 0.0 ? 0.0 : e->first_ms);
    }
    fprintf(f, "]");

    /* The time axis: one triple per execution, [entry, start_ms, duration_ms],
     * in the order they ran. Indices rather than repeated op/kind/stack strings
     * -- at 14k executions that difference is megabytes. */
    fprintf(f, ",\"timeline\":{\"dropped\":%d,\"spans\":[", g_dropped);
    for (int i = 0; i < g_nspans; i++) {
        FlameSpan* sp = &g_spans[i];
        fprintf(f, "%s[%d,%.4f,%.4f]", i ? "," : "", sp->entry, (double)sp->t0, (double)sp->dur);
    }
    fprintf(f, "]}");

    fprintf(f, "}\n");
    fclose(f);
    return 0;
}
