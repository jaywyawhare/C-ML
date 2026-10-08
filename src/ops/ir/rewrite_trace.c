#include "ops/ir/rewrite_trace.h"
#include "alloc/cml_allocator.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* One recorded rule application. Strings are owned copies: emitted nodes and
 * rules can be freed or renamed after the match, so the trace cannot borrow. */
typedef struct {
    char* rule;
    char* op;
    char* from;
    char* to;
    int pass; /* index into g_passes */
    double dur_us;
} TraceEvent;

typedef struct {
    char* name;
} TracePass;

static bool g_enabled_checked = false;
static bool g_enabled         = false;

static TraceEvent* g_events = NULL;
static int g_nevents        = 0;
static int g_ecap           = 0;

static TracePass* g_passes = NULL;
static int g_npasses       = 0;
static int g_pcap          = 0;
static int g_cur_pass      = -1;

static char* dup_cstr(const char* s) {
    if (!s)
        s = "";
    size_t n  = strlen(s) + 1;
    char* out = (char*)cml_malloc(n);
    if (out)
        memcpy(out, s, n);
    return out;
}

bool cml_rewrite_trace_enabled(void) {
    if (!g_enabled_checked) {
        g_enabled_checked = true;
        /* VIZ already pays graph/kernel export; piggyback on it. REWRITE_TRACE
         * lets the trace be captured on its own. NO_EXPORT disables file output
         * everywhere, so it suppresses this too. */
        if (getenv("NO_EXPORT"))
            g_enabled = false;
        else
            g_enabled = (getenv("VIZ") != NULL) || (getenv("REWRITE_TRACE") != NULL);
    }
    return g_enabled;
}

void cml_rewrite_trace_begin_pass(const char* pass_name) {
    if (!cml_rewrite_trace_enabled())
        return;
    if (g_npasses >= g_pcap) {
        int ncap        = g_pcap ? g_pcap * 2 : 8;
        TracePass* npas = (TracePass*)cml_realloc(g_passes, (size_t)ncap * sizeof(TracePass));
        if (!npas)
            return;
        g_passes = npas;
        g_pcap   = ncap;
    }
    g_passes[g_npasses].name = dup_cstr(pass_name);
    g_cur_pass               = g_npasses;
    g_npasses++;
}

void cml_rewrite_trace_record(const char* rule, const char* op, const char* from, const char* to,
                              double dur_us) {
    if (!cml_rewrite_trace_enabled())
        return;
    if (g_cur_pass < 0)
        cml_rewrite_trace_begin_pass("rewrite");
    if (g_nevents >= g_ecap) {
        int ncap       = g_ecap ? g_ecap * 2 : 64;
        TraceEvent* ne = (TraceEvent*)cml_realloc(g_events, (size_t)ncap * sizeof(TraceEvent));
        if (!ne)
            return;
        g_events = ne;
        g_ecap   = ncap;
    }
    TraceEvent* e = &g_events[g_nevents++];
    e->rule       = dup_cstr(rule);
    e->op         = dup_cstr(op);
    e->from       = dup_cstr(from);
    e->to         = dup_cstr(to);
    e->pass       = g_cur_pass;
    e->dur_us     = dur_us;
}

void cml_rewrite_trace_reset(void) {
    for (int i = 0; i < g_nevents; i++) {
        cml_free(g_events[i].rule);
        cml_free(g_events[i].op);
        cml_free(g_events[i].from);
        cml_free(g_events[i].to);
    }
    for (int i = 0; i < g_npasses; i++)
        cml_free(g_passes[i].name);
    g_nevents  = 0;
    g_npasses  = 0;
    g_cur_pass = -1;
}

/* ── JSON export ── */

typedef struct {
    char* buf;
    size_t len;
    size_t cap;
} Buf;

static void buf_ensure(Buf* b, size_t extra) {
    if (b->len + extra + 1 <= b->cap)
        return;
    size_t ncap = b->cap ? b->cap : 256;
    while (b->len + extra + 1 > ncap)
        ncap *= 2;
    char* nb = (char*)cml_realloc(b->buf, ncap);
    if (!nb)
        return;
    b->buf = nb;
    b->cap = ncap;
}

static void buf_puts(Buf* b, const char* s) {
    size_t n = strlen(s);
    buf_ensure(b, n);
    if (!b->buf)
        return;
    memcpy(b->buf + b->len, s, n);
    b->len += n;
    b->buf[b->len] = '\0';
}

/** Append `s` as a quoted, JSON-escaped string. */
static void buf_json_str(Buf* b, const char* s) {
    if (!s)
        s = "";
    buf_puts(b, "\"");
    for (const char* p = s; *p; p++) {
        char c = *p;
        if (c == '"' || c == '\\') {
            char esc[3] = {'\\', c, 0};
            buf_puts(b, esc);
        } else if (c == '\n') {
            buf_puts(b, "\\n");
        } else if ((unsigned char)c < 0x20) {
            char u[8];
            snprintf(u, sizeof(u), "\\u%04x", c);
            buf_puts(b, u);
        } else {
            char one[2] = {c, 0};
            buf_puts(b, one);
        }
    }
    buf_puts(b, "\"");
}

char* cml_rewrite_trace_export_json(void) {
    if (g_nevents == 0)
        return NULL;

    Buf b = {0};
    buf_puts(&b, "{\"passes\":[");
    for (int p = 0; p < g_npasses; p++) {
        if (p)
            buf_puts(&b, ",");
        buf_puts(&b, "{\"name\":");
        buf_json_str(&b, g_passes[p].name);
        buf_puts(&b, ",\"matches\":[");
        int first = 1;
        for (int i = 0; i < g_nevents; i++) {
            if (g_events[i].pass != p)
                continue;
            if (!first)
                buf_puts(&b, ",");
            first = 0;
            char head[64];
            snprintf(head, sizeof(head), "{\"i\":%d,\"us\":%.3f,\"rule\":", i, g_events[i].dur_us);
            buf_puts(&b, head);
            buf_json_str(&b, g_events[i].rule);
            buf_puts(&b, ",\"op\":");
            buf_json_str(&b, g_events[i].op);
            buf_puts(&b, ",\"from\":");
            buf_json_str(&b, g_events[i].from);
            buf_puts(&b, ",\"to\":");
            buf_json_str(&b, g_events[i].to);
            buf_puts(&b, "}");
        }
        buf_puts(&b, "]}");
    }
    buf_puts(&b, "]}");
    return b.buf; /* caller frees */
}
