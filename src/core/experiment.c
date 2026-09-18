/* experiment.c — see experiment.h. Dependency-free append-only run logger. */
#include "core/experiment.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#ifndef _WIN32
#include <sys/utsname.h>
#endif
#include <unistd.h>
#include <pthread.h>

#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#define mkdir(path, mode) _mkdir(path)          /* Win32 mkdir takes no mode */
#ifndef _SC_CLK_TCK
#define _SC_CLK_TCK 2
#endif
#define sysconf(x) 100L                          /* no sysconf; CPU% uses /proc */
#endif

struct CMLRun {
    char            dir[1024];
    char            id[128];
    FILE*           events;
    FILE*           console;
    pthread_mutex_t lock;
    /* config assembled as inner JSON pairs: `"k":v,"k2":v2` */
    char*  config_pairs;
    size_t config_len, config_cap;
    char*  summary_pairs;                 /* same, for the run-table summary */
    size_t summary_len, summary_cap;
    char*  tags;                          /* inner JSON array: "a","b" */
    size_t tags_len, tags_cap;
    char*  notes;                         /* raw, escaped at write time */
    /* system-metric CPU tracking */
    double last_wall;
    double last_cpu;
    double start_time;
    char   project[256];
    char   name[256];
    char   sweep[128];
    char   host[256];
    char   os[256];
    char   git[64];
    char   cmd[512];
};

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void mkdirs(const char* path) {
    char tmp[1024];
    size_t n = strlen(path);
    if (n >= sizeof(tmp)) return;
    memcpy(tmp, path, n + 1);
    for (char* p = tmp + 1; *p; p++) {
        if (*p == '/') { *p = '\0'; mkdir(tmp, 0755); *p = '/'; }
    }
    mkdir(tmp, 0755);
}

/* Append a JSON-escaped copy of `s` to file. */
static void json_escape(FILE* f, const char* s) {
    if (!s) { fputs("null", f); return; }
    fputc('"', f);
    for (const unsigned char* p = (const unsigned char*)s; *p; p++) {
        switch (*p) {
        case '"':  fputs("\\\"", f); break;
        case '\\': fputs("\\\\", f); break;
        case '\n': fputs("\\n", f);  break;
        case '\r': fputs("\\r", f);  break;
        case '\t': fputs("\\t", f);  break;
        default:
            if (*p < 0x20) fprintf(f, "\\u%04x", *p);
            else fputc(*p, f);
        }
    }
    fputc('"', f);
}

static void write_meta(CMLRun* run, const char* status) {
    char path[1100];
    snprintf(path, sizeof(path), "%s/meta.json", run->dir);
    FILE* f = fopen(path, "wb");
    if (!f) return;
    double end = now_sec();
    int done = !(strcmp(status, "running") == 0);
    fputs("{\n", f);
    fputs("  \"id\": ", f);      json_escape(f, run->id);      fputs(",\n", f);
    fputs("  \"project\": ", f); json_escape(f, run->project); fputs(",\n", f);
    fputs("  \"name\": ", f);    json_escape(f, run->name);    fputs(",\n", f);
    fputs("  \"sweep\": ", f);   json_escape(f, run->sweep[0] ? run->sweep : NULL); fputs(",\n", f);
    fputs("  \"status\": ", f);  json_escape(f, status);       fputs(",\n", f);
    fputs("  \"host\": ", f);    json_escape(f, run->host);    fputs(",\n", f);
    fputs("  \"os\": ", f);      json_escape(f, run->os);      fputs(",\n", f);
    fputs("  \"git\": ", f);     json_escape(f, run->git[0] ? run->git : NULL); fputs(",\n", f);
    fputs("  \"cmd\": ", f);     json_escape(f, run->cmd[0] ? run->cmd : NULL); fputs(",\n", f);
    fputs("  \"notes\": ", f);   json_escape(f, run->notes);   fputs(",\n", f);
    fprintf(f, "  \"tags\": [%s],\n", run->tags ? run->tags : "");
    fprintf(f, "  \"start\": %.3f,\n", run->start_time);
    fprintf(f, "  \"end\": %.3f,\n", done ? end : 0.0);
    fprintf(f, "  \"duration\": %.3f,\n", done ? (end - run->start_time) : 0.0);
    fprintf(f, "  \"summary\": {%s},\n", run->summary_pairs ? run->summary_pairs : "");
    fprintf(f, "  \"config\": {%s}\n", run->config_pairs ? run->config_pairs : "");
    fputs("}\n", f);
    fclose(f);
}

/* Append `frag` (already formatted) to a growing buffer. */
static void sappend(char** buf, size_t* len, size_t* cap, const char* frag) {
    size_t fl = strlen(frag);
    size_t need = *len + fl + 1;
    if (need > *cap) {
        size_t c = *cap ? *cap : 256;
        while (c < need) c *= 2;
        char* np = realloc(*buf, c);
        if (!np) return;
        *buf = np;
        *cap = c;
        if (*len == 0) (*buf)[0] = '\0';
    }
    memcpy(*buf + *len, frag, fl + 1);
    *len += fl;
}

static void config_append(CMLRun* run, const char* key, const char* value_json) {
    if (!key || !value_json) return;
    char pair[512];
    snprintf(pair, sizeof(pair), "%s\"%s\":%s", (run->config_len > 0 ? "," : ""), key, value_json);
    sappend(&run->config_pairs, &run->config_len, &run->config_cap, pair);
}

/* Read the current git commit (short) from .git; empty on failure. */
static void read_git(char* out, size_t n) {
    out[0] = '\0';
    FILE* h = fopen(".git/HEAD", "r");
    if (!h) return;
    char line[256];
    if (fgets(line, sizeof(line), h)) {
        char* nl = strchr(line, '\n'); if (nl) *nl = '\0';
        if (strncmp(line, "ref: ", 5) == 0) {
            char path[320];
            snprintf(path, sizeof(path), ".git/%.250s", line + 5);
            FILE* rf = fopen(path, "r");
            if (rf) { if (fgets(out, (int)n, rf)) { char* x = strchr(out, '\n'); if (x) *x = '\0'; } fclose(rf); }
        } else {
            snprintf(out, n, "%.*s", (int)n - 1, line);
        }
    }
    fclose(h);
    out[12] = '\0'; /* short sha */
}

CMLRun* cml_exp_run_init(const char* project, const char* name, const char* config_json) {
    CMLRun* run = calloc(1, sizeof(CMLRun));
    if (!run) return NULL;
    pthread_mutex_init(&run->lock, NULL);

    static unsigned long counter = 0;
    unsigned long c = __sync_fetch_and_add(&counter, 1);
    double t = now_sec();
    snprintf(run->id, sizeof(run->id), "%ld-%d-%lu", (long)t, (int)getpid(), c);

    snprintf(run->project, sizeof(run->project), "%s", project ? project : "default");
    snprintf(run->name, sizeof(run->name), "%s", name ? name : run->id);

    const char* root = getenv("CML_EXP_DIR");
    if (!root || !root[0]) root = ".cml/experiments";
    snprintf(run->dir, sizeof(run->dir), "%s/runs/%s", root, run->id);
    mkdirs(run->dir);
    char media[1100];
    snprintf(media, sizeof(media), "%s/media", run->dir);
    mkdirs(media);

    char epath[1100];
    snprintf(epath, sizeof(epath), "%s/events.jsonl", run->dir);
    run->events = fopen(epath, "wb");
    char cpath[1100];
    snprintf(cpath, sizeof(cpath), "%s/console.log", run->dir);
    run->console = fopen(cpath, "wb");

    /* Run metadata (git commit, host, OS) — the run-overview page. */
#ifdef _WIN32
    { DWORD _sz = (DWORD)sizeof(run->host);
      if (!GetComputerNameA(run->host, &_sz)) run->host[0] = '\0'; }
#else
    if (gethostname(run->host, sizeof(run->host)) != 0) run->host[0] = '\0';
#endif
#ifdef _WIN32
    snprintf(run->os, sizeof(run->os), "Windows");
#else
    struct utsname un;
    if (uname(&un) == 0) snprintf(run->os, sizeof(run->os), "%s %s", un.sysname, un.release);
#endif
    read_git(run->git, sizeof(run->git));
    /* Capture the launching command from /proc/self/cmdline (NUL-separated). */
    FILE* cl = fopen("/proc/self/cmdline", "rb");
    if (cl) {
        size_t n = fread(run->cmd, 1, sizeof(run->cmd) - 1, cl);
        for (size_t i = 0; i + 1 < n; i++) if (run->cmd[i] == '\0') run->cmd[i] = ' ';
        run->cmd[n] = '\0';
        fclose(cl);
    }

    /* Seed config from the init object (strip outer braces, keep inner pairs). */
    if (config_json) {
        const char* s = config_json;
        while (*s == ' ' || *s == '{') s++;
        const char* e = s + strlen(s);
        while (e > s && (e[-1] == ' ' || e[-1] == '}' || e[-1] == '\n')) e--;
        if (e > s) {
            size_t len = (size_t)(e - s);
            run->config_pairs = malloc(len + 1);
            if (run->config_pairs) {
                memcpy(run->config_pairs, s, len);
                run->config_pairs[len] = '\0';
                run->config_len = len;
                run->config_cap = len + 1;
            }
        }
    }

    run->last_wall  = t;
    run->last_cpu   = 0.0;
    run->start_time = t;
    write_meta(run, "running");
    return run;
}

void cml_exp_set_sweep(CMLRun* run, const char* sweep_id) {
    if (!run || !sweep_id) return;
    snprintf(run->sweep, sizeof(run->sweep), "%s", sweep_id);
    write_meta(run, "running");
}

void cml_exp_config_set(CMLRun* run, const char* key, const char* value_json) {
    if (!run) return;
    pthread_mutex_lock(&run->lock);
    config_append(run, key, value_json);
    write_meta(run, "running");
    pthread_mutex_unlock(&run->lock);
}

void cml_exp_log_scalar(CMLRun* run, const char* name, long step, double value) {
    if (!run || !run->events) return;
    pthread_mutex_lock(&run->lock);
    fprintf(run->events, "{\"t\":\"scalar\",\"name\":");
    json_escape(run->events, name);
    fprintf(run->events, ",\"step\":%ld,\"value\":%.9g,\"wall\":%.3f}\n", step, value, now_sec());
    fflush(run->events);
    pthread_mutex_unlock(&run->lock);
}

void cml_exp_log_histogram(CMLRun* run, const char* name, long step,
                           const float* values, size_t n, int bins) {
    if (!run || !run->events || !values || n == 0) return;
    if (bins < 1) bins = 30;
    if (bins > 256) bins = 256;
    float mn = values[0], mx = values[0];
    for (size_t i = 1; i < n; i++) {
        if (values[i] < mn) mn = values[i];
        if (values[i] > mx) mx = values[i];
    }
    double range = (double)mx - (double)mn;
    if (range <= 0) range = 1e-9;
    long* counts = calloc((size_t)bins, sizeof(long));
    if (!counts) return;
    for (size_t i = 0; i < n; i++) {
        int b = (int)(((double)values[i] - mn) / range * bins);
        if (b < 0) b = 0;
        if (b >= bins) b = bins - 1;
        counts[b]++;
    }
    pthread_mutex_lock(&run->lock);
    fprintf(run->events, "{\"t\":\"histogram\",\"name\":");
    json_escape(run->events, name);
    fprintf(run->events, ",\"step\":%ld,\"min\":%.6g,\"max\":%.6g,\"counts\":[", step, mn, mx);
    for (int b = 0; b < bins; b++) fprintf(run->events, "%s%ld", b ? "," : "", counts[b]);
    fprintf(run->events, "],\"wall\":%.3f}\n", now_sec());
    fflush(run->events);
    pthread_mutex_unlock(&run->lock);
    free(counts);
}

void cml_exp_log_system(CMLRun* run, long step) {
    if (!run || !run->events) return;
    double rss_mb = 0.0, cpu_pct = 0.0;
    FILE* st = fopen("/proc/self/status", "r");
    if (st) {
        char line[256];
        while (fgets(line, sizeof(line), st)) {
            long kb;
            if (sscanf(line, "VmRSS: %ld kB", &kb) == 1) { rss_mb = kb / 1024.0; break; }
        }
        fclose(st);
    }
    FILE* stat = fopen("/proc/self/stat", "r");
    if (stat) {
        /* field 14 = utime, 15 = stime (clock ticks) */
        long utime = 0, stime = 0; char buf[1024];
        if (fgets(buf, sizeof(buf), stat)) {
            char* p = strrchr(buf, ')');
            if (p) {
                int f = 0; p += 2;
                for (char* tok = strtok(p, " "); tok; tok = strtok(NULL, " ")) {
                    f++;
                    if (f == 12) utime = atol(tok);
                    else if (f == 13) { stime = atol(tok); break; }
                }
            }
        }
        fclose(stat);
        double hz = (double)sysconf(_SC_CLK_TCK);
        double cpu = (utime + stime) / (hz > 0 ? hz : 100.0);
        double wall = now_sec();
        double dw = wall - run->last_wall;
        if (dw > 1e-6) cpu_pct = (cpu - run->last_cpu) / dw * 100.0;
        run->last_wall = wall;
        run->last_cpu  = cpu;
    }
    pthread_mutex_lock(&run->lock);
    fprintf(run->events,
            "{\"t\":\"system\",\"step\":%ld,\"cpu_pct\":%.2f,\"rss_mb\":%.2f,\"wall\":%.3f}\n",
            step, cpu_pct, rss_mb, now_sec());
    fflush(run->events);
    pthread_mutex_unlock(&run->lock);
}

void cml_exp_log_image(CMLRun* run, const char* name, long step,
                       const unsigned char* rgb, int width, int height) {
    if (!run || !run->events || !rgb || width <= 0 || height <= 0) return;
    char fname[256];
    snprintf(fname, sizeof(fname), "media/%.200s_%ld.rgb", name, step);
    char full[1282]; /* dir(1024) + "/" + fname(256) */
    snprintf(full, sizeof(full), "%.1000s/%.255s", run->dir, fname);
    FILE* img = fopen(full, "wb");
    if (img) {
        fwrite(rgb, 1, (size_t)width * (size_t)height * 3, img);
        fclose(img);
    }
    pthread_mutex_lock(&run->lock);
    fprintf(run->events, "{\"t\":\"image\",\"name\":");
    json_escape(run->events, name);
    fprintf(run->events, ",\"step\":%ld,\"w\":%d,\"h\":%d,\"path\":", step, width, height);
    json_escape(run->events, fname);
    fprintf(run->events, ",\"wall\":%.3f}\n", now_sec());
    fflush(run->events);
    pthread_mutex_unlock(&run->lock);
}

void cml_exp_log_table(CMLRun* run, const char* name, long step, const char* csv) {
    if (!run || !run->events || !csv) return;
    pthread_mutex_lock(&run->lock);
    fprintf(run->events, "{\"t\":\"table\",\"name\":");
    json_escape(run->events, name);
    fprintf(run->events, ",\"step\":%ld,\"csv\":", step);
    json_escape(run->events, csv);
    fprintf(run->events, ",\"wall\":%.3f}\n", now_sec());
    fflush(run->events);
    pthread_mutex_unlock(&run->lock);
}

void cml_exp_log_artifact(CMLRun* run, const char* name, const char* type,
                          const char* path, const char* aliases) {
    if (!run || !run->events) return;
    unsigned long long hash = 1469598103934665603ULL; /* FNV-1a 64 */
    long size = 0;
    FILE* f = fopen(path ? path : "", "rb");
    if (f) {
        int ch;
        while ((ch = fgetc(f)) != EOF) { hash ^= (unsigned char)ch; hash *= 1099511628211ULL; size++; }
        fclose(f);
    }
    pthread_mutex_lock(&run->lock);
    fprintf(run->events, "{\"t\":\"artifact\",\"name\":");
    json_escape(run->events, name);
    fprintf(run->events, ",\"atype\":");
    json_escape(run->events, type);
    fprintf(run->events, ",\"path\":");
    json_escape(run->events, path);
    fprintf(run->events, ",\"aliases\":");
    json_escape(run->events, aliases);
    fprintf(run->events, ",\"size\":%ld,\"hash\":\"%016llx\",\"wall\":%.3f}\n", size, hash, now_sec());
    fflush(run->events);
    pthread_mutex_unlock(&run->lock);
}

void cml_exp_add_tag(CMLRun* run, const char* tag) {
    if (!run || !tag) return;
    pthread_mutex_lock(&run->lock);
    char frag[160];
    snprintf(frag, sizeof(frag), "%s\"%s\"", run->tags_len ? "," : "", tag);
    sappend(&run->tags, &run->tags_len, &run->tags_cap, frag);
    write_meta(run, "running");
    pthread_mutex_unlock(&run->lock);
}

void cml_exp_set_notes(CMLRun* run, const char* notes) {
    if (!run) return;
    pthread_mutex_lock(&run->lock);
    free(run->notes);
    run->notes = notes ? strdup(notes) : NULL;
    write_meta(run, "running");
    pthread_mutex_unlock(&run->lock);
}

void cml_exp_log_console(CMLRun* run, const char* line) {
    if (!run || !run->console || !line) return;
    pthread_mutex_lock(&run->lock);
    fprintf(run->console, "%s\n", line);
    fflush(run->console);
    pthread_mutex_unlock(&run->lock);
}

void cml_exp_summary_set(CMLRun* run, const char* key, double value) {
    if (!run || !key) return;
    pthread_mutex_lock(&run->lock);
    char frag[160];
    snprintf(frag, sizeof(frag), "%s\"%s\":%.9g", run->summary_len ? "," : "", key, value);
    sappend(&run->summary_pairs, &run->summary_len, &run->summary_cap, frag);
    pthread_mutex_unlock(&run->lock);
}

void cml_exp_alert(CMLRun* run, const char* level, const char* message) {
    if (!run || !run->events) return;
    pthread_mutex_lock(&run->lock);
    fprintf(run->events, "{\"t\":\"alert\",\"level\":");
    json_escape(run->events, level ? level : "info");
    fprintf(run->events, ",\"message\":");
    json_escape(run->events, message);
    fprintf(run->events, ",\"wall\":%.3f}\n", now_sec());
    fflush(run->events);
    pthread_mutex_unlock(&run->lock);
    if (run->console) { fprintf(run->console, "[alert:%s] %s\n", level ? level : "info", message ? message : ""); fflush(run->console); }
}

void cml_exp_log_curve(CMLRun* run, const char* name, const char* xlabel,
                       const char* ylabel, const float* xs, const float* ys, size_t n) {
    if (!run || !run->events || !xs || !ys || n == 0) return;
    pthread_mutex_lock(&run->lock);
    fprintf(run->events, "{\"t\":\"curve\",\"name\":");
    json_escape(run->events, name);
    fprintf(run->events, ",\"xlabel\":");
    json_escape(run->events, xlabel);
    fprintf(run->events, ",\"ylabel\":");
    json_escape(run->events, ylabel);
    fprintf(run->events, ",\"xs\":[");
    for (size_t i = 0; i < n; i++) fprintf(run->events, "%s%.5g", i ? "," : "", xs[i]);
    fprintf(run->events, "],\"ys\":[");
    for (size_t i = 0; i < n; i++) fprintf(run->events, "%s%.5g", i ? "," : "", ys[i]);
    fprintf(run->events, "],\"wall\":%.3f}\n", now_sec());
    fflush(run->events);
    pthread_mutex_unlock(&run->lock);
}

void cml_exp_run_finish(CMLRun* run, const char* status) {
    if (!run) return;
    write_meta(run, status ? status : "finished");
    if (run->events) { fclose(run->events); run->events = NULL; }
    if (run->console) { fclose(run->console); run->console = NULL; }
    pthread_mutex_destroy(&run->lock);
    free(run->config_pairs);
    free(run->summary_pairs);
    free(run->tags);
    free(run->notes);
    free(run);
}

const char* cml_exp_run_id(const CMLRun* run) { return run ? run->id : ""; }
