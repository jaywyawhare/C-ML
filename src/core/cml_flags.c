#include "core/cml_flags.h"

#include <pthread.h>
#include <stdlib.h>

typedef struct {
    const char* name;
    const char* desc;
    int default_value;
    int value;
    bool set;
} FlagEntry;

static FlagEntry g_flags[CML_FLAG_COUNT] = {
#define CML_FLAG_ENTRY(id, name, dflt, desc) [CML_FLAG_##id] = {name, desc, (dflt), (dflt), false},
    CML_FLAG_LIST(CML_FLAG_ENTRY)
#undef CML_FLAG_ENTRY
};

static pthread_once_t g_flags_once = PTHREAD_ONCE_INIT;

static void flags_init_once(void) {
    for (int i = 0; i < CML_FLAG_COUNT; i++) {
        const char* v = getenv(g_flags[i].name);
        if (v && *v) {
            g_flags[i].value = (int)strtol(v, NULL, 10);
            g_flags[i].set   = true;
        }
    }
}

void cml_flags_init(void) { pthread_once(&g_flags_once, flags_init_once); }

int cml_flag(CmlFlag id) {
    if ((unsigned)id >= CML_FLAG_COUNT)
        return 0;
    cml_flags_init();
    return g_flags[id].value;
}

bool cml_flag_enabled(CmlFlag id) { return cml_flag(id) != 0; }

bool cml_flag_was_set(CmlFlag id) {
    if ((unsigned)id >= CML_FLAG_COUNT)
        return false;
    cml_flags_init();
    return g_flags[id].set;
}

const char* cml_flag_name(CmlFlag id) {
    return (unsigned)id < CML_FLAG_COUNT ? g_flags[id].name : "";
}

const char* cml_flag_desc(CmlFlag id) {
    return (unsigned)id < CML_FLAG_COUNT ? g_flags[id].desc : "";
}

int cml_flag_push(CmlFlag id, int value) {
    if ((unsigned)id >= CML_FLAG_COUNT)
        return 0;
    cml_flags_init();
    int prev          = g_flags[id].value;
    g_flags[id].value = value;
    return prev;
}

void cml_flag_pop(CmlFlag id, int previous) {
    if ((unsigned)id >= CML_FLAG_COUNT)
        return;
    g_flags[id].value = previous;
}

void cml_flags_dump(FILE* out) {
    if (!out)
        return;
    cml_flags_init();
    fprintf(out, "cml flags (non-default):\n");
    bool any = false;
    for (int i = 0; i < CML_FLAG_COUNT; i++) {
        if (g_flags[i].set || g_flags[i].value != g_flags[i].default_value) {
            fprintf(out, "  %-20s = %-4d %s\n", g_flags[i].name, g_flags[i].value, g_flags[i].desc);
            any = true;
        }
    }
    if (!any)
        fprintf(out, "  (all defaults)\n");
}
