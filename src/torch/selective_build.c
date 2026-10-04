/*
 * selective_build.c — Selective kernel build for embedded deployment
 */

#include "torch/selective_build.h"
#include "torch/pte.h"
#include "core/logging.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#ifdef CML_TORCH_SELECTIVE_OPS
#include <string.h>
#endif

static bool g_active[UOP_COUNT];
static bool g_dtype_active[CML_TORCH_MAX_SELECTIVE_DTYPES];
static bool g_initialized = false;

static const struct {
    const char* name;
    UOpType op;
} g_selective_op_table[] = {
    {"add", UOP_ADD},
    {"sub", UOP_SUB},
    {"mul", UOP_MUL},
    {"div", UOP_DIV},
    {"matmul", UOP_MATMUL},
    {"relu", UOP_RELU},
    {"sigmoid", UOP_SIGMOID},
    {"tanh", UOP_TANH},
    {"sum", UOP_SUM},
    {"mean", UOP_MEAN},
    {"quick_gelu", UOP_QUICK_GELU},
    {"reshape", UOP_RESHAPE},
    {"transpose", UOP_PERMUTE},
    {"linear", UOP_LINEAR},
    {NULL, UOP_COUNT},
};

/** Resolve a case-insensitive op name to its UOpType, or UOP_COUNT if unknown. */
static UOpType selective_op_from_name(const char* name) {
    if (!name)
        return UOP_COUNT;
    for (int i = 0; g_selective_op_table[i].name; i++) {
        if (strcasecmp(name, g_selective_op_table[i].name) == 0)
            return g_selective_op_table[i].op;
    }
    return UOP_COUNT;
}

/** Initialize the active-op/dtype tables once, honoring any compile-time
 *  CML_TORCH_SELECTIVE_OPS allowlist. */
static void selective_init_defaults(void) {
    if (g_initialized)
        return;
    for (int i = 0; i < UOP_COUNT; i++)
        g_active[i] = true;
    for (int i = 0; i < CML_TORCH_MAX_SELECTIVE_DTYPES; i++)
        g_dtype_active[i] = true;
#ifdef CML_TORCH_SELECTIVE_OPS
    {
        for (int i = 0; i < UOP_COUNT; i++)
            g_active[i] = false;
        char spec_copy[512];
        strncpy(spec_copy, CML_TORCH_SELECTIVE_OPS, sizeof(spec_copy) - 1);
        spec_copy[sizeof(spec_copy) - 1] = '\0';
        char* save                       = NULL;
        for (char* tok = strtok_r(spec_copy, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
            while (*tok == ' ')
                tok++;
            UOpType op = selective_op_from_name(tok);
            if (op < UOP_COUNT)
                g_active[op] = true;
        }
    }
#endif
    g_initialized = true;
}

/** Build a config with every op and dtype enabled. */
TorchSelectiveBuildConfig torch_selective_build_all(void) {
    selective_init_defaults();
    TorchSelectiveBuildConfig cfg = {0};
    cfg.all_enabled               = true;
    for (int i = 0; i < UOP_COUNT; i++)
        cfg.enabled[i] = true;
    for (int i = 0; i < CML_TORCH_MAX_SELECTIVE_DTYPES; i++)
        cfg.dtype_enabled[i] = true;
    return cfg;
}

/** Build a config with all ops disabled (dtypes left enabled) to opt in explicitly. */
TorchSelectiveBuildConfig torch_selective_build_none(void) {
    selective_init_defaults();
    TorchSelectiveBuildConfig cfg = {0};
    cfg.all_enabled               = false;
    memset(cfg.enabled, 0, sizeof(cfg.enabled));
    for (int i = 0; i < CML_TORCH_MAX_SELECTIVE_DTYPES; i++)
        cfg.dtype_enabled[i] = true;
    return cfg;
}

/** Enable a single op in the config (ignores out-of-range ops). */
void torch_selective_build_enable_op(TorchSelectiveBuildConfig* cfg, UOpType op) {
    if (!cfg || op < 0 || op >= UOP_COUNT)
        return;
    cfg->enabled[op] = true;
}

/** Enable an array of ops in the config. */
void torch_selective_build_enable_ops(TorchSelectiveBuildConfig* cfg, const UOpType* ops,
                                      int count) {
    if (!cfg || !ops)
        return;
    for (int i = 0; i < count; i++)
        torch_selective_build_enable_op(cfg, ops[i]);
}

/** Make a config active: copy its enabled op/dtype sets into the global runtime tables. */
void torch_selective_build_apply(const TorchSelectiveBuildConfig* cfg) {
    selective_init_defaults();
    if (!cfg)
        return;
    if (cfg->all_enabled) {
        for (int i = 0; i < UOP_COUNT; i++)
            g_active[i] = true;
    } else {
        for (int i = 0; i < UOP_COUNT; i++)
            g_active[i] = cfg->enabled[i];
    }
    for (int i = 0; i < CML_TORCH_MAX_SELECTIVE_DTYPES; i++)
        g_dtype_active[i] = cfg->dtype_enabled[i];
}

/** Re-enable every op in the global runtime tables (undo a selective apply). */
void torch_selective_build_reset(void) {
    selective_init_defaults();
    for (int i = 0; i < UOP_COUNT; i++)
        g_active[i] = true;
}

/** Query whether an op kernel is currently linked/active in the runtime. */
bool torch_selective_build_is_op_enabled(UOpType op) {
    selective_init_defaults();
    if (op < 0 || op >= UOP_COUNT)
        return false;
    return g_active[op];
}

/** Query whether a dtype is currently active (out-of-range dtypes default to enabled). */
bool torch_selective_build_is_dtype_enabled(DType dtype) {
    selective_init_defaults();
    if ((int)dtype < 0 || (int)dtype >= CML_TORCH_MAX_SELECTIVE_DTYPES)
        return true;
    return g_dtype_active[(int)dtype];
}

/** Parse a comma-separated op-name spec into a config; returns 0, or -1 on null args. */
int torch_selective_build_from_string(const char* spec, TorchSelectiveBuildConfig* out) {
    if (!spec || !out)
        return -1;
    *out = torch_selective_build_none();

    char* copy = strdup(spec);
    if (!copy)
        return -1;

    char* save = NULL;
    for (char* tok = strtok_r(copy, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
        while (*tok == ' ')
            tok++;
        UOpType op = selective_op_from_name(tok);
        if (op < UOP_COUNT)
            torch_selective_build_enable_op(out, op);
    }
    free(copy);
    return 0;
}

/** Derive the required-op config from the instructions of a loaded .cpte program. */
int torch_selective_build_from_pte(const char* pte_path, TorchSelectiveBuildConfig* out) {
    if (!pte_path || !out)
        return -1;
    CMLPTEModel* m = torch_pte_load(pte_path);
    if (!m)
        return -1;

    *out = torch_selective_build_none();
    for (uint32_t i = 0; i < m->meta.num_instructions; i++)
        torch_selective_build_enable_op(out, (UOpType)m->instructions[i].kernel_id);

    torch_pte_free(m);
    return 0;
}

/** Write a config to a newline-delimited kernel manifest file ("all" or op names). */
int torch_selective_build_save(const TorchSelectiveBuildConfig* cfg, const char* path) {
    if (!cfg || !path)
        return -1;
    FILE* f = fopen(path, "w");
    if (!f)
        return -1;
    if (cfg->all_enabled) {
        fprintf(f, "all\n");
    } else {
        for (int i = 0; g_selective_op_table[i].name; i++) {
            if (cfg->enabled[g_selective_op_table[i].op])
                fprintf(f, "%s\n", g_selective_op_table[i].name);
        }
    }
    fclose(f);
    return 0;
}

/** Load a config from a kernel manifest file written by torch_selective_build_save. */
int torch_selective_build_load(const char* path, TorchSelectiveBuildConfig* out) {
    if (!path || !out)
        return -1;
    FILE* f = fopen(path, "r");
    if (!f)
        return -1;

    char line[256];
    *out = torch_selective_build_none();
    while (fgets(line, sizeof(line), f)) {
        line[strcspn(line, "\r\n")] = '\0';
        if (strcmp(line, "all") == 0) {
            *out = torch_selective_build_all();
            break;
        }
        UOpType op = selective_op_from_name(line);
        if (op < UOP_COUNT)
            torch_selective_build_enable_op(out, op);
    }
    fclose(f);
    return 0;
}
