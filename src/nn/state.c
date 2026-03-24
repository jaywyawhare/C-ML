#include "nn/state.h"
#include "core/safetensors.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* =========================================================================
 * StateDict lifecycle
 * ========================================================================= */

#define SD_INIT_CAP 32

StateDict* nn_state_dict_create(void) {
    StateDict* sd = calloc(1, sizeof(StateDict));
    if (!sd) return NULL;
    sd->entries  = malloc(SD_INIT_CAP * sizeof(StateDictEntry));
    if (!sd->entries) { free(sd); return NULL; }
    sd->capacity = SD_INIT_CAP;
    sd->count    = 0;
    return sd;
}

void nn_state_dict_free(StateDict* sd) {
    if (!sd) return;
    for (int i = 0; i < sd->count; ++i)
        free(sd->entries[i].key);
    /* Tensors are borrowed — we do not free them. */
    free(sd->entries);
    free(sd);
}

int nn_state_dict_set(StateDict* sd, const char* key, Tensor* value) {
    if (!sd || !key) return -1;
    /* Update existing entry. */
    for (int i = 0; i < sd->count; ++i) {
        if (strcmp(sd->entries[i].key, key) == 0) {
            sd->entries[i].value = value;
            return 0;
        }
    }
    /* Grow if needed. */
    if (sd->count >= sd->capacity) {
        int new_cap = sd->capacity * 2;
        StateDictEntry* tmp = realloc(sd->entries,
                                       (size_t)new_cap * sizeof(StateDictEntry));
        if (!tmp) return -1;
        sd->entries  = tmp;
        sd->capacity = new_cap;
    }
    sd->entries[sd->count].key   = strdup(key);
    sd->entries[sd->count].value = value;
    if (!sd->entries[sd->count].key) return -1;
    ++sd->count;
    return 0;
}

Tensor* nn_state_dict_get(const StateDict* sd, const char* key) {
    if (!sd || !key) return NULL;
    for (int i = 0; i < sd->count; ++i)
        if (strcmp(sd->entries[i].key, key) == 0)
            return sd->entries[i].value;
    return NULL;
}

int nn_state_dict_remove(StateDict* sd, const char* key) {
    if (!sd || !key) return 0;
    for (int i = 0; i < sd->count; ++i) {
        if (strcmp(sd->entries[i].key, key) == 0) {
            free(sd->entries[i].key);
            sd->entries[i] = sd->entries[sd->count - 1];
            --sd->count;
            return 1;
        }
    }
    return 0;
}

/* =========================================================================
 * Building from a Module
 * ========================================================================= */

static int collect_params(const Module* module, StateDict* sd, const char* prefix) {
    if (!module || !sd) return -1;
    char key[256];
    for (int i = 0; i < module->num_parameters; ++i) {
        Parameter* p = module->parameters[i];
        if (!p || !p->tensor) continue;
        if (prefix && prefix[0])
            snprintf(key, sizeof(key), "%s.%s",
                     prefix, p->name ? p->name : "param");
        else
            snprintf(key, sizeof(key), "%s",
                     p->name ? p->name : "param");
        if (nn_state_dict_set(sd, key, p->tensor) != 0) return -1;
    }
    /* Recurse into chained submodules (Sequential, etc.). */
    Module* sub = module->next;
    int idx = 0;
    while (sub) {
        char sub_prefix[256];
        if (prefix && prefix[0])
            snprintf(sub_prefix, sizeof(sub_prefix), "%s.%s%d",
                     prefix, sub->name ? sub->name : "layer", idx);
        else
            snprintf(sub_prefix, sizeof(sub_prefix), "%s%d",
                     sub->name ? sub->name : "layer", idx);
        if (collect_params(sub, sd, sub_prefix) != 0) return -1;
        sub = sub->next;
        ++idx;
    }
    return 0;
}

StateDict* nn_get_state_dict(const Module* module, const char* prefix) {
    StateDict* sd = nn_state_dict_create();
    if (!sd) return NULL;
    if (collect_params(module, sd, prefix ? prefix : "") != 0) {
        nn_state_dict_free(sd);
        return NULL;
    }
    return sd;
}

int nn_load_state_dict(Module* module, const StateDict* sd, bool strict) {
    if (!module || !sd) return -1;
    /* Build the module's state dict so we can check which keys exist. */
    StateDict* mod_sd = nn_get_state_dict(module, NULL);
    if (!mod_sd) return -1;

    int rc = 0;
    /* Apply each entry in sd to the module. */
    for (int i = 0; i < sd->count; ++i) {
        const char* key   = sd->entries[i].key;
        Tensor* src       = sd->entries[i].value;
        Tensor* dst       = nn_state_dict_get(mod_sd, key);
        if (!dst) {
            if (strict) {
                LOG_ERROR("nn_load_state_dict: key '%s' not found in module", key);
                rc = -1;
            }
            continue;
        }
        /* Copy data from src into dst (in-place update of existing tensor). */
        if (src->numel != dst->numel) {
            LOG_ERROR("nn_load_state_dict: size mismatch for key '%s': "
                          "source %zu != dest %zu", key, src->numel, dst->numel);
            rc = -1;
            continue;
        }
        size_t byte_size = src->numel * cml_dtype_size(src->dtype);
        if (dst->data && src->data) {
            memcpy(dst->data, src->data, byte_size);
            dst->is_executed = true;
        }
    }
    nn_state_dict_free(mod_sd);
    return rc;
}

/* =========================================================================
 * Serialisation
 * ========================================================================= */

int nn_save(const StateDict* sd, const char* path) {
    if (!sd || !path) return -1;
    SafeTensorsContext* ctx = safetensors_open_write(path);
    if (!ctx) return -1;
    int rc = 0;
    for (int i = 0; i < sd->count; ++i) {
        if (safetensors_write_tensor(ctx, sd->entries[i].key,
                                     sd->entries[i].value) != 0) {
            LOG_ERROR("nn_save: failed to write tensor '%s'", sd->entries[i].key);
            rc = -1;
        }
    }
    safetensors_close(ctx);
    return rc;
}

StateDict* nn_load(const char* path) {
    if (!path) return NULL;
    SafeTensorsContext* ctx = safetensors_open_read(path);
    if (!ctx) return NULL;
    StateDict* sd = nn_state_dict_create();
    if (!sd) { safetensors_close(ctx); return NULL; }
    int n = safetensors_get_num_tensors(ctx);
    for (int i = 0; i < n; ++i) {
        const char* name = safetensors_get_tensor_name(ctx, i);
        if (!name) continue;
        Tensor* t = safetensors_read_tensor(ctx, name);
        if (!t) continue;
        nn_state_dict_set(sd, name, t);
    }
    safetensors_close(ctx);
    return sd;
}

/* =========================================================================
 * Utilities
 * ========================================================================= */

size_t nn_state_dict_num_params(const StateDict* sd) {
    if (!sd) return 0;
    size_t total = 0;
    for (int i = 0; i < sd->count; ++i)
        if (sd->entries[i].value)
            total += sd->entries[i].value->numel;
    return total;
}

size_t nn_state_dict_bytes(const StateDict* sd) {
    if (!sd) return 0;
    size_t total = 0;
    for (int i = 0; i < sd->count; ++i) {
        Tensor* t = sd->entries[i].value;
        if (t) total += t->numel * cml_dtype_size(t->dtype);
    }
    return total;
}

void nn_state_dict_print(const StateDict* sd) {
    if (!sd) { printf("StateDict(NULL)\n"); return; }
    printf("StateDict (%d entries, %.2f MB total):\n",
           sd->count, (double)nn_state_dict_bytes(sd) / (1024.0 * 1024.0));
    for (int i = 0; i < sd->count; ++i) {
        Tensor* t = sd->entries[i].value;
        if (!t) { printf("  %-40s  <null>\n", sd->entries[i].key); continue; }
        printf("  %-40s  [", sd->entries[i].key);
        for (int d = 0; d < t->ndim; ++d) {
            if (d > 0) printf(", ");
            printf("%d", t->shape[d]);
        }
        printf("] numel=%zu\n", t->numel);
    }
}

int nn_state_dict_lerp(StateDict* dst, const StateDict* src, float alpha) {
    if (!dst || !src) return -1;
    for (int i = 0; i < src->count; ++i) {
        Tensor* s = src->entries[i].value;
        Tensor* d = nn_state_dict_get(dst, src->entries[i].key);
        if (!s || !d || s->numel != d->numel) continue;
        if (!s->data || !d->data) continue;
        float* sf = (float*)s->data;
        float* df = (float*)d->data;
        for (size_t j = 0; j < d->numel; ++j)
            df[j] = alpha * sf[j] + (1.0f - alpha) * df[j];
    }
    return 0;
}
