#include "nn/state.h"
#include "core/safetensors.h"
#include "core/logging.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "alloc/cml_allocator.h"

#define SD_INIT_CAP 32

/**
 * Allocate an empty state dict with an initial entry capacity; NULL on OOM.
 * @return (StateDict*)
 */
StateDict* nn_state_dict_create(void) {
    StateDict* sd = cml_calloc(1, sizeof(StateDict));
    if (!sd)
        return NULL;
    sd->entries = cml_malloc(SD_INIT_CAP * sizeof(StateDictEntry));
    if (!sd->entries) {
        cml_free(sd);
        return NULL;
    }
    sd->capacity = SD_INIT_CAP;
    sd->count    = 0;
    return sd;
}

/**
 * Free the dict and its owned key strings; the tensor values are not owned.
 * @param sd (StateDict*)
 * @return (void)
 */
void nn_state_dict_free(StateDict* sd) {
    if (!sd)
        return;
    for (int i = 0; i < sd->count; ++i)
        cml_free(sd->entries[i].key);

    cml_free(sd->entries);
    cml_free(sd);
}

/**
 * Insert or overwrite the entry for `key`, growing the backing array as needed.
 * @param sd (StateDict*)
 * @param key (const char*)
 * @param value (Tensor*)
 * @return (int)
 */
int nn_state_dict_set(StateDict* sd, const char* key, Tensor* value) {
    if (!sd || !key)
        return -1;
    for (int i = 0; i < sd->count; ++i) {
        if (strcmp(sd->entries[i].key, key) == 0) {
            sd->entries[i].value = value;
            return 0;
        }
    }
    if (sd->count >= sd->capacity) {
        int new_cap         = sd->capacity * 2;
        StateDictEntry* tmp = cml_realloc(sd->entries, (size_t)new_cap * sizeof(StateDictEntry));
        if (!tmp)
            return -1;
        sd->entries  = tmp;
        sd->capacity = new_cap;
    }
    sd->entries[sd->count].key   = cml_strdup(key);
    sd->entries[sd->count].value = value;
    if (!sd->entries[sd->count].key)
        return -1;
    ++sd->count;
    return 0;
}

/**
 * Look up the tensor stored under `key`, or NULL if absent.
 * @param sd (const StateDict*)
 * @param key (const char*)
 * @return (Tensor*)
 */
Tensor* nn_state_dict_get(const StateDict* sd, const char* key) {
    if (!sd || !key)
        return NULL;
    for (int i = 0; i < sd->count; ++i)
        if (strcmp(sd->entries[i].key, key) == 0)
            return sd->entries[i].value;
    return NULL;
}

/**
 * Remove the entry for `key` (swap-with-last); returns 1 if removed, 0 if not found.
 * @param sd (StateDict*)
 * @param key (const char*)
 * @return (int)
 */
int nn_state_dict_remove(StateDict* sd, const char* key) {
    if (!sd || !key)
        return 0;
    for (int i = 0; i < sd->count; ++i) {
        if (strcmp(sd->entries[i].key, key) == 0) {
            cml_free(sd->entries[i].key);
            sd->entries[i] = sd->entries[sd->count - 1];
            --sd->count;
            return 1;
        }
    }
    return 0;
}

/**
 * Recursively flatten a module tree's parameters into `sd` under dotted `prefix` keys.
 * @param module (const Module*)
 * @param sd (StateDict*)
 * @param prefix (const char*)
 * @return (int)
 */
static int collect_params(const Module* module, StateDict* sd, const char* prefix) {
    if (!module || !sd)
        return -1;
    char key[256];
    for (int i = 0; i < module->num_parameters; ++i) {
        Parameter* p = module->parameters[i];
        if (!p || !p->tensor)
            continue;
        if (prefix && prefix[0])
            snprintf(key, sizeof(key), "%s.%s", prefix, p->name ? p->name : "param");
        else
            snprintf(key, sizeof(key), "%s", p->name ? p->name : "param");
        if (nn_state_dict_set(sd, key, p->tensor) != 0)
            return -1;
    }
    Module* sub = module->next;
    int idx     = 0;
    while (sub) {
        char sub_prefix[256];
        if (prefix && prefix[0])
            snprintf(sub_prefix, sizeof(sub_prefix), "%s.%s%d", prefix,
                     sub->name ? sub->name : "layer", idx);
        else
            snprintf(sub_prefix, sizeof(sub_prefix), "%s%d", sub->name ? sub->name : "layer", idx);
        if (collect_params(sub, sd, sub_prefix) != 0)
            return -1;
        sub = sub->next;
        ++idx;
    }
    return 0;
}

/**
 * Build a state dict of all parameters in `module`, keyed by optional `prefix`.
 * @param module (const Module*)
 * @param prefix (const char*)
 * @return (StateDict*)
 */
StateDict* nn_get_state_dict(const Module* module, const char* prefix) {
    StateDict* sd = nn_state_dict_create();
    if (!sd)
        return NULL;
    if (collect_params(module, sd, prefix ? prefix : "") != 0) {
        nn_state_dict_free(sd);
        return NULL;
    }
    return sd;
}

/**
 * Copy matching tensors from `sd` into `module`'s parameters by key. Mismatched
 * sizes are skipped; with `strict`, missing keys also make the call fail (-1).
 * @param module (Module*)
 * @param sd (const StateDict*)
 * @param strict (bool)
 * @return (int)
 */
int nn_load_state_dict(Module* module, const StateDict* sd, bool strict) {
    if (!module || !sd)
        return -1;
    StateDict* mod_sd = nn_get_state_dict(module, NULL);
    if (!mod_sd)
        return -1;

    int rc = 0;
    for (int i = 0; i < sd->count; ++i) {
        const char* key = sd->entries[i].key;
        Tensor* src     = sd->entries[i].value;
        Tensor* dst     = nn_state_dict_get(mod_sd, key);
        if (!dst) {
            if (strict) {
                LOG_ERROR("nn_load_state_dict: key '%s' not found in module", key);
                rc = -1;
            }
            continue;
        }
        if (src->numel != dst->numel) {
            LOG_ERROR("nn_load_state_dict: size mismatch for key '%s': "
                      "source %zu != dest %zu",
                      key, src->numel, dst->numel);
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

/**
 * Serialize every tensor in `sd` to a safetensors file at `path`.
 * @param sd (const StateDict*)
 * @param path (const char*)
 * @return (int)
 */
int nn_save(const StateDict* sd, const char* path) {
    if (!sd || !path)
        return -1;
    SafeTensorsContext* ctx = safetensors_open_write(path);
    if (!ctx)
        return -1;
    int rc = 0;
    for (int i = 0; i < sd->count; ++i) {
        if (safetensors_write_tensor(ctx, sd->entries[i].key, sd->entries[i].value) != 0) {
            LOG_ERROR("nn_save: failed to write tensor '%s'", sd->entries[i].key);
            rc = -1;
        }
    }
    safetensors_close(ctx);
    return rc;
}

/**
 * Load a safetensors file into a fresh state dict; NULL on failure.
 * @param path (const char*)
 * @return (StateDict*)
 */
StateDict* nn_load(const char* path) {
    if (!path)
        return NULL;
    SafeTensorsContext* ctx = safetensors_open_read(path);
    if (!ctx)
        return NULL;
    StateDict* sd = nn_state_dict_create();
    if (!sd) {
        safetensors_close(ctx);
        return NULL;
    }
    int n = safetensors_get_num_tensors(ctx);
    for (int i = 0; i < n; ++i) {
        const char* name = safetensors_get_tensor_name(ctx, i);
        if (!name)
            continue;
        Tensor* t = safetensors_read_tensor(ctx, name);
        if (!t)
            continue;
        nn_state_dict_set(sd, name, t);
    }
    safetensors_close(ctx);
    return sd;
}

/**
 * Total element count summed across all tensors in the dict.
 * @param sd (const StateDict*)
 * @return (size_t)
 */
size_t nn_state_dict_num_params(const StateDict* sd) {
    if (!sd)
        return 0;
    size_t total = 0;
    for (int i = 0; i < sd->count; ++i)
        if (sd->entries[i].value)
            total += sd->entries[i].value->numel;
    return total;
}

/**
 * Total byte footprint of all tensors in the dict (numel * dtype size).
 * @param sd (const StateDict*)
 * @return (size_t)
 */
size_t nn_state_dict_bytes(const StateDict* sd) {
    if (!sd)
        return 0;
    size_t total = 0;
    for (int i = 0; i < sd->count; ++i) {
        Tensor* t = sd->entries[i].value;
        if (t)
            total += t->numel * cml_dtype_size(t->dtype);
    }
    return total;
}

/**
 * Print a human-readable summary of each entry's key, shape and element count.
 * @param sd (const StateDict*)
 * @return (void)
 */
void nn_state_dict_print(const StateDict* sd) {
    if (!sd) {
        printf("StateDict(NULL)\n");
        return;
    }
    printf("StateDict (%d entries, %.2f MB total):\n", sd->count,
           (double)nn_state_dict_bytes(sd) / (1024.0 * 1024.0));
    for (int i = 0; i < sd->count; ++i) {
        Tensor* t = sd->entries[i].value;
        if (!t) {
            printf("  %-40s  <null>\n", sd->entries[i].key);
            continue;
        }
        printf("  %-40s  [", sd->entries[i].key);
        for (int d = 0; d < t->ndim; ++d) {
            if (d > 0)
                printf(", ");
            printf("%d", t->shape[d]);
        }
        printf("] numel=%zu\n", t->numel);
    }
}

/**
 * In-place linear interpolation of matching tensors: dst = alpha*src + (1-alpha)*dst.
 * @param dst (StateDict*)
 * @param src (const StateDict*)
 * @param alpha (float)
 * @return (int)
 */
int nn_state_dict_lerp(StateDict* dst, const StateDict* src, float alpha) {
    if (!dst || !src)
        return -1;
    for (int i = 0; i < src->count; ++i) {
        Tensor* s = src->entries[i].value;
        Tensor* d = nn_state_dict_get(dst, src->entries[i].key);
        if (!s || !d || s->numel != d->numel)
            continue;
        if (!s->data || !d->data)
            continue;
        float* sf = (float*)s->data;
        float* df = (float*)d->data;
        for (size_t j = 0; j < d->numel; ++j)
            df[j] = alpha * sf[j] + (1.0f - alpha) * df[j];
    }
    return 0;
}

/** Create a weight-average tracker seeded with a clone of @p model's current weights.
 * @param model (const Module*)
 * @param decay (float)
 * @return (CMLWeightAvg*)
 */
/** Clone @p module's current weights into a StateDict that owns its tensors.
 * @param module (const Module*)
 * @return (StateDict*)
 */
StateDict* nn_state_dict_clone(const Module* module) {
    if (!module)
        return NULL;
    StateDict* live = nn_get_state_dict(module, NULL);
    if (!live)
        return NULL;
    StateDict* out = nn_state_dict_create();
    if (!out) {
        nn_state_dict_free(live);
        return NULL;
    }
    for (int i = 0; i < live->count; i++) {
        Tensor* c = tensor_clone(live->entries[i].value);
        if (!c || nn_state_dict_set(out, live->entries[i].key, c) != 0) {
            tensor_free(c);
            nn_state_dict_free_full(out);
            nn_state_dict_free(live);
            return NULL;
        }
    }
    nn_state_dict_free(live);
    return out;
}

/** Free a StateDict that owns its tensors.
 * @param sd (StateDict*)
 * @return (void)
 */
void nn_state_dict_free_full(StateDict* sd) {
    if (!sd)
        return;
    for (int i = 0; i < sd->count; i++)
        tensor_free(sd->entries[i].value);
    nn_state_dict_free(sd);
}

/** Create a keep-best checkpoint.
 * @param maximize (int)
 * @return (CMLCheckpoint*)
 */
CMLCheckpoint* cml_checkpoint_create(int maximize) {
    CMLCheckpoint* c = cml_calloc(1, sizeof(CMLCheckpoint));
    if (c)
        c->maximize = maximize ? 1 : 0;
    return c;
}

/** Snapshot @p model's weights if @p metric improves on the best so far.
 * @param ckpt (CMLCheckpoint*)
 * @param model (Module*)
 * @param metric (float)
 * @return (int)
 */
int cml_checkpoint_update(CMLCheckpoint* ckpt, Module* model, float metric) {
    if (!ckpt || !model)
        return 0;
    int improved = !ckpt->has_best ||
                   (ckpt->maximize ? metric > ckpt->best_metric : metric < ckpt->best_metric);
    if (!improved)
        return 0;
    StateDict* snap = nn_state_dict_clone(model);
    if (!snap)
        return 0;
    nn_state_dict_free_full(ckpt->best);
    ckpt->best        = snap;
    ckpt->best_metric = metric;
    ckpt->has_best    = 1;
    return 1;
}

/** Load the best snapshot back into @p model. Returns 0 on success, -1 if none saved.
 * @param ckpt (CMLCheckpoint*)
 * @param model (Module*)
 * @return (int)
 */
int cml_checkpoint_restore(CMLCheckpoint* ckpt, Module* model) {
    if (!ckpt || !model || !ckpt->has_best)
        return -1;
    return nn_load_state_dict(model, ckpt->best, false);
}

/** Free a checkpoint and its snapshot.
 * @param ckpt (CMLCheckpoint*)
 * @return (void)
 */
void cml_checkpoint_free(CMLCheckpoint* ckpt) {
    if (!ckpt)
        return;
    nn_state_dict_free_full(ckpt->best);
    cml_free(ckpt);
}

CMLWeightAvg* cml_weight_avg_create(const Module* model, float decay) {
    if (!model)
        return NULL;
    StateDict* avg = nn_state_dict_clone(model);
    if (!avg)
        return NULL;
    CMLWeightAvg* wa = cml_calloc(1, sizeof(CMLWeightAvg));
    if (!wa) {
        nn_state_dict_free_full(avg);
        return NULL;
    }
    wa->avg   = avg;
    wa->decay = decay;
    wa->count = 0;
    return wa;
}

/** EMA update: avg = decay*avg + (1-decay)*model.
 * @param wa (CMLWeightAvg*)
 * @param model (const Module*)
 * @return (void)
 */
void cml_weight_avg_update_ema(CMLWeightAvg* wa, const Module* model) {
    if (!wa || !model)
        return;
    StateDict* cur = nn_get_state_dict(model, NULL);
    if (!cur)
        return;
    nn_state_dict_lerp(wa->avg, cur, 1.0f - wa->decay); /* dst = alpha*src + (1-alpha)*dst */
    nn_state_dict_free(cur);                            /* refs live params; values untouched */
    wa->count++;
}

/** SWA update: fold @p model into the equal running average of snapshots.
 * @param wa (CMLWeightAvg*)
 * @param model (const Module*)
 * @return (void)
 */
void cml_weight_avg_update_swa(CMLWeightAvg* wa, const Module* model) {
    if (!wa || !model)
        return;
    StateDict* cur = nn_get_state_dict(model, NULL);
    if (!cur)
        return;
    float alpha = 1.0f / (float)(wa->count + 1); /* equal weight to each snapshot */
    nn_state_dict_lerp(wa->avg, cur, alpha);
    nn_state_dict_free(cur);
    wa->count++;
}

/** Copy the averaged weights into @p model. Returns 0 on success.
 * @param wa (CMLWeightAvg*)
 * @param model (Module*)
 * @return (int)
 */
int cml_weight_avg_copy_to(CMLWeightAvg* wa, Module* model) {
    if (!wa || !model)
        return -1;
    return nn_load_state_dict(model, wa->avg, false);
}

/** Free a weight-average tracker and its cloned tensors.
 * @param wa (CMLWeightAvg*)
 * @return (void)
 */
void cml_weight_avg_free(CMLWeightAvg* wa) {
    if (!wa)
        return;
    nn_state_dict_free_full(wa->avg);
    cml_free(wa);
}

/* Ascending comparator over float magnitudes. */
static int prune_mag_cmp(const void* a, const void* b) {
    float x = *(const float*)a, y = *(const float*)b;
    return (x > y) - (x < y);
}

/** Fraction of float32 parameter weights that are exactly zero.
 * @param model (const Module*)
 * @return (float)
 */
float cml_model_sparsity(const Module* model) {
    if (!model)
        return 0.0f;
    Parameter** p = NULL;
    int np        = 0;
    module_collect_parameters((Module*)model, &p, &np, true);
    size_t total = 0, zeros = 0;
    for (int i = 0; i < np; i++) {
        Tensor* t = p[i]->tensor;
        if (!t || t->dtype != DTYPE_FLOAT32)
            continue;
        float* d = (float*)tensor_data_ptr(t);
        if (!d)
            continue;
        for (size_t j = 0; j < t->numel; j++) {
            total++;
            if (d[j] == 0.0f)
                zeros++;
        }
    }
    cml_free(p);
    return total ? (float)zeros / (float)total : 0.0f;
}

/** Global magnitude pruning: zero the smallest-|w| float32 weights across all parameters
 * until the target fraction are zero. Returns the achieved sparsity.
 * @param model (Module*)
 * @param sparsity (float)
 * @return (float)
 */
float cml_prune_magnitude(Module* model, float sparsity) {
    if (!model || sparsity <= 0.0f)
        return cml_model_sparsity(model);
    if (sparsity > 1.0f)
        sparsity = 1.0f;
    Parameter** p = NULL;
    int np        = 0;
    module_collect_parameters(model, &p, &np, true);
    size_t total = 0;
    for (int i = 0; i < np; i++) {
        Tensor* t = p[i]->tensor;
        if (t && t->dtype == DTYPE_FLOAT32)
            total += t->numel;
    }
    if (total == 0) {
        cml_free(p);
        return 0.0f;
    }
    float* mags = cml_malloc(total * sizeof(float));
    if (!mags) {
        cml_free(p);
        return cml_model_sparsity(model);
    }
    size_t idx = 0;
    for (int i = 0; i < np; i++) {
        Tensor* t = p[i]->tensor;
        if (!t || t->dtype != DTYPE_FLOAT32)
            continue;
        float* d = (float*)tensor_data_ptr(t);
        if (!d)
            continue;
        for (size_t j = 0; j < t->numel; j++)
            mags[idx++] = fabsf(d[j]);
    }
    qsort(mags, idx, sizeof(float), prune_mag_cmp);
    size_t k = (size_t)(sparsity * (float)idx);
    float threshold;
    if (k == 0)
        threshold = 0.0f;
    else if (k >= idx)
        threshold = mags[idx - 1] + 1.0f;
    else
        threshold = mags[k];
    cml_free(mags);
    for (int i = 0; i < np; i++) {
        Tensor* t = p[i]->tensor;
        if (!t || t->dtype != DTYPE_FLOAT32)
            continue;
        float* d = (float*)tensor_data_ptr(t);
        if (!d)
            continue;
        for (size_t j = 0; j < t->numel; j++)
            if (fabsf(d[j]) < threshold)
                d[j] = 0.0f;
    }
    cml_free(p);
    return cml_model_sparsity(model);
}
