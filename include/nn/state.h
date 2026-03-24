/*
 * nn/state — model state dict API.
 *
 * Mirrors TinyGrad's tinygrad/nn/state.py.
 *
 * Provides clean utilities for:
 *   - Collecting all named parameters from a Module tree (state dict)
 *   - Loading a state dict back into a Module (in-place weight assignment)
 *   - Saving / loading state dicts to/from SafeTensors files
 *   - Printing parameter summaries
 */

#ifndef CML_NN_STATE_H
#define CML_NN_STATE_H

#include "nn.h"              /* Module, Parameter */
#include "tensor/tensor.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- StateDict ---- */

typedef struct StateDictEntry {
    char*   key;    /* fully-qualified param name, e.g. "encoder.layer0.weight" */
    Tensor* value;  /* parameter tensor (borrowed — Module owns the data) */
} StateDictEntry;

typedef struct StateDict {
    StateDictEntry* entries;
    int             count;
    int             capacity;
} StateDict;

/* ---- Lifecycle ---- */

StateDict* nn_state_dict_create(void);
void       nn_state_dict_free(StateDict* sd);

/* Add / replace a key.  Tensor is borrowed (not copied or freed by StateDict). */
int nn_state_dict_set(StateDict* sd, const char* key, Tensor* value);

/* Look up a key.  Returns NULL if not found. */
Tensor* nn_state_dict_get(const StateDict* sd, const char* key);

/* Remove a key.  Returns 1 if removed, 0 if not found. */
int nn_state_dict_remove(StateDict* sd, const char* key);

/* ---- Building from a Module ---- */

/*
 * Recursively collect all parameters from module into a new StateDict.
 * Keys are formed by joining module->name and param->name with '.'.
 * prefix is prepended to all keys (pass "" or NULL for none).
 */
StateDict* nn_get_state_dict(const Module* module, const char* prefix);

/*
 * Copy weights from sd into the matching parameters of module.
 * Only updates parameters whose key exists in sd; others are left unchanged.
 * strict=true: error if sd contains keys not present in module.
 * Returns 0 on success.
 */
int nn_load_state_dict(Module* module, const StateDict* sd, bool strict);

/* ---- Serialization (SafeTensors format) ---- */

/*
 * Save all tensors in sd to a SafeTensors file at path.
 * Returns 0 on success.
 */
int nn_save(const StateDict* sd, const char* path);

/*
 * Load a SafeTensors file and return a new StateDict.
 * Returns NULL on failure.
 */
StateDict* nn_load(const char* path);

/* ---- Utilities ---- */

/* Total number of scalar parameters across all entries. */
size_t nn_state_dict_num_params(const StateDict* sd);

/* Total memory used by all parameter tensors in bytes. */
size_t nn_state_dict_bytes(const StateDict* sd);

/* Print a summary table: key, shape, dtype, num_params. */
void nn_state_dict_print(const StateDict* sd);

/*
 * Copy weights between two matching state dicts (e.g. EMA update):
 *   dst[k] = alpha * src[k] + (1-alpha) * dst[k]
 * Returns 0 on success.
 */
int nn_state_dict_lerp(StateDict* dst, const StateDict* src, float alpha);

#ifdef __cplusplus
}
#endif

#endif /* CML_NN_STATE_H */
