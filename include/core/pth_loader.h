/**
 * @file pth_loader.h
 * @brief PyTorch .pth/.pt file loader
 *
 * Loads PyTorch model state dictionaries saved via torch.save().
 * Supports the ZIP-based format used by modern PyTorch (pickle protocol + tensors).
 * Handles float32, float16, bfloat16, and int tensors.
 */

#ifndef CML_PTH_LOADER_H
#define CML_PTH_LOADER_H

#include "tensor/tensor.h"
#include <stdbool.h>
#include <stddef.h>

struct Module; /* forward declaration */

#ifdef __cplusplus
extern "C" {
#endif

/** Maximum key length for state dict entries */
#define CML_PTH_MAX_KEY_LEN 256

/** State dict entry */
typedef struct CMLPthEntry {
    char key[CML_PTH_MAX_KEY_LEN];
    Tensor* tensor;
    DType original_dtype;
    size_t storage_offset;
    size_t num_elements;
} CMLPthEntry;

/** Loaded state dict */
typedef struct CMLPthStateDict {
    CMLPthEntry* entries;
    int num_entries;
    int entry_capacity;

    /* Metadata */
    char* model_name;          /* If available from pickle data */
    int pytorch_version;       /* Detected PyTorch version */
    bool is_half_precision;    /* Model uses fp16 */
} CMLPthStateDict;

/** Open and parse a .pth/.pt file */
CMLPthStateDict* cml_pth_load(const char* path);

/** Free loaded state dict */
void cml_pth_free(CMLPthStateDict* sd);

/** Get tensor by key name */
Tensor* cml_pth_get_tensor(const CMLPthStateDict* sd, const char* key);

/** Get number of entries */
int cml_pth_num_entries(const CMLPthStateDict* sd);

/** Get entry key by index */
const char* cml_pth_get_key(const CMLPthStateDict* sd, int index);

/** Check if key exists */
bool cml_pth_has_key(const CMLPthStateDict* sd, const char* key);

/** List all keys (returns array of strings, sets count) */
const char** cml_pth_list_keys(const CMLPthStateDict* sd, int* count);

/** Load state dict into a Module */
int cml_pth_load_into_module(const CMLPthStateDict* sd, struct Module* module);

/** Print state dict summary */
void cml_pth_print(const CMLPthStateDict* sd);

/** Get total parameter count */
size_t cml_pth_total_params(const CMLPthStateDict* sd);

/** Get total memory size in bytes */
size_t cml_pth_total_bytes(const CMLPthStateDict* sd);

#ifdef __cplusplus
}
#endif

#endif /* CML_PTH_LOADER_H */
