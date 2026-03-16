/**
 * @file pth_loader.c
 * @brief PyTorch .pth/.pt file loader implementation
 *
 * PyTorch saves models as ZIP archives containing:
 *   - archive/data.pkl (pickle-serialized state dict structure)
 *   - archive/data/0, archive/data/1, ... (raw tensor storage)
 *
 * This is a minimal loader that handles common float32 state dicts.
 * For full pickle support, consider using an external pickle parser.
 */

#include "core/pth_loader.h"
#include "tensor/tensor.h"
#include "nn.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

/* Minimal ZIP local file header parsing */
#define ZIP_LOCAL_HEADER_SIG 0x04034b50
#define ZIP_CENTRAL_DIR_SIG  0x02014b50
#define ZIP_END_CENTRAL_SIG  0x06054b50

typedef struct {
    uint32_t signature;
    uint16_t version_needed;
    uint16_t flags;
    uint16_t compression;
    uint16_t mod_time;
    uint16_t mod_date;
    uint32_t crc32;
    uint32_t compressed_size;
    uint32_t uncompressed_size;
    uint16_t filename_len;
    uint16_t extra_len;
} __attribute__((packed)) ZipLocalHeader;

/* Read little-endian uint32 */
static uint32_t read_u32_le(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint16_t read_u16_le(const uint8_t* p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

/* Simple pickle opcode scanner for state dict keys */
typedef struct {
    char** keys;
    int num_keys;
    int key_capacity;
    /* Storage info: maps storage index to dtype + size */
    int* storage_dtypes;
    size_t* storage_sizes;
    int num_storages;
} PickleScanResult;

/*
 * Minimal pickle scanner: looks for SHORT_BINUNICODE opcodes
 * to extract parameter name strings from the pickle stream.
 * This handles the common case of torch.save(model.state_dict(), path).
 */
static PickleScanResult* scan_pickle_keys(const uint8_t* data, size_t size) {
    PickleScanResult* result = (PickleScanResult*)calloc(1, sizeof(PickleScanResult));
    if (!result) return NULL;

    result->key_capacity = 64;
    result->keys = (char**)calloc((size_t)result->key_capacity, sizeof(char*));
    if (!result->keys) { free(result); return NULL; }

    size_t pos = 0;
    while (pos < size) {
        uint8_t op = data[pos++];

        /* SHORT_BINUNICODE: 1-byte length + string */
        if (op == 0x8c && pos < size) {
            uint8_t len = data[pos++];
            if (pos + len <= size && len > 0) {
                /* Check if this looks like a parameter key (contains '.weight' or '.bias') */
                char* str = (char*)malloc(len + 1);
                if (str) {
                    memcpy(str, &data[pos], len);
                    str[len] = '\0';

                    /* Keep strings that look like parameter keys */
                    if (strstr(str, ".weight") || strstr(str, ".bias") ||
                        strstr(str, ".running_mean") || strstr(str, ".running_var") ||
                        strstr(str, ".num_batches_tracked") ||
                        strstr(str, "embedding")) {
                        if (result->num_keys >= result->key_capacity) {
                            result->key_capacity *= 2;
                            char** new_keys = (char**)realloc(result->keys,
                                (size_t)result->key_capacity * sizeof(char*));
                            if (new_keys) result->keys = new_keys;
                        }
                        if (result->num_keys < result->key_capacity)
                            result->keys[result->num_keys++] = str;
                        else
                            free(str);
                    } else {
                        free(str);
                    }
                }
                pos += len;
            }
        }
        /* BINUNICODE: 4-byte length + string */
        else if (op == 0x8d && pos + 4 <= size) {
            uint32_t len = read_u32_le(&data[pos]);
            pos += 4;
            if (pos + len <= size && len > 0 && len < 1024) {
                char* str = (char*)malloc(len + 1);
                if (str) {
                    memcpy(str, &data[pos], len);
                    str[len] = '\0';
                    if (strstr(str, ".weight") || strstr(str, ".bias")) {
                        if (result->num_keys < result->key_capacity)
                            result->keys[result->num_keys++] = str;
                        else
                            free(str);
                    } else {
                        free(str);
                    }
                }
                pos += len;
            }
        }
    }

    return result;
}

static void free_pickle_result(PickleScanResult* result) {
    if (!result) return;
    for (int i = 0; i < result->num_keys; i++)
        free(result->keys[i]);
    free(result->keys);
    free(result->storage_dtypes);
    free(result->storage_sizes);
    free(result);
}

CMLPthStateDict* cml_pth_load(const char* path) {
    if (!path) return NULL;

    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    /* Read entire file */
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size <= 0 || file_size > (long)(2ULL * 1024 * 1024 * 1024)) {
        fclose(f);
        return NULL;
    }

    uint8_t* file_data = (uint8_t*)malloc((size_t)file_size);
    if (!file_data) { fclose(f); return NULL; }

    size_t read_size = fread(file_data, 1, (size_t)file_size, f);
    fclose(f);
    if (read_size != (size_t)file_size) { free(file_data); return NULL; }

    /* Create state dict */
    CMLPthStateDict* sd = (CMLPthStateDict*)calloc(1, sizeof(CMLPthStateDict));
    if (!sd) { free(file_data); return NULL; }

    sd->entry_capacity = 64;
    sd->entries = (CMLPthEntry*)calloc((size_t)sd->entry_capacity, sizeof(CMLPthEntry));
    if (!sd->entries) { free(sd); free(file_data); return NULL; }

    /* Scan for ZIP entries */
    PickleScanResult* pickle_keys = NULL;
    int data_file_count = 0;

    /* Track raw data file positions */
    size_t data_offsets[256];
    size_t data_sizes[256];
    memset(data_offsets, 0, sizeof(data_offsets));
    memset(data_sizes, 0, sizeof(data_sizes));

    size_t pos = 0;
    while (pos + sizeof(ZipLocalHeader) < (size_t)file_size) {
        uint32_t sig = read_u32_le(&file_data[pos]);
        if (sig != ZIP_LOCAL_HEADER_SIG) break;

        uint16_t version = read_u16_le(&file_data[pos + 4]);
        (void)version;
        uint16_t compression = read_u16_le(&file_data[pos + 8]);
        uint32_t comp_size = read_u32_le(&file_data[pos + 18]);
        uint32_t uncomp_size = read_u32_le(&file_data[pos + 22]);
        uint16_t fname_len = read_u16_le(&file_data[pos + 26]);
        uint16_t extra_len = read_u16_le(&file_data[pos + 28]);

        size_t header_size = 30;
        size_t data_start = pos + header_size + fname_len + extra_len;

        char fname[512];
        size_t copy_len = fname_len < 511 ? fname_len : 511;
        memcpy(fname, &file_data[pos + header_size], copy_len);
        fname[copy_len] = '\0';

        /* Look for pickle file */
        if (strstr(fname, ".pkl") && compression == 0 && uncomp_size > 0) {
            pickle_keys = scan_pickle_keys(&file_data[data_start], uncomp_size);
        }

        /* Look for raw data files (archive/data/0, archive/data/1, ...) */
        char* data_part = strstr(fname, "data/");
        if (data_part && compression == 0 && uncomp_size > 0) {
            char* num_str = data_part + 5;
            int idx = atoi(num_str);
            if (idx >= 0 && idx < 256) {
                data_offsets[idx] = data_start;
                data_sizes[idx] = uncomp_size;
                if (idx >= data_file_count) data_file_count = idx + 1;
            }
        }

        pos = data_start + (comp_size > 0 ? comp_size : uncomp_size);
    }

    /* Match keys to data files */
    if (pickle_keys) {
        int data_idx = 0;
        for (int i = 0; i < pickle_keys->num_keys && data_idx < data_file_count; i++) {
            if (data_sizes[data_idx] == 0) { data_idx++; i--; continue; }

            if (sd->num_entries >= sd->entry_capacity) {
                sd->entry_capacity *= 2;
                CMLPthEntry* new_entries = (CMLPthEntry*)realloc(
                    sd->entries, (size_t)sd->entry_capacity * sizeof(CMLPthEntry));
                if (!new_entries) break;
                sd->entries = new_entries;
            }

            CMLPthEntry* entry = &sd->entries[sd->num_entries];
            memset(entry, 0, sizeof(*entry));
            strncpy(entry->key, pickle_keys->keys[i], CML_PTH_MAX_KEY_LEN - 1);
            entry->original_dtype = DTYPE_FLOAT32;
            entry->num_elements = data_sizes[data_idx] / sizeof(float);
            entry->storage_offset = data_offsets[data_idx];

            /* Create 1D tensor for now (shape unknown from minimal parsing) */
            int shape[1] = { (int)entry->num_elements };
            TensorConfig tc = {0};
            entry->tensor = tensor_empty(shape, 1, &tc);
            if (entry->tensor && entry->tensor->data) {
                memcpy(entry->tensor->data, &file_data[data_offsets[data_idx]],
                       data_sizes[data_idx]);
            }

            sd->num_entries++;
            data_idx++;
        }
        free_pickle_result(pickle_keys);
    }

    free(file_data);
    return sd;
}

void cml_pth_free(CMLPthStateDict* sd) {
    if (!sd) return;
    for (int i = 0; i < sd->num_entries; i++) {
        if (sd->entries[i].tensor)
            tensor_free(sd->entries[i].tensor);
    }
    free(sd->entries);
    free(sd->model_name);
    free(sd);
}

Tensor* cml_pth_get_tensor(const CMLPthStateDict* sd, const char* key) {
    if (!sd || !key) return NULL;
    for (int i = 0; i < sd->num_entries; i++) {
        if (strcmp(sd->entries[i].key, key) == 0)
            return sd->entries[i].tensor;
    }
    return NULL;
}

int cml_pth_num_entries(const CMLPthStateDict* sd) {
    return sd ? sd->num_entries : 0;
}

const char* cml_pth_get_key(const CMLPthStateDict* sd, int index) {
    if (!sd || index < 0 || index >= sd->num_entries) return NULL;
    return sd->entries[index].key;
}

bool cml_pth_has_key(const CMLPthStateDict* sd, const char* key) {
    return cml_pth_get_tensor(sd, key) != NULL;
}

const char** cml_pth_list_keys(const CMLPthStateDict* sd, int* count) {
    if (!sd || !count) return NULL;
    *count = sd->num_entries;
    const char** keys = (const char**)malloc((size_t)sd->num_entries * sizeof(char*));
    if (!keys) return NULL;
    for (int i = 0; i < sd->num_entries; i++)
        keys[i] = sd->entries[i].key;
    return keys;
}

int cml_pth_load_into_module(const CMLPthStateDict* sd, struct Module* module) {
    if (!sd || !module) return -1;
    if (!module->parameters || module->num_parameters == 0) return -1;

    int loaded = 0;

    /* Match state dict keys to module parameters by name */
    for (int i = 0; i < module->num_parameters; i++) {
        Parameter* param = module->parameters[i];
        if (!param || !param->name) continue;

        /* Try exact match first */
        Tensor* t = cml_pth_get_tensor(sd, param->name);

        /* Try suffix match: "layer.weight" matches "model.layer.weight" */
        if (!t) {
            size_t pname_len = strlen(param->name);
            for (int j = 0; j < sd->num_entries; j++) {
                const char* key = sd->entries[j].key;
                size_t klen = strlen(key);
                if (klen >= pname_len) {
                    const char* suffix = key + klen - pname_len;
                    if (strcmp(suffix, param->name) == 0 &&
                        (suffix == key || *(suffix - 1) == '.')) {
                        t = sd->entries[j].tensor;
                        break;
                    }
                }
            }
        }

        if (!t) continue;

        /* Verify shapes match */
        if (!param->tensor || param->tensor->numel != t->numel) continue;

        /* Copy data from state dict tensor to parameter tensor */
        void* dst = tensor_data_ptr(param->tensor);
        void* src = tensor_data_ptr(t);
        if (dst && src) {
            size_t data_size = t->numel * cml_dtype_size(t->dtype);
            memcpy(dst, src, data_size);
            loaded++;
        }
    }

    return loaded > 0 ? 0 : -1;
}

void cml_pth_print(const CMLPthStateDict* sd) {
    if (!sd) {
        printf("PthStateDict: NULL\n");
        return;
    }

    printf("=== PyTorch State Dict ===\n");
    printf("Entries: %d\n", sd->num_entries);
    printf("Total params: %zu\n", cml_pth_total_params(sd));
    printf("Total bytes: %.2f MB\n", (double)cml_pth_total_bytes(sd) / (1024.0 * 1024.0));
    printf("\nParameters:\n");
    for (int i = 0; i < sd->num_entries; i++) {
        CMLPthEntry* e = &sd->entries[i];
        printf("  %-40s elements=%zu\n", e->key, e->num_elements);
    }
    printf("==========================\n");
}

size_t cml_pth_total_params(const CMLPthStateDict* sd) {
    if (!sd) return 0;
    size_t total = 0;
    for (int i = 0; i < sd->num_entries; i++)
        total += sd->entries[i].num_elements;
    return total;
}

size_t cml_pth_total_bytes(const CMLPthStateDict* sd) {
    if (!sd) return 0;
    size_t total = 0;
    for (int i = 0; i < sd->num_entries; i++)
        total += sd->entries[i].num_elements * sizeof(float);
    return total;
}
