#include "core/tinyfs.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

CMLTinyFS* cml_tinyfs_create(const char* base_path, int num_shards, size_t shard_size) {
    if (!base_path || num_shards < 1 || num_shards > CML_TINYFS_MAX_SHARDS)
        return NULL;

    CMLTinyFS* fs = (CMLTinyFS*)calloc(1, sizeof(CMLTinyFS));
    if (!fs) return NULL;

    strncpy(fs->base_path, base_path, sizeof(fs->base_path) - 1);
    fs->num_shards = num_shards;
    fs->shard_size = shard_size;
    fs->replication_factor = 1;
    fs->initialized = true;

    for (int i = 0; i < num_shards; i++) {
        snprintf(fs->shards[i].path, sizeof(fs->shards[i].path),
                 "%s/shard_%03d.bin", base_path, i);
        fs->shards[i].shard_id = i;
        fs->shards[i].is_remote = false;
    }

    return fs;
}

void cml_tinyfs_free(CMLTinyFS* fs) {
    free(fs);
}

int cml_tinyfs_store(CMLTinyFS* fs, const char* name, Tensor* tensor) {
    if (!fs || !fs->initialized || !name || !tensor) return -1;

    size_t data_size = tensor->numel * sizeof(float);

    /* Determine target shard based on hash of name */
    unsigned hash = 0;
    for (const char* p = name; *p; p++)
        hash = hash * 31 + (unsigned)*p;
    int shard_id = (int)(hash % (unsigned)fs->num_shards);

    char path[512];
    snprintf(path, sizeof(path), "%s/%s.tfs", fs->shards[shard_id].path, name);

    char actual_path[512];
    snprintf(actual_path, sizeof(actual_path), "%s/%s.tfs", fs->base_path, name);

    FILE* f = fopen(actual_path, "wb");
    if (!f) return -1;

    /* Header: ndim, shape, dtype */
    fwrite(&tensor->ndim, sizeof(int), 1, f);
    fwrite(tensor->shape, sizeof(int), (size_t)tensor->ndim, f);
    int dtype = (int)tensor->dtype;
    fwrite(&dtype, sizeof(int), 1, f);
    fwrite(&data_size, sizeof(size_t), 1, f);
    if (tensor->data)
        fwrite(tensor->data, 1, data_size, f);

    fclose(f);
    fs->shards[shard_id].size += data_size;
    return 0;
}

Tensor* cml_tinyfs_load(CMLTinyFS* fs, const char* name) {
    if (!fs || !fs->initialized || !name) return NULL;

    char path[512];
    snprintf(path, sizeof(path), "%s/%s.tfs", fs->base_path, name);

    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    int ndim;
    if (fread(&ndim, sizeof(int), 1, f) != 1) { fclose(f); return NULL; }
    if (ndim <= 0 || ndim > 8) { fclose(f); return NULL; }

    int shape[8];
    if (fread(shape, sizeof(int), (size_t)ndim, f) != (size_t)ndim) { fclose(f); return NULL; }

    int dtype;
    if (fread(&dtype, sizeof(int), 1, f) != 1) { fclose(f); return NULL; }

    size_t data_size;
    if (fread(&data_size, sizeof(size_t), 1, f) != 1) { fclose(f); return NULL; }

    TensorConfig tc = {0};
    Tensor* t = tensor_empty(shape, ndim, &tc);
    if (!t) { fclose(f); return NULL; }

    if (t->data && data_size > 0) {
        size_t read = fread(t->data, 1, data_size, f);
        (void)read;
    }

    fclose(f);
    return t;
}

bool cml_tinyfs_exists(CMLTinyFS* fs, const char* name) {
    if (!fs || !name) return false;
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.tfs", fs->base_path, name);
    FILE* f = fopen(path, "rb");
    if (f) { fclose(f); return true; }
    return false;
}

int cml_tinyfs_delete(CMLTinyFS* fs, const char* name) {
    if (!fs || !name) return -1;
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.tfs", fs->base_path, name);
    return remove(path);
}

char** cml_tinyfs_list(CMLTinyFS* fs, int* count) {
    (void)fs;
    if (count) *count = 0;
    return NULL; /* Would require directory listing */
}

size_t cml_tinyfs_used_bytes(const CMLTinyFS* fs) {
    if (!fs) return 0;
    size_t total = 0;
    for (int i = 0; i < fs->num_shards; i++)
        total += fs->shards[i].size;
    return total;
}

void cml_tinyfs_print(const CMLTinyFS* fs) {
    if (!fs) { printf("TinyFS: NULL\n"); return; }
    printf("TinyFS\n");
    printf("Base path: %s\n", fs->base_path);
    printf("Shards: %d, Shard size: %zu bytes\n", fs->num_shards, fs->shard_size);
    printf("Replication: %d\n", fs->replication_factor);
    printf("Used: %zu bytes\n", cml_tinyfs_used_bytes(fs));
    printf("\n");
}
