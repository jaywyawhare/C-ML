#ifndef CML_TINYFS_H
#define CML_TINYFS_H

#include "tensor/tensor.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CML_TINYFS_MAX_SHARDS 64

typedef struct CMLTinyFSShard {
    char path[256];         /* Path to shard file */
    size_t offset;          /* Offset within shard */
    size_t size;            /* Size of data in this shard */
    int shard_id;
    bool is_remote;         /* Whether shard is on remote node */
    char host[128];         /* Remote host (if is_remote) */
    int port;               /* Remote port (if is_remote) */
} CMLTinyFSShard;

typedef struct CMLTinyFS {
    char base_path[256];
    CMLTinyFSShard shards[CML_TINYFS_MAX_SHARDS];
    int num_shards;
    int replication_factor;
    size_t shard_size;       /* Target shard size in bytes */
    bool initialized;
} CMLTinyFS;

CMLTinyFS* cml_tinyfs_create(const char* base_path, int num_shards, size_t shard_size);
void cml_tinyfs_free(CMLTinyFS* fs);
int cml_tinyfs_store(CMLTinyFS* fs, const char* name, Tensor* tensor);
Tensor* cml_tinyfs_load(CMLTinyFS* fs, const char* name);
bool cml_tinyfs_exists(CMLTinyFS* fs, const char* name);
int cml_tinyfs_delete(CMLTinyFS* fs, const char* name);
char** cml_tinyfs_list(CMLTinyFS* fs, int* count);
size_t cml_tinyfs_used_bytes(const CMLTinyFS* fs);
void cml_tinyfs_print(const CMLTinyFS* fs);

#ifdef __cplusplus
}
#endif

#endif /* CML_TINYFS_H */
