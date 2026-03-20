#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CMLDiskCache CMLDiskCache;

CMLDiskCache* cml_disk_cache_open(const char* path);
void cml_disk_cache_close(CMLDiskCache* cache);

int cml_disk_cache_put(CMLDiskCache* cache, uint64_t hash, const void* data, size_t size);
int cml_disk_cache_get(CMLDiskCache* cache, uint64_t hash, void** out_data, size_t* out_size);
bool cml_disk_cache_has(CMLDiskCache* cache, uint64_t hash);

int cml_disk_cache_count(CMLDiskCache* cache);
int cml_disk_cache_clear(CMLDiskCache* cache);

bool cml_disk_cache_enabled(void);

#ifdef __cplusplus
}
#endif
