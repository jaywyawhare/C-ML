#ifndef CML_GPU_AM_MOCK_H
#define CML_GPU_AM_MOCK_H

#ifdef CML_AM_MOCK_GPU

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <dirent.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CMLAMMockGPU {
    uint32_t gpu_id;
    char     name[64];
    char     gfx_version[32];
    int      cu_count;
    size_t   vram_size;
    uint32_t lds_size_per_cu;
    int      sdma_count;
    uint32_t simd_per_cu;
    uint32_t max_waves_per_simd;

    uint64_t next_handle;
    uint32_t next_queue_id;
    void**   alloc_table;
    int      num_allocs;
    int      alloc_capacity;

    bool     auto_complete;
    int      dispatches_seen;
    int      sdma_copies_seen;

    char     topology_dir[256];
} CMLAMMockGPU;

void          cml_am_mock_init(CMLAMMockGPU* config);
void          cml_am_mock_shutdown(void);
CMLAMMockGPU* cml_am_mock_get(void);
void          cml_am_mock_complete_dispatch(void);

int   cml_am_mock_open(const char* path, int flags, ...);
int   cml_am_mock_close(int fd);
int   cml_am_mock_ioctl(int fd, unsigned long request, void* arg);
void* cml_am_mock_mmap(void* addr, size_t length, int prot, int flags,
                        int fd, off_t offset);
int   cml_am_mock_munmap(void* addr, size_t length);
FILE* cml_am_mock_fopen(const char* path, const char* mode);
int   cml_am_mock_access(const char* path, int mode);
DIR*  cml_am_mock_opendir(const char* path);

#ifdef __cplusplus
}
#endif

#endif /* CML_AM_MOCK_GPU */
#endif /* CML_GPU_AM_MOCK_H */
