#ifndef CML_GPU_NV_MOCK_H
#define CML_GPU_NV_MOCK_H

#ifdef CML_NV_MOCK_GPU

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CMLNVMockGPU {
    uint32_t gpu_arch;
    uint32_t compute_cap_major;
    uint32_t compute_cap_minor;
    size_t   vram_size;
    char     name[64];

    uint32_t next_handle;
    void**   alloc_table;
    int      num_allocs;
    int      alloc_capacity;

    volatile uint32_t* last_semaphore;
    uint32_t last_semaphore_value;
    bool     auto_complete;
} CMLNVMockGPU;

void cml_nv_mock_init(CMLNVMockGPU* config);
void cml_nv_mock_shutdown(void);

CMLNVMockGPU* cml_nv_mock_get(void);

void cml_nv_mock_complete_kernel(void);

int   cml_nv_mock_open(const char* path, int flags, ...);
int   cml_nv_mock_close(int fd);
int   cml_nv_mock_ioctl(int fd, unsigned long request, void* arg);
void* cml_nv_mock_mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset);
int   cml_nv_mock_munmap(void* addr, size_t length);

#ifdef __cplusplus
}
#endif

#endif /* CML_NV_MOCK_GPU */
#endif /* CML_GPU_NV_MOCK_H */
