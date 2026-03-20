#ifndef CML_BACKEND_USB3_GPU_H
#define CML_BACKEND_USB3_GPU_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CMLUSB3GPU {
    int fd;
    uint16_t vendor_id;
    uint16_t product_id;
    char device_name[128];
    bool connected;

    uint64_t bar0_addr;
    size_t bar0_size;
    void* bar0_map;

    uint8_t* bulk_buf;
    size_t bulk_buf_size;
    int ep_in;
    int ep_out;
} CMLUSB3GPU;

bool cml_usb3_gpu_available(void);
CMLUSB3GPU* cml_usb3_gpu_open(void);
void cml_usb3_gpu_close(CMLUSB3GPU* dev);

int cml_usb3_gpu_read32(CMLUSB3GPU* dev, uint64_t offset, uint32_t* value);
int cml_usb3_gpu_write32(CMLUSB3GPU* dev, uint64_t offset, uint32_t value);

int cml_usb3_gpu_upload(CMLUSB3GPU* dev, uint64_t gpu_addr, const void* data, size_t size);
int cml_usb3_gpu_download(CMLUSB3GPU* dev, uint64_t gpu_addr, void* data, size_t size);

int cml_usb3_gpu_scsi_cmd(CMLUSB3GPU* dev, const uint8_t* cdb, int cdb_len,
                          void* data, size_t data_size, bool is_write);

#ifdef __cplusplus
}
#endif

#endif /* CML_BACKEND_USB3_GPU_H */
