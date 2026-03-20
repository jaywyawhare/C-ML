#include "backend/usb3_gpu.h"
#include "core/logging.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/usbdevice_fs.h>

#define ASM2464PD_VENDOR     0x174c
#define ASM2464PD_PRODUCT    0x2362

#define USB3_MAX_PACKET      1024
#define USB3_BULK_BUF_SIZE   (64 * 1024)
#define USB3_TIMEOUT_MS      5000

#define SCSI_VENDOR_CMD      0xE4
#define SCSI_PCIE_READ       0x01
#define SCSI_PCIE_WRITE      0x02
#define SCSI_BAR_READ        0x03
#define SCSI_BAR_WRITE       0x04

#define DEFAULT_BAR0_SIZE    (16 * 1024 * 1024)
#define DEFAULT_EP_IN        0x81
#define DEFAULT_EP_OUT       0x02

#pragma pack(push, 1)
typedef struct {
    uint8_t  bLength;
    uint8_t  bDescriptorType;
    uint16_t bcdUSB;
    uint8_t  bDeviceClass;
    uint8_t  bDeviceSubClass;
    uint8_t  bDeviceProtocol;
    uint8_t  bMaxPacketSize0;
    uint16_t idVendor;
    uint16_t idProduct;
    uint16_t bcdDevice;
    uint8_t  iManufacturer;
    uint8_t  iProduct;
    uint8_t  iSerialNumber;
    uint8_t  bNumConfigurations;
} usb3_dev_desc_t;
#pragma pack(pop)

static int read_dev_descriptor(int fd, usb3_dev_desc_t* desc) {
    /* usbfs exposes the descriptor at the start of the device file */
    if (lseek(fd, 0, SEEK_SET) != 0) return -1;
    ssize_t n = read(fd, desc, sizeof(*desc));
    if (n < (ssize_t)sizeof(*desc)) return -1;
    return 0;
}

static int usb3_claim_interface(int fd, int iface) {
    return ioctl(fd, USBDEVFS_CLAIMINTERFACE, &iface);
}

static int usb3_release_interface(int fd, int iface) {
    return ioctl(fd, USBDEVFS_RELEASEINTERFACE, &iface);
}

static int usb3_bulk_xfer(int fd, int ep, void* data, size_t size, int timeout_ms) {
    struct usbdevfs_bulktransfer bulk = {
        .ep = (unsigned int)ep,
        .len = (unsigned int)size,
        .timeout = (unsigned int)timeout_ms,
        .data = data,
    };
    int ret = ioctl(fd, USBDEVFS_BULK, &bulk);
    return ret;
}

static int scan_usb_bus(char* path, size_t path_size) {
    DIR* buses = opendir("/dev/bus/usb");
    if (!buses) return -1;

    struct dirent* bus_ent;
    while ((bus_ent = readdir(buses)) != NULL) {
        if (bus_ent->d_name[0] == '.') continue;

        char bus_path[256];
        snprintf(bus_path, sizeof(bus_path), "/dev/bus/usb/%s", bus_ent->d_name);
        DIR* devs = opendir(bus_path);
        if (!devs) continue;

        struct dirent* dev_ent;
        while ((dev_ent = readdir(devs)) != NULL) {
            if (dev_ent->d_name[0] == '.') continue;

            char dev_path[512];
            snprintf(dev_path, sizeof(dev_path), "%s/%s", bus_path, dev_ent->d_name);

            int fd = open(dev_path, O_RDWR);
            if (fd < 0) continue;

            usb3_dev_desc_t desc;
            if (read_dev_descriptor(fd, &desc) == 0 && desc.idVendor == ASM2464PD_VENDOR) {
                snprintf(path, path_size, "%s", dev_path);
                close(fd);
                closedir(devs);
                closedir(buses);
                return 0;
            }
            close(fd);
        }
        closedir(devs);
    }
    closedir(buses);
    return -1;
}

bool cml_usb3_gpu_available(void) {
    char path[512];
    return scan_usb_bus(path, sizeof(path)) == 0;
}

CMLUSB3GPU* cml_usb3_gpu_open(void) {
    char path[512];
    if (scan_usb_bus(path, sizeof(path)) != 0) {
        LOG_DEBUG("USB3 GPU: no ASM2464PD bridge found");
        return NULL;
    }

    int fd = open(path, O_RDWR);
    if (fd < 0) {
        LOG_ERROR("USB3 GPU: failed to open %s", path);
        return NULL;
    }

    usb3_dev_desc_t desc;
    if (read_dev_descriptor(fd, &desc) != 0) {
        LOG_ERROR("USB3 GPU: failed to read descriptor");
        close(fd);
        return NULL;
    }

    if (usb3_claim_interface(fd, 0) != 0) {
        LOG_ERROR("USB3 GPU: failed to claim interface 0");
        close(fd);
        return NULL;
    }

    CMLUSB3GPU* dev = (CMLUSB3GPU*)calloc(1, sizeof(CMLUSB3GPU));
    if (!dev) {
        usb3_release_interface(fd, 0);
        close(fd);
        return NULL;
    }

    dev->fd = fd;
    dev->vendor_id = desc.idVendor;
    dev->product_id = desc.idProduct;
    snprintf(dev->device_name, sizeof(dev->device_name), "ASM2464PD PCIe-USB3 Bridge (pid=0x%04x)", desc.idProduct);
    dev->connected = true;
    dev->bar0_addr = 0;
    dev->bar0_size = DEFAULT_BAR0_SIZE;
    dev->bar0_map = NULL;
    dev->ep_in = DEFAULT_EP_IN;
    dev->ep_out = DEFAULT_EP_OUT;

    dev->bulk_buf_size = USB3_BULK_BUF_SIZE;
    dev->bulk_buf = (uint8_t*)malloc(dev->bulk_buf_size);
    if (!dev->bulk_buf) {
        usb3_release_interface(fd, 0);
        close(fd);
        free(dev);
        return NULL;
    }

    LOG_INFO("USB3 GPU: opened %s", dev->device_name);
    return dev;
}

void cml_usb3_gpu_close(CMLUSB3GPU* dev) {
    if (!dev) return;

    if (dev->bar0_map && dev->bar0_size > 0) {
        munmap(dev->bar0_map, dev->bar0_size);
        dev->bar0_map = NULL;
    }

    if (dev->fd >= 0) {
        usb3_release_interface(dev->fd, 0);
        close(dev->fd);
    }

    free(dev->bulk_buf);
    free(dev);
}

int cml_usb3_gpu_scsi_cmd(CMLUSB3GPU* dev, const uint8_t* cdb, int cdb_len,
                          void* data, size_t data_size, bool is_write) {
    if (!dev || !dev->connected || dev->fd < 0) return -1;
    if (!cdb || cdb_len <= 0 || cdb_len > 16) return -1;

    /* Build SCSI passthrough via usbfs control transfer.
     * The ASM2464PD accepts vendor-specific SCSI commands over bulk endpoints. */
    uint8_t cmd_buf[16 + sizeof(uint32_t)];
    memset(cmd_buf, 0, sizeof(cmd_buf));
    memcpy(cmd_buf, cdb, (size_t)cdb_len);

    /* Send command block */
    int ret = usb3_bulk_xfer(dev->fd, dev->ep_out, cmd_buf, (size_t)cdb_len, USB3_TIMEOUT_MS);
    if (ret < 0) {
        LOG_ERROR("USB3 GPU: SCSI cmd send failed");
        return -1;
    }

    if (data && data_size > 0) {
        if (is_write) {
            size_t offset = 0;
            while (offset < data_size) {
                size_t chunk = data_size - offset;
                if (chunk > USB3_MAX_PACKET) chunk = USB3_MAX_PACKET;
                ret = usb3_bulk_xfer(dev->fd, dev->ep_out, (uint8_t*)data + offset, chunk, USB3_TIMEOUT_MS);
                if (ret < 0) {
                    LOG_ERROR("USB3 GPU: SCSI write data failed at offset %zu", offset);
                    return -1;
                }
                offset += chunk;
            }
        } else {
            size_t offset = 0;
            while (offset < data_size) {
                size_t chunk = data_size - offset;
                if (chunk > USB3_MAX_PACKET) chunk = USB3_MAX_PACKET;
                ret = usb3_bulk_xfer(dev->fd, dev->ep_in, (uint8_t*)data + offset, chunk, USB3_TIMEOUT_MS);
                if (ret < 0) {
                    LOG_ERROR("USB3 GPU: SCSI read data failed at offset %zu", offset);
                    return -1;
                }
                offset += chunk;
            }
        }
    }

    return 0;
}

int cml_usb3_gpu_read32(CMLUSB3GPU* dev, uint64_t offset, uint32_t* value) {
    if (!dev || !dev->connected || !value) return -1;

    uint8_t cdb[10];
    memset(cdb, 0, sizeof(cdb));
    cdb[0] = SCSI_VENDOR_CMD;
    cdb[1] = SCSI_BAR_READ;
    cdb[2] = (uint8_t)((offset >> 24) & 0xFF);
    cdb[3] = (uint8_t)((offset >> 16) & 0xFF);
    cdb[4] = (uint8_t)((offset >> 8) & 0xFF);
    cdb[5] = (uint8_t)(offset & 0xFF);
    cdb[6] = 0;
    cdb[7] = 0;
    cdb[8] = 4;
    cdb[9] = 0;

    uint32_t buf = 0;
    int ret = cml_usb3_gpu_scsi_cmd(dev, cdb, 10, &buf, sizeof(buf), false);
    if (ret != 0) return -1;

    *value = buf;
    return 0;
}

int cml_usb3_gpu_write32(CMLUSB3GPU* dev, uint64_t offset, uint32_t value) {
    if (!dev || !dev->connected) return -1;

    uint8_t cdb[10];
    memset(cdb, 0, sizeof(cdb));
    cdb[0] = SCSI_VENDOR_CMD;
    cdb[1] = SCSI_BAR_WRITE;
    cdb[2] = (uint8_t)((offset >> 24) & 0xFF);
    cdb[3] = (uint8_t)((offset >> 16) & 0xFF);
    cdb[4] = (uint8_t)((offset >> 8) & 0xFF);
    cdb[5] = (uint8_t)(offset & 0xFF);
    cdb[6] = 0;
    cdb[7] = 0;
    cdb[8] = 4;
    cdb[9] = 0;

    uint32_t buf = value;
    return cml_usb3_gpu_scsi_cmd(dev, cdb, 10, &buf, sizeof(buf), true);
}

int cml_usb3_gpu_upload(CMLUSB3GPU* dev, uint64_t gpu_addr, const void* data, size_t size) {
    if (!dev || !dev->connected || !data || size == 0) return -1;

    const uint8_t* src = (const uint8_t*)data;
    size_t offset = 0;

    while (offset < size) {
        size_t chunk = size - offset;
        if (chunk > dev->bulk_buf_size) chunk = dev->bulk_buf_size;

        uint8_t cdb[16];
        memset(cdb, 0, sizeof(cdb));
        cdb[0] = SCSI_VENDOR_CMD;
        cdb[1] = SCSI_PCIE_WRITE;

        uint64_t addr = gpu_addr + offset;
        cdb[2] = (uint8_t)((addr >> 56) & 0xFF);
        cdb[3] = (uint8_t)((addr >> 48) & 0xFF);
        cdb[4] = (uint8_t)((addr >> 40) & 0xFF);
        cdb[5] = (uint8_t)((addr >> 32) & 0xFF);
        cdb[6] = (uint8_t)((addr >> 24) & 0xFF);
        cdb[7] = (uint8_t)((addr >> 16) & 0xFF);
        cdb[8] = (uint8_t)((addr >> 8) & 0xFF);
        cdb[9] = (uint8_t)(addr & 0xFF);

        uint32_t len32 = (uint32_t)chunk;
        cdb[10] = (uint8_t)((len32 >> 24) & 0xFF);
        cdb[11] = (uint8_t)((len32 >> 16) & 0xFF);
        cdb[12] = (uint8_t)((len32 >> 8) & 0xFF);
        cdb[13] = (uint8_t)(len32 & 0xFF);

        memcpy(dev->bulk_buf, src + offset, chunk);
        int ret = cml_usb3_gpu_scsi_cmd(dev, cdb, 14, dev->bulk_buf, chunk, true);
        if (ret != 0) {
            LOG_ERROR("USB3 GPU: upload failed at offset %zu", offset);
            return -1;
        }
        offset += chunk;
    }

    return 0;
}

int cml_usb3_gpu_download(CMLUSB3GPU* dev, uint64_t gpu_addr, void* data, size_t size) {
    if (!dev || !dev->connected || !data || size == 0) return -1;

    uint8_t* dst = (uint8_t*)data;
    size_t offset = 0;

    while (offset < size) {
        size_t chunk = size - offset;
        if (chunk > dev->bulk_buf_size) chunk = dev->bulk_buf_size;

        uint8_t cdb[16];
        memset(cdb, 0, sizeof(cdb));
        cdb[0] = SCSI_VENDOR_CMD;
        cdb[1] = SCSI_PCIE_READ;

        uint64_t addr = gpu_addr + offset;
        cdb[2] = (uint8_t)((addr >> 56) & 0xFF);
        cdb[3] = (uint8_t)((addr >> 48) & 0xFF);
        cdb[4] = (uint8_t)((addr >> 40) & 0xFF);
        cdb[5] = (uint8_t)((addr >> 32) & 0xFF);
        cdb[6] = (uint8_t)((addr >> 24) & 0xFF);
        cdb[7] = (uint8_t)((addr >> 16) & 0xFF);
        cdb[8] = (uint8_t)((addr >> 8) & 0xFF);
        cdb[9] = (uint8_t)(addr & 0xFF);

        uint32_t len32 = (uint32_t)chunk;
        cdb[10] = (uint8_t)((len32 >> 24) & 0xFF);
        cdb[11] = (uint8_t)((len32 >> 16) & 0xFF);
        cdb[12] = (uint8_t)((len32 >> 8) & 0xFF);
        cdb[13] = (uint8_t)(len32 & 0xFF);

        int ret = cml_usb3_gpu_scsi_cmd(dev, cdb, 14, dev->bulk_buf, chunk, false);
        if (ret != 0) {
            LOG_ERROR("USB3 GPU: download failed at offset %zu", offset);
            return -1;
        }
        memcpy(dst + offset, dev->bulk_buf, chunk);
        offset += chunk;
    }

    return 0;
}

#else /* !__linux__ */

bool cml_usb3_gpu_available(void) { return false; }
CMLUSB3GPU* cml_usb3_gpu_open(void) { return NULL; }
void cml_usb3_gpu_close(CMLUSB3GPU* dev) { (void)dev; }

int cml_usb3_gpu_read32(CMLUSB3GPU* dev, uint64_t offset, uint32_t* value) {
    (void)dev; (void)offset; (void)value;
    return -1;
}

int cml_usb3_gpu_write32(CMLUSB3GPU* dev, uint64_t offset, uint32_t value) {
    (void)dev; (void)offset; (void)value;
    return -1;
}

int cml_usb3_gpu_upload(CMLUSB3GPU* dev, uint64_t gpu_addr, const void* data, size_t size) {
    (void)dev; (void)gpu_addr; (void)data; (void)size;
    return -1;
}

int cml_usb3_gpu_download(CMLUSB3GPU* dev, uint64_t gpu_addr, void* data, size_t size) {
    (void)dev; (void)gpu_addr; (void)data; (void)size;
    return -1;
}

int cml_usb3_gpu_scsi_cmd(CMLUSB3GPU* dev, const uint8_t* cdb, int cdb_len,
                          void* data, size_t data_size, bool is_write) {
    (void)dev; (void)cdb; (void)cdb_len; (void)data; (void)data_size; (void)is_write;
    return -1;
}

#endif /* __linux__ */
