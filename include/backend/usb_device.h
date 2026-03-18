#ifndef CML_USB_DEVICE_H
#define CML_USB_DEVICE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CMLUSBDevice {
    uint16_t vendor_id;
    uint16_t product_id;
    char product_name[128];
    char serial[64];
    int bus;
    int port;
    bool is_open;
    void* handle;             /* libusb device handle */
    int interface_num;
    uint8_t endpoint_in;
    uint8_t endpoint_out;
} CMLUSBDevice;

bool cml_usb_available(void);
int cml_usb_enumerate(CMLUSBDevice** devices, int* num_devices);
int cml_usb_open(CMLUSBDevice* device);
void cml_usb_close(CMLUSBDevice* device);
int cml_usb_send(CMLUSBDevice* device, const void* data, size_t size);
int cml_usb_recv(CMLUSBDevice* device, void* buffer, size_t size, int timeout_ms);
void cml_usb_free_devices(CMLUSBDevice* devices, int num_devices);

#ifdef __cplusplus
}
#endif

#endif /* CML_USB_DEVICE_H */
