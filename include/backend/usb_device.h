/**
 * @file usb_device.h
 * @brief USB device access for external accelerators
 *
 * Provides a transport layer for communicating with USB-connected
 * ML accelerators (e.g., Google Coral, Intel NCS).
 */

#ifndef CML_USB_DEVICE_H
#define CML_USB_DEVICE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** USB device descriptor */
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

/** Check if USB support is available */
bool cml_usb_available(void);

/** Enumerate connected USB accelerators */
int cml_usb_enumerate(CMLUSBDevice** devices, int* num_devices);

/** Open a USB device */
int cml_usb_open(CMLUSBDevice* device);

/** Close a USB device */
void cml_usb_close(CMLUSBDevice* device);

/** Send data to USB device */
int cml_usb_send(CMLUSBDevice* device, const void* data, size_t size);

/** Receive data from USB device */
int cml_usb_recv(CMLUSBDevice* device, void* buffer, size_t size, int timeout_ms);

/** Free device list from cml_usb_enumerate */
void cml_usb_free_devices(CMLUSBDevice* devices, int num_devices);

#ifdef __cplusplus
}
#endif

#endif /* CML_USB_DEVICE_H */
