/**
 * @file usb_device.c
 * @brief USB device access (stub)
 */

#include "backend/usb_device.h"
#include <stdlib.h>
#include <string.h>

bool cml_usb_available(void) {
    /* Would check for libusb */
    return false;
}

int cml_usb_enumerate(CMLUSBDevice** devices, int* num_devices) {
    if (devices) *devices = NULL;
    if (num_devices) *num_devices = 0;
    return 0;
}

int cml_usb_open(CMLUSBDevice* device) {
    if (!device) return -1;
    return -1; /* Not available */
}

void cml_usb_close(CMLUSBDevice* device) {
    if (!device) return;
    device->is_open = false;
}

int cml_usb_send(CMLUSBDevice* device, const void* data, size_t size) {
    (void)data; (void)size;
    if (!device || !device->is_open) return -1;
    return -1;
}

int cml_usb_recv(CMLUSBDevice* device, void* buffer, size_t size, int timeout_ms) {
    (void)buffer; (void)size; (void)timeout_ms;
    if (!device || !device->is_open) return -1;
    return -1;
}

void cml_usb_free_devices(CMLUSBDevice* devices, int num_devices) {
    (void)num_devices;
    free(devices);
}
