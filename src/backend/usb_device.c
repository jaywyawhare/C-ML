#include "backend/usb_device.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

/* libusb descriptor structure (partial, matches libusb_device_descriptor layout) */
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
} usb_device_descriptor_t;
#pragma pack(pop)

/* Known ML accelerator USB IDs */
#define CORAL_VENDOR_1      0x1a6e
#define CORAL_VENDOR_2      0x18d1
#define CORAL_PRODUCT       0x089a
#define NCS2_VENDOR         0x03e7
#define NCS2_PRODUCT        0x2485

/* Default bulk endpoints for ML accelerators */
#define DEFAULT_ENDPOINT_IN  0x81
#define DEFAULT_ENDPOINT_OUT 0x01
#define DEFAULT_INTERFACE    0

/* libusb function pointer types */
typedef int      (*libusb_init_fn)(void** ctx);
typedef void     (*libusb_exit_fn)(void* ctx);
typedef ssize_t  (*libusb_get_device_list_fn)(void* ctx, void*** list);
typedef void     (*libusb_free_device_list_fn)(void** list, int unref);
typedef int      (*libusb_get_device_descriptor_fn)(void* dev, void* desc);
typedef int      (*libusb_open_fn)(void* dev, void** handle);
typedef void     (*libusb_close_fn)(void* handle);
typedef int      (*libusb_claim_interface_fn)(void* handle, int iface);
typedef int      (*libusb_release_interface_fn)(void* handle, int iface);
typedef int      (*libusb_bulk_transfer_fn)(void* handle, uint8_t endpoint,
                                            uint8_t* data, int length,
                                            int* transferred, unsigned int timeout);
typedef uint8_t  (*libusb_get_bus_number_fn)(void* dev);
typedef uint8_t  (*libusb_get_port_number_fn)(void* dev);
typedef int      (*libusb_get_string_descriptor_ascii_fn)(void* handle, uint8_t index,
                                                          uint8_t* data, int length);

/* Loaded library and function pointers */
static void* s_usb_lib = NULL;

static libusb_init_fn                        fn_libusb_init                        = NULL;
static libusb_exit_fn                        fn_libusb_exit                        = NULL;
static libusb_get_device_list_fn             fn_libusb_get_device_list             = NULL;
static libusb_free_device_list_fn            fn_libusb_free_device_list            = NULL;
static libusb_get_device_descriptor_fn       fn_libusb_get_device_descriptor       = NULL;
static libusb_open_fn                        fn_libusb_open                        = NULL;
static libusb_close_fn                       fn_libusb_close                       = NULL;
static libusb_claim_interface_fn             fn_libusb_claim_interface             = NULL;
static libusb_release_interface_fn           fn_libusb_release_interface           = NULL;
static libusb_bulk_transfer_fn               fn_libusb_bulk_transfer               = NULL;
static libusb_get_bus_number_fn              fn_libusb_get_bus_number              = NULL;
static libusb_get_port_number_fn             fn_libusb_get_port_number             = NULL;
static libusb_get_string_descriptor_ascii_fn fn_libusb_get_string_descriptor_ascii = NULL;

static void* open_libusb(void) {
    void* h = dlopen("libusb-1.0.so", RTLD_LAZY);
    if (!h) {
        h = dlopen("libusb-1.0.so.0", RTLD_LAZY);
    }
    return h;
}

#define LOAD_USB_SYM(name) do { \
    fn_##name = (name##_fn)dlsym(lib, #name); \
    if (!fn_##name) { \
        LOG_ERROR("USB: failed to load %s: %s", #name, dlerror()); \
        return -1; \
    } \
} while (0)

static int load_libusb_symbols(void* lib) {
    LOAD_USB_SYM(libusb_init);
    LOAD_USB_SYM(libusb_exit);
    LOAD_USB_SYM(libusb_get_device_list);
    LOAD_USB_SYM(libusb_free_device_list);
    LOAD_USB_SYM(libusb_get_device_descriptor);
    LOAD_USB_SYM(libusb_open);
    LOAD_USB_SYM(libusb_close);
    LOAD_USB_SYM(libusb_claim_interface);
    LOAD_USB_SYM(libusb_release_interface);
    LOAD_USB_SYM(libusb_bulk_transfer);
    LOAD_USB_SYM(libusb_get_bus_number);
    LOAD_USB_SYM(libusb_get_port_number);
    LOAD_USB_SYM(libusb_get_string_descriptor_ascii);
    return 0;
}

static bool is_known_ml_accelerator(uint16_t vendor_id, uint16_t product_id) {
    if ((vendor_id == CORAL_VENDOR_1 || vendor_id == CORAL_VENDOR_2) && product_id == CORAL_PRODUCT) {
        return true;
    }
    if (vendor_id == NCS2_VENDOR && product_id == NCS2_PRODUCT) {
        return true;
    }
    return false;
}

static const char* identify_device(uint16_t vendor_id, uint16_t product_id) {
    if ((vendor_id == CORAL_VENDOR_1 || vendor_id == CORAL_VENDOR_2) && product_id == CORAL_PRODUCT) {
        return "Google Coral USB Accelerator";
    }
    if (vendor_id == NCS2_VENDOR && product_id == NCS2_PRODUCT) {
        return "Intel Neural Compute Stick 2";
    }
    return "Unknown ML Accelerator";
}

bool cml_usb_available(void) {
    void* h = open_libusb();
    if (h) {
        dlclose(h);
        return true;
    }
    return false;
}

int cml_usb_enumerate(CMLUSBDevice** devices, int* num_devices) {
    if (!devices || !num_devices) return -1;

    *devices = NULL;
    *num_devices = 0;

    void* lib = open_libusb();
    if (!lib) {
        LOG_ERROR("USB enumerate: libusb not available");
        return -1;
    }

    if (!s_usb_lib) {
        if (load_libusb_symbols(lib) != 0) {
            dlclose(lib);
            return -1;
        }
        s_usb_lib = lib;
    } else {
        dlclose(lib);
    }

    void* ctx = NULL;
    int rc = fn_libusb_init(&ctx);
    if (rc < 0) {
        LOG_ERROR("USB enumerate: libusb_init failed (rc=%d)", rc);
        return -1;
    }

    void** dev_list = NULL;
    ssize_t count = fn_libusb_get_device_list(ctx, &dev_list);
    if (count < 0) {
        LOG_ERROR("USB enumerate: libusb_get_device_list failed (rc=%zd)", count);
        fn_libusb_exit(ctx);
        return -1;
    }

    /* First pass: count matching devices */
    int matched = 0;
    for (ssize_t i = 0; i < count; i++) {
        usb_device_descriptor_t desc;
        memset(&desc, 0, sizeof(desc));
        if (fn_libusb_get_device_descriptor(dev_list[i], &desc) != 0) continue;
        if (is_known_ml_accelerator(desc.idVendor, desc.idProduct)) {
            matched++;
        }
    }

    if (matched == 0) {
        LOG_DEBUG("USB enumerate: no ML accelerators found among %zd USB devices", count);
        fn_libusb_free_device_list(dev_list, 1);
        fn_libusb_exit(ctx);
        return 0;
    }

    CMLUSBDevice* devs = (CMLUSBDevice*)calloc((size_t)matched, sizeof(CMLUSBDevice));
    if (!devs) {
        LOG_ERROR("USB enumerate: allocation failed");
        fn_libusb_free_device_list(dev_list, 1);
        fn_libusb_exit(ctx);
        return -1;
    }

    /* Second pass: populate device structs */
    int idx = 0;
    for (ssize_t i = 0; i < count && idx < matched; i++) {
        usb_device_descriptor_t desc;
        memset(&desc, 0, sizeof(desc));
        if (fn_libusb_get_device_descriptor(dev_list[i], &desc) != 0) continue;
        if (!is_known_ml_accelerator(desc.idVendor, desc.idProduct)) continue;

        CMLUSBDevice* d = &devs[idx];
        d->vendor_id = desc.idVendor;
        d->product_id = desc.idProduct;
        d->bus = fn_libusb_get_bus_number(dev_list[i]);
        d->port = fn_libusb_get_port_number(dev_list[i]);
        d->is_open = false;
        d->handle = NULL;
        d->interface_num = DEFAULT_INTERFACE;
        d->endpoint_in = DEFAULT_ENDPOINT_IN;
        d->endpoint_out = DEFAULT_ENDPOINT_OUT;

        /* Try to read product name and serial via a temporary handle */
        const char* known_name = identify_device(desc.idVendor, desc.idProduct);
        strncpy(d->product_name, known_name, sizeof(d->product_name) - 1);

        void* tmp_handle = NULL;
        if (fn_libusb_open(dev_list[i], &tmp_handle) == 0 && tmp_handle) {
            if (desc.iProduct) {
                uint8_t buf[128];
                int len = fn_libusb_get_string_descriptor_ascii(tmp_handle, desc.iProduct, buf, (int)sizeof(buf));
                if (len > 0) {
                    memset(d->product_name, 0, sizeof(d->product_name));
                    memcpy(d->product_name, buf, (size_t)(len < 127 ? len : 127));
                }
            }
            if (desc.iSerialNumber) {
                uint8_t buf[64];
                int len = fn_libusb_get_string_descriptor_ascii(tmp_handle, desc.iSerialNumber, buf, (int)sizeof(buf));
                if (len > 0) {
                    memcpy(d->serial, buf, (size_t)(len < 63 ? len : 63));
                }
            }
            fn_libusb_close(tmp_handle);
        }

        LOG_INFO("USB enumerate: found %s (bus=%d, port=%d, vid=0x%04x, pid=0x%04x)",
                 d->product_name, d->bus, d->port, d->vendor_id, d->product_id);
        idx++;
    }

    fn_libusb_free_device_list(dev_list, 1);
    fn_libusb_exit(ctx);

    *devices = devs;
    *num_devices = idx;
    return 0;
}

int cml_usb_open(CMLUSBDevice* device) {
    if (!device) return -1;
    if (device->is_open) {
        LOG_WARNING("USB device already open");
        return 0;
    }

    if (!s_usb_lib) {
        void* lib = open_libusb();
        if (!lib) {
            LOG_ERROR("USB open: libusb not available");
            return -1;
        }
        if (load_libusb_symbols(lib) != 0) {
            dlclose(lib);
            return -1;
        }
        s_usb_lib = lib;
    }

    /*
     * To open a specific device by vendor/product, we need to enumerate again
     * and match on vid/pid plus bus/port to find the right libusb device object.
     */
    void* ctx = NULL;
    int rc = fn_libusb_init(&ctx);
    if (rc < 0) {
        LOG_ERROR("USB open: libusb_init failed (rc=%d)", rc);
        return -1;
    }

    void** dev_list = NULL;
    ssize_t count = fn_libusb_get_device_list(ctx, &dev_list);
    if (count < 0) {
        LOG_ERROR("USB open: libusb_get_device_list failed");
        fn_libusb_exit(ctx);
        return -1;
    }

    void* found_dev = NULL;
    for (ssize_t i = 0; i < count; i++) {
        usb_device_descriptor_t desc;
        memset(&desc, 0, sizeof(desc));
        if (fn_libusb_get_device_descriptor(dev_list[i], &desc) != 0) continue;

        if (desc.idVendor == device->vendor_id && desc.idProduct == device->product_id) {
            int bus = fn_libusb_get_bus_number(dev_list[i]);
            int port = fn_libusb_get_port_number(dev_list[i]);
            if (bus == device->bus && port == device->port) {
                found_dev = dev_list[i];
                break;
            }
        }
    }

    if (!found_dev) {
        LOG_ERROR("USB open: device not found (vid=0x%04x, pid=0x%04x, bus=%d, port=%d)",
                  device->vendor_id, device->product_id, device->bus, device->port);
        fn_libusb_free_device_list(dev_list, 1);
        fn_libusb_exit(ctx);
        return -1;
    }

    void* handle = NULL;
    rc = fn_libusb_open(found_dev, &handle);
    if (rc != 0 || !handle) {
        LOG_ERROR("USB open: libusb_open failed (rc=%d)", rc);
        fn_libusb_free_device_list(dev_list, 1);
        fn_libusb_exit(ctx);
        return -1;
    }

    rc = fn_libusb_claim_interface(handle, device->interface_num);
    if (rc != 0) {
        LOG_ERROR("USB open: libusb_claim_interface failed (rc=%d, iface=%d)", rc, device->interface_num);
        fn_libusb_close(handle);
        fn_libusb_free_device_list(dev_list, 1);
        fn_libusb_exit(ctx);
        return -1;
    }

    fn_libusb_free_device_list(dev_list, 1);
    /* Note: ctx is kept alive while device is open. In a full implementation,
     * we would store ctx in the device struct for proper cleanup. */

    device->handle = handle;
    device->is_open = true;

    LOG_INFO("USB device opened: %s (bus=%d, port=%d)", device->product_name, device->bus, device->port);
    return 0;
}

void cml_usb_close(CMLUSBDevice* device) {
    if (!device || !device->is_open) return;

    if (device->handle && fn_libusb_release_interface && fn_libusb_close) {
        fn_libusb_release_interface(device->handle, device->interface_num);
        fn_libusb_close(device->handle);
    }

    device->handle = NULL;
    device->is_open = false;

    LOG_DEBUG("USB device closed: %s", device->product_name);
}

int cml_usb_send(CMLUSBDevice* device, const void* data, size_t size) {
    if (!device || !device->is_open || !device->handle) {
        LOG_ERROR("USB send: device not open");
        return -1;
    }
    if (!data || size == 0) {
        LOG_ERROR("USB send: invalid data or size");
        return -1;
    }
    if (!fn_libusb_bulk_transfer) {
        LOG_ERROR("USB send: libusb not loaded");
        return -1;
    }

    int transferred = 0;
    int rc = fn_libusb_bulk_transfer(device->handle, device->endpoint_out,
                                     (uint8_t*)data, (int)size,
                                     &transferred, 5000);
    if (rc != 0) {
        LOG_ERROR("USB send: bulk transfer failed (rc=%d, sent=%d/%zu)",
                  rc, transferred, size);
        return -1;
    }

    LOG_DEBUG("USB send: %d/%zu bytes sent", transferred, size);
    return transferred;
}

int cml_usb_recv(CMLUSBDevice* device, void* buffer, size_t size, int timeout_ms) {
    if (!device || !device->is_open || !device->handle) {
        LOG_ERROR("USB recv: device not open");
        return -1;
    }
    if (!buffer || size == 0) {
        LOG_ERROR("USB recv: invalid buffer or size");
        return -1;
    }
    if (!fn_libusb_bulk_transfer) {
        LOG_ERROR("USB recv: libusb not loaded");
        return -1;
    }

    int transferred = 0;
    unsigned int timeout = (timeout_ms > 0) ? (unsigned int)timeout_ms : 5000;
    int rc = fn_libusb_bulk_transfer(device->handle, device->endpoint_in,
                                     (uint8_t*)buffer, (int)size,
                                     &transferred, timeout);
    if (rc != 0) {
        LOG_ERROR("USB recv: bulk transfer failed (rc=%d, received=%d/%zu, timeout=%ums)",
                  rc, transferred, size, timeout);
        return -1;
    }

    LOG_DEBUG("USB recv: %d/%zu bytes received", transferred, size);
    return transferred;
}

void cml_usb_free_devices(CMLUSBDevice* devices, int num_devices) {
    if (!devices) return;

    for (int i = 0; i < num_devices; i++) {
        if (devices[i].is_open) {
            cml_usb_close(&devices[i]);
        }
    }

    free(devices);
}
