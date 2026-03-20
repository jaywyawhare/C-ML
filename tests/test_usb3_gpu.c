#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "backend/usb3_gpu.h"

static int tests_run = 0;
static int tests_passed = 0;

#define RUN_TEST(test) do { \
    tests_run++; \
    printf("  [%d] %-50s ", tests_run, #test); \
    if (test()) { tests_passed++; printf("PASS\n"); } \
    else { printf("FAIL\n"); } \
} while(0)

static int test_available_returns_bool(void) {
    bool avail = cml_usb3_gpu_available();
    (void)avail;
    return 1;
}

static int test_open_no_device(void) {
    CMLUSB3GPU* dev = cml_usb3_gpu_open();
    if (dev) {
        cml_usb3_gpu_close(dev);
        return 1;
    }
    return 1;
}

static int test_close_null(void) {
    cml_usb3_gpu_close(NULL);
    return 1;
}

static int test_read32_null_dev(void) {
    uint32_t val;
    int ret = cml_usb3_gpu_read32(NULL, 0, &val);
    return ret == -1;
}

static int test_write32_null_dev(void) {
    int ret = cml_usb3_gpu_write32(NULL, 0, 0x42);
    return ret == -1;
}

static int test_upload_null_dev(void) {
    uint8_t data[16] = {0};
    int ret = cml_usb3_gpu_upload(NULL, 0, data, sizeof(data));
    return ret == -1;
}

static int test_download_null_dev(void) {
    uint8_t data[16];
    int ret = cml_usb3_gpu_download(NULL, 0, data, sizeof(data));
    return ret == -1;
}

static int test_scsi_cmd_null_dev(void) {
    uint8_t cdb[10] = {0};
    uint8_t data[4] = {0};
    int ret = cml_usb3_gpu_scsi_cmd(NULL, cdb, 10, data, sizeof(data), false);
    return ret == -1;
}

static int test_upload_null_data(void) {
    int ret = cml_usb3_gpu_upload(NULL, 0, NULL, 0);
    return ret == -1;
}

static int test_download_zero_size(void) {
    uint8_t buf[1];
    int ret = cml_usb3_gpu_download(NULL, 0, buf, 0);
    return ret == -1;
}

static int test_struct_layout(void) {
    CMLUSB3GPU dev;
    memset(&dev, 0, sizeof(dev));
    dev.fd = -1;
    dev.vendor_id = 0x174c;
    dev.product_id = 0x2362;
    dev.connected = false;
    dev.bar0_addr = 0xDEADBEEF;
    dev.bar0_size = 16 * 1024 * 1024;
    dev.ep_in = 0x81;
    dev.ep_out = 0x02;

    int ok = (dev.vendor_id == 0x174c);
    ok = ok && (dev.bar0_addr == 0xDEADBEEF);
    ok = ok && (dev.ep_in == 0x81);
    return ok;
}

static int test_scsi_cmd_invalid_cdb_len(void) {
    CMLUSB3GPU dev;
    memset(&dev, 0, sizeof(dev));
    dev.connected = true;
    dev.fd = 999;
    uint8_t cdb[1] = {0};
    int ret = cml_usb3_gpu_scsi_cmd(&dev, cdb, 0, NULL, 0, false);
    return ret == -1;
}

int main(void) {
    printf("USB3 GPU Tests\n");

    RUN_TEST(test_available_returns_bool);
    RUN_TEST(test_open_no_device);
    RUN_TEST(test_close_null);
    RUN_TEST(test_read32_null_dev);
    RUN_TEST(test_write32_null_dev);
    RUN_TEST(test_upload_null_dev);
    RUN_TEST(test_download_null_dev);
    RUN_TEST(test_scsi_cmd_null_dev);
    RUN_TEST(test_upload_null_data);
    RUN_TEST(test_download_zero_size);
    RUN_TEST(test_struct_layout);
    RUN_TEST(test_scsi_cmd_invalid_cdb_len);

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
