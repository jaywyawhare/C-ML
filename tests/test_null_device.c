#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "backend/null_device.h"

static int tests_run = 0;
static int tests_passed = 0;

#define RUN_TEST(test) do { \
    tests_run++; \
    printf("  [%d] %-50s ", tests_run, #test); \
    if (test()) { tests_passed++; printf("PASS\n"); } \
    else { printf("FAIL\n"); } \
} while(0)

static int test_create_destroy(void) {
    CMLNullDevice* dev = cml_null_device_create();
    if (!dev) return 0;
    int ok = dev->initialized;
    cml_null_device_free(dev);
    return ok;
}

static int test_create_with_spec(void) {
    CMLNullDevice* dev = cml_null_device_create_with_spec(
        8ULL * 1024 * 1024 * 1024, 500.0, 10.0);
    if (!dev) return 0;
    int ok = (dev->simulated_memory == 8ULL * 1024 * 1024 * 1024);
    ok = ok && (dev->simulated_bandwidth_gbps == 500.0);
    ok = ok && (dev->simulated_tflops == 10.0);
    cml_null_device_free(dev);
    return ok;
}

static int test_alloc(void) {
    CMLNullDevice* dev = cml_null_device_create();
    void* ptr = cml_null_device_alloc(dev, 1024);
    int ok = (ptr != NULL);
    CMLNullDeviceStats stats = cml_null_device_get_stats(dev);
    ok = ok && (stats.num_allocs == 1);
    ok = ok && (stats.total_bytes_allocated == 1024);
    cml_null_device_free(dev);
    return ok;
}

static int test_alloc_memory_limit(void) {
    CMLNullDevice* dev = cml_null_device_create_with_spec(100, 1.0, 1.0);
    void* ptr1 = cml_null_device_alloc(dev, 50);
    void* ptr2 = cml_null_device_alloc(dev, 60); /* Should fail: 50+60 > 100 */
    int ok = (ptr1 != NULL) && (ptr2 == NULL);
    cml_null_device_free(dev);
    return ok;
}

static int test_free_mem(void) {
    CMLNullDevice* dev = cml_null_device_create();
    void* ptr = cml_null_device_alloc(dev, 1024);
    cml_null_device_free_mem(dev, ptr, 1024);
    CMLNullDeviceStats stats = cml_null_device_get_stats(dev);
    int ok = (stats.num_frees == 1);
    ok = ok && (dev->current_allocated == 0);
    cml_null_device_free(dev);
    return ok;
}

static int test_copy(void) {
    CMLNullDevice* dev = cml_null_device_create();
    cml_null_device_copy(dev, NULL, NULL, 4096);
    CMLNullDeviceStats stats = cml_null_device_get_stats(dev);
    int ok = (stats.num_copies == 1);
    ok = ok && (stats.total_bytes_copied == 4096);
    cml_null_device_free(dev);
    return ok;
}

static int test_launch_kernel(void) {
    CMLNullDevice* dev = cml_null_device_create();
    size_t grid[3] = {256, 1, 1};
    size_t block[3] = {256, 1, 1};
    int ret = cml_null_device_launch_kernel(dev, "test_kernel", grid, block);
    CMLNullDeviceStats stats = cml_null_device_get_stats(dev);
    int ok = (ret == 0) && (stats.num_kernel_launches == 1);
    cml_null_device_free(dev);
    return ok;
}

static int test_reset_stats(void) {
    CMLNullDevice* dev = cml_null_device_create();
    cml_null_device_alloc(dev, 1024);
    cml_null_device_reset_stats(dev);
    CMLNullDeviceStats stats = cml_null_device_get_stats(dev);
    int ok = (stats.num_allocs == 0) && (stats.total_bytes_allocated == 0);
    cml_null_device_free(dev);
    return ok;
}

static int test_estimate_time_compute_bound(void) {
    CMLNullDevice* dev = cml_null_device_create_with_spec(
        16ULL * 1024 * 1024 * 1024, 900.0, 19.5);
    /* High flops, low memory = compute bound */
    double t = cml_null_device_estimate_time_ms(dev, (size_t)1e12, 1000);
    int ok = (t > 0);
    cml_null_device_free(dev);
    return ok;
}

static int test_estimate_time_memory_bound(void) {
    CMLNullDevice* dev = cml_null_device_create_with_spec(
        16ULL * 1024 * 1024 * 1024, 900.0, 19.5);
    /* Low flops, high memory = memory bound */
    double t = cml_null_device_estimate_time_ms(dev, 100, (size_t)1e9);
    int ok = (t > 0);
    cml_null_device_free(dev);
    return ok;
}

static int test_print_no_crash(void) {
    CMLNullDevice* dev = cml_null_device_create();
    cml_null_device_alloc(dev, 1024);
    size_t grid[3] = {256, 1, 1};
    size_t block[3] = {256, 1, 1};
    cml_null_device_launch_kernel(dev, "test", grid, block);
    cml_null_device_print(dev);
    cml_null_device_print(NULL);
    cml_null_device_free(dev);
    return 1;
}

static int test_free_null(void) {
    cml_null_device_free(NULL);
    return 1;
}

int main(void) {
    printf("NULL Device Tests\n");

    RUN_TEST(test_create_destroy);
    RUN_TEST(test_create_with_spec);
    RUN_TEST(test_alloc);
    RUN_TEST(test_alloc_memory_limit);
    RUN_TEST(test_free_mem);
    RUN_TEST(test_copy);
    RUN_TEST(test_launch_kernel);
    RUN_TEST(test_reset_stats);
    RUN_TEST(test_estimate_time_compute_bound);
    RUN_TEST(test_estimate_time_memory_bound);
    RUN_TEST(test_print_no_crash);
    RUN_TEST(test_free_null);

    printf("\nResults: %d/%d passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
