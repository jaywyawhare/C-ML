#include "backend/null_device.h"
#include "ops/ir/ir.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _POSIX_C_SOURCE
#include <time.h>
static double get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}
#else
static double get_time_us(void) { return 0.0; }
#endif

CMLNullDevice* cml_null_device_create(void) {
    return cml_null_device_create_with_spec(
        (size_t)16 * 1024 * 1024 * 1024ULL,  /* 16 GB */
        900.0,   /* ~900 GB/s (A100) */
        19.5     /* ~19.5 TFLOPS (A100 FP32) */
    );
}

CMLNullDevice* cml_null_device_create_with_spec(size_t memory_bytes,
                                                  double bandwidth_gbps,
                                                  double tflops) {
    CMLNullDevice* dev = (CMLNullDevice*)calloc(1, sizeof(CMLNullDevice));
    if (!dev) return NULL;

    dev->initialized = true;
    dev->simulated_memory = memory_bytes;
    dev->simulated_bandwidth_gbps = bandwidth_gbps;
    dev->simulated_tflops = tflops;
    memset(&dev->stats, 0, sizeof(dev->stats));

    return dev;
}

void cml_null_device_free(CMLNullDevice* dev) {
    free(dev);
}

void* cml_null_device_alloc(CMLNullDevice* dev, size_t size) {
    if (!dev || !dev->initialized) return NULL;

    /* Check simulated memory limit */
    if (dev->current_allocated + size > dev->simulated_memory) return NULL;

    dev->stats.num_allocs++;
    dev->stats.total_bytes_allocated += size;
    dev->current_allocated += size;

    /* Return a dummy non-NULL pointer. We don't actually allocate.
     * Use a sentinel value that's recognizable. */
    return (void*)(uintptr_t)(0xDEAD0000ULL + dev->stats.num_allocs);
}

void cml_null_device_free_mem(CMLNullDevice* dev, void* ptr, size_t size) {
    if (!dev || !ptr) return;
    dev->stats.num_frees++;
    if (dev->current_allocated >= size)
        dev->current_allocated -= size;
}

void cml_null_device_copy(CMLNullDevice* dev, void* dst, const void* src, size_t size) {
    (void)dst; (void)src;
    if (!dev) return;
    dev->stats.num_copies++;
    dev->stats.total_bytes_copied += size;
}

int cml_null_device_launch_kernel(CMLNullDevice* dev, const char* kernel_name,
                                   size_t grid[3], size_t block[3]) {
    (void)kernel_name; (void)grid; (void)block;
    if (!dev || !dev->initialized) return -1;

    double start = get_time_us();
    /* No actual work - just measure dispatch overhead */
    double end = get_time_us();

    dev->stats.num_kernel_launches++;
    dev->stats.total_dispatch_time_us += (end - start);
    return 0;
}

int cml_null_device_execute(CMLNullDevice* dev, CMLGraph_t ir) {
    if (!dev || !dev->initialized || !ir) return -1;

    double start = get_time_us();

    /* Walk IR graph, count ops without executing */
    /* We access the graph node count through its public API */
    dev->stats.total_ops_dispatched++;

    double end = get_time_us();
    dev->stats.total_dispatch_time_us += (end - start);

    return 0;
}

CMLNullDeviceStats cml_null_device_get_stats(const CMLNullDevice* dev) {
    CMLNullDeviceStats zero = {0};
    return dev ? dev->stats : zero;
}

void cml_null_device_reset_stats(CMLNullDevice* dev) {
    if (!dev) return;
    memset(&dev->stats, 0, sizeof(dev->stats));
    dev->current_allocated = 0;
}

double cml_null_device_estimate_time_ms(const CMLNullDevice* dev,
                                         size_t flops, size_t memory_bytes) {
    if (!dev || !dev->initialized) return 0.0;

    /* Roofline model: time = max(compute_time, memory_time) */
    double compute_time_ms = 0.0;
    double memory_time_ms = 0.0;

    if (dev->simulated_tflops > 0)
        compute_time_ms = (double)flops / (dev->simulated_tflops * 1e12) * 1e3;

    if (dev->simulated_bandwidth_gbps > 0)
        memory_time_ms = (double)memory_bytes / (dev->simulated_bandwidth_gbps * 1e9) * 1e3;

    return compute_time_ms > memory_time_ms ? compute_time_ms : memory_time_ms;
}

void cml_null_device_print(const CMLNullDevice* dev) {
    if (!dev) {
        printf("NullDevice: NULL\n");
        return;
    }

    printf("NULL Benchmark Device\n");
    printf("Simulated specs:\n");
    printf("  Memory: %.1f GB\n", (double)dev->simulated_memory / (1024.0 * 1024.0 * 1024.0));
    printf("  Bandwidth: %.1f GB/s\n", dev->simulated_bandwidth_gbps);
    printf("  Compute: %.1f TFLOPS\n", dev->simulated_tflops);
    printf("  Current allocated: %.1f MB\n", (double)dev->current_allocated / (1024.0 * 1024.0));
    printf("\nStatistics:\n");
    printf("  Allocs: %lu, Frees: %lu\n",
           (unsigned long)dev->stats.num_allocs, (unsigned long)dev->stats.num_frees);
    printf("  Copies: %lu (%.1f MB)\n",
           (unsigned long)dev->stats.num_copies,
           (double)dev->stats.total_bytes_copied / (1024.0 * 1024.0));
    printf("  Kernel launches: %lu\n", (unsigned long)dev->stats.num_kernel_launches);
    printf("  Dispatch overhead: %.2f us total\n", dev->stats.total_dispatch_time_us);
    if (dev->stats.num_kernel_launches > 0)
        printf("  Avg launch overhead: %.2f us\n",
               dev->stats.total_dispatch_time_us / (double)dev->stats.num_kernel_launches);
    printf("\n");
}
