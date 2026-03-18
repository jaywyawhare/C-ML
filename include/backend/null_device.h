/*
 * A virtual device that accepts all operations but performs no computation.
 * Used for benchmarking dispatch overhead, measuring kernel launch latency,
 * and testing scheduling/fusion without actual hardware.
 */

#ifndef CML_NULL_DEVICE_H
#define CML_NULL_DEVICE_H

#include "tensor/tensor.h"
#include "ops/ir/ir.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CMLNullDeviceStats {
    uint64_t num_allocs;
    uint64_t num_frees;
    uint64_t num_copies;
    uint64_t num_kernel_launches;
    uint64_t total_bytes_allocated;
    uint64_t total_bytes_copied;
    uint64_t total_ops_dispatched;
    double total_dispatch_time_us;   /* Microseconds spent in dispatch */
} CMLNullDeviceStats;

typedef struct CMLNullDevice {
    bool initialized;
    CMLNullDeviceStats stats;
    size_t simulated_memory;        /* Total "available" memory */
    size_t current_allocated;       /* Currently allocated bytes */
    double simulated_bandwidth_gbps;/* Simulated memory bandwidth */
    double simulated_tflops;        /* Simulated compute throughput */
} CMLNullDevice;

CMLNullDevice* cml_null_device_create(void);
CMLNullDevice* cml_null_device_create_with_spec(size_t memory_bytes,
                                                  double bandwidth_gbps,
                                                  double tflops);
void cml_null_device_free(CMLNullDevice* dev);

/* Simulate memory allocation (tracks stats, allocates nothing) */
void* cml_null_device_alloc(CMLNullDevice* dev, size_t size);
void cml_null_device_free_mem(CMLNullDevice* dev, void* ptr, size_t size);
void cml_null_device_copy(CMLNullDevice* dev, void* dst, const void* src, size_t size);
int cml_null_device_launch_kernel(CMLNullDevice* dev, const char* kernel_name,
                                   size_t grid[3], size_t block[3]);

/* Measures dispatch overhead only */
int cml_null_device_execute(CMLNullDevice* dev, CMLGraph_t ir);
CMLNullDeviceStats cml_null_device_get_stats(const CMLNullDevice* dev);
void cml_null_device_reset_stats(CMLNullDevice* dev);

/* Estimate execution time based on simulated specs */
double cml_null_device_estimate_time_ms(const CMLNullDevice* dev,
                                         size_t flops, size_t memory_bytes);
void cml_null_device_print(const CMLNullDevice* dev);

#ifdef __cplusplus
}
#endif

#endif /* CML_NULL_DEVICE_H */
