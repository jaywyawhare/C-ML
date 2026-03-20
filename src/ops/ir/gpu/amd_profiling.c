#include "ops/ir/gpu/amd_profiling.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <inttypes.h>

#define PROFILE_INITIAL_CAPACITY 64
#define SQTT_BUFFER_SIZE         (4 * 1024 * 1024)
#define KFD_IOCTL_BASE           'K'

/* KFD performance counter ioctl (simplified) */
#define KFD_IOC_GET_PERF_COUNTER _IOWR(KFD_IOCTL_BASE, 0x20, struct kfd_perf_counter_request)

struct kfd_perf_counter_request {
    uint32_t gpu_id;
    uint32_t counter_id;
    uint64_t value;
};

/* SQTT (Shader Queue Thread Trace) ioctl structures */
#define KFD_IOC_SQTT_START _IOW(KFD_IOCTL_BASE, 0x30, struct kfd_sqtt_request)
#define KFD_IOC_SQTT_STOP  _IOW(KFD_IOCTL_BASE, 0x31, struct kfd_sqtt_request)
#define KFD_IOC_SQTT_READ  _IOWR(KFD_IOCTL_BASE, 0x32, struct kfd_sqtt_request)

struct kfd_sqtt_request {
    uint32_t gpu_id;
    uint64_t buffer_addr;
    uint64_t buffer_size;
    uint32_t num_dispatches;
    uint32_t flags;
};

static uint64_t get_timestamp_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static int profile_ensure_capacity(CMLAMDProfile* prof) {
    if (prof->num_entries < prof->capacity) return 0;

    int new_cap = prof->capacity * 2;
    if (new_cap < PROFILE_INITIAL_CAPACITY) new_cap = PROFILE_INITIAL_CAPACITY;

    uint64_t* ts = realloc(prof->timestamps, (size_t)new_cap * sizeof(uint64_t));
    uint32_t* wc = realloc(prof->wave_counts, (size_t)new_cap * sizeof(uint32_t));
    uint64_t* bc = realloc(prof->busy_cycles, (size_t)new_cap * sizeof(uint64_t));
    uint64_t* mr = realloc(prof->mem_reads, (size_t)new_cap * sizeof(uint64_t));
    uint64_t* mw = realloc(prof->mem_writes, (size_t)new_cap * sizeof(uint64_t));

    if (!ts || !wc || !bc || !mr || !mw) {
        free(ts); free(wc); free(bc); free(mr); free(mw);
        return -1;
    }

    prof->timestamps = ts;
    prof->wave_counts = wc;
    prof->busy_cycles = bc;
    prof->mem_reads = mr;
    prof->mem_writes = mw;
    prof->capacity = new_cap;
    return 0;
}

CMLAMDProfile* cml_amd_profile_create(void) {
    CMLAMDProfile* prof = calloc(1, sizeof(CMLAMDProfile));
    if (!prof) return NULL;

    prof->capacity = PROFILE_INITIAL_CAPACITY;
    prof->timestamps  = calloc((size_t)prof->capacity, sizeof(uint64_t));
    prof->wave_counts = calloc((size_t)prof->capacity, sizeof(uint32_t));
    prof->busy_cycles = calloc((size_t)prof->capacity, sizeof(uint64_t));
    prof->mem_reads   = calloc((size_t)prof->capacity, sizeof(uint64_t));
    prof->mem_writes  = calloc((size_t)prof->capacity, sizeof(uint64_t));

    if (!prof->timestamps || !prof->wave_counts || !prof->busy_cycles ||
        !prof->mem_reads || !prof->mem_writes) {
        cml_amd_profile_free(prof);
        return NULL;
    }

    return prof;
}

int cml_amd_profile_start(CMLAMDProfile* prof, CMLAMDriver* drv) {
    if (!prof || !drv || !drv->initialized) return -1;

    prof->num_entries = 0;

    if (profile_ensure_capacity(prof) != 0) return -1;

    prof->timestamps[0] = get_timestamp_ns();
    prof->num_entries = 1;

    /* Read initial PMC values for baseline */
    uint64_t busy = 0, mem_r = 0, mem_w = 0;
    cml_amd_pmc_read(drv, 0x04, &busy);
    cml_amd_pmc_read(drv, 0x0A, &mem_r);
    cml_amd_pmc_read(drv, 0x0B, &mem_w);

    prof->busy_cycles[0] = busy;
    prof->mem_reads[0] = mem_r;
    prof->mem_writes[0] = mem_w;
    prof->wave_counts[0] = 0;

    LOG_INFO("AMD profiling started for GPU %u", drv->gpu_id);
    return 0;
}

int cml_amd_profile_stop(CMLAMDProfile* prof, CMLAMDriver* drv) {
    if (!prof || !drv || !drv->initialized) return -1;

    if (profile_ensure_capacity(prof) != 0) return -1;

    int idx = prof->num_entries;
    prof->timestamps[idx] = get_timestamp_ns();

    uint64_t busy = 0, mem_r = 0, mem_w = 0;
    cml_amd_pmc_read(drv, 0x04, &busy);
    cml_amd_pmc_read(drv, 0x0A, &mem_r);
    cml_amd_pmc_read(drv, 0x0B, &mem_w);

    prof->busy_cycles[idx] = busy;
    prof->mem_reads[idx] = mem_r;
    prof->mem_writes[idx] = mem_w;
    prof->wave_counts[idx] = 0;

    prof->num_entries++;

    LOG_INFO("AMD profiling stopped: %d entries captured", prof->num_entries);
    return 0;
}

void cml_amd_profile_free(CMLAMDProfile* prof) {
    if (!prof) return;
    free(prof->timestamps);
    free(prof->wave_counts);
    free(prof->busy_cycles);
    free(prof->mem_reads);
    free(prof->mem_writes);
    free(prof);
}

int cml_amd_pmc_read(CMLAMDriver* drv, uint32_t counter_id, uint64_t* value) {
    if (!drv || !drv->initialized || !value) return -1;

    *value = 0;

    /* Try debugfs first: /sys/kernel/debug/dri/0/amdgpu_pm_info */
    FILE* fp = fopen("/sys/kernel/debug/dri/0/amdgpu_pm_info", "r");
    if (fp) {
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            uint64_t val;
            if (counter_id == 0x04 && strstr(line, "GPU Load")) {
                if (sscanf(line, "%" SCNu64, &val) == 1) *value = val;
                break;
            }
            if (counter_id == 0x0A && strstr(line, "MemRead")) {
                if (sscanf(line, "%" SCNu64, &val) == 1) *value = val;
                break;
            }
            if (counter_id == 0x0B && strstr(line, "MemWrite")) {
                if (sscanf(line, "%" SCNu64, &val) == 1) *value = val;
                break;
            }
        }
        fclose(fp);
        return 0;
    }

    /* Fallback: KFD perf counter ioctl */
    if (drv->fd_kfd >= 0) {
        struct kfd_perf_counter_request req = {
            .gpu_id = drv->gpu_id,
            .counter_id = counter_id,
            .value = 0
        };
        if (ioctl(drv->fd_kfd, KFD_IOC_GET_PERF_COUNTER, &req) == 0) {
            *value = req.value;
            return 0;
        }
    }

    return -1;
}

CMLAMDSQTTTrace* cml_amd_sqtt_capture(CMLAMDriver* drv, int num_dispatches) {
    if (!drv || !drv->initialized || num_dispatches <= 0) return NULL;

    CMLAMDSQTTTrace* trace = calloc(1, sizeof(CMLAMDSQTTTrace));
    if (!trace) return NULL;

    trace->data = calloc(1, SQTT_BUFFER_SIZE);
    if (!trace->data) {
        free(trace);
        return NULL;
    }
    trace->size = SQTT_BUFFER_SIZE;
    trace->num_waves = 0;

    if (drv->fd_kfd < 0) {
        LOG_WARNING("No KFD fd, SQTT capture will be empty");
        return trace;
    }

    struct kfd_sqtt_request start_req = {
        .gpu_id = drv->gpu_id,
        .buffer_addr = (uint64_t)(uintptr_t)trace->data,
        .buffer_size = (uint64_t)trace->size,
        .num_dispatches = (uint32_t)num_dispatches,
        .flags = 0
    };

    if (ioctl(drv->fd_kfd, KFD_IOC_SQTT_START, &start_req) != 0) {
        LOG_WARNING("SQTT start ioctl failed, trace will be empty");
        return trace;
    }

    /* Wait for dispatches to complete - in practice the caller runs
     * kernel launches between start and stop. Here we just issue stop. */

    struct kfd_sqtt_request stop_req = {
        .gpu_id = drv->gpu_id,
        .buffer_addr = 0,
        .buffer_size = 0,
        .num_dispatches = 0,
        .flags = 0
    };

    ioctl(drv->fd_kfd, KFD_IOC_SQTT_STOP, &stop_req);

    struct kfd_sqtt_request read_req = {
        .gpu_id = drv->gpu_id,
        .buffer_addr = (uint64_t)(uintptr_t)trace->data,
        .buffer_size = (uint64_t)trace->size,
        .num_dispatches = 0,
        .flags = 0
    };

    if (ioctl(drv->fd_kfd, KFD_IOC_SQTT_READ, &read_req) == 0) {
        trace->size = (size_t)read_req.buffer_size;
        trace->num_waves = (int)read_req.num_dispatches;
    }

    LOG_INFO("SQTT capture: %zu bytes, %d waves", trace->size, trace->num_waves);
    return trace;
}

void cml_amd_sqtt_free(CMLAMDSQTTTrace* trace) {
    if (!trace) return;
    free(trace->data);
    free(trace);
}

void cml_amd_profile_print(const CMLAMDProfile* prof) {
    if (!prof || prof->num_entries < 2) {
        printf("AMD Profile: no data\n");
        return;
    }

    printf("AMD GPU Profile (%d entries)\n", prof->num_entries);

    uint64_t total_ns = prof->timestamps[prof->num_entries - 1] - prof->timestamps[0];
    double total_ms = (double)total_ns / 1e6;
    printf("Total time: %.3f ms\n", total_ms);

    uint64_t total_busy = prof->busy_cycles[prof->num_entries - 1] - prof->busy_cycles[0];
    printf("GPU busy cycles: %" PRIu64 "\n", total_busy);

    uint64_t total_reads = prof->mem_reads[prof->num_entries - 1] - prof->mem_reads[0];
    uint64_t total_writes = prof->mem_writes[prof->num_entries - 1] - prof->mem_writes[0];
    printf("Memory reads:  %" PRIu64 " bytes\n", total_reads);
    printf("Memory writes: %" PRIu64 " bytes\n", total_writes);

    if (total_ns > 0) {
        double bw_read = (double)total_reads / ((double)total_ns / 1e9) / (1024.0 * 1024.0 * 1024.0);
        double bw_write = (double)total_writes / ((double)total_ns / 1e9) / (1024.0 * 1024.0 * 1024.0);
        printf("Read BW:  %.2f GB/s\n", bw_read);
        printf("Write BW: %.2f GB/s\n", bw_write);
    }

    printf("\nPer-entry breakdown:\n");
    for (int i = 1; i < prof->num_entries; i++) {
        uint64_t dt = prof->timestamps[i] - prof->timestamps[i - 1];
        double ms = (double)dt / 1e6;
        printf("  [%d] %.3f ms, waves=%u, busy=%" PRIu64 ", rd=%" PRIu64 ", wr=%" PRIu64 "\n",
               i - 1, ms,
               prof->wave_counts[i] - prof->wave_counts[i - 1],
               prof->busy_cycles[i] - prof->busy_cycles[i - 1],
               prof->mem_reads[i] - prof->mem_reads[i - 1],
               prof->mem_writes[i] - prof->mem_writes[i - 1]);
    }
}
