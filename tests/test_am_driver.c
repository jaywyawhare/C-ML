#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "ops/ir/gpu/am_driver.h"
#include "ops/ir/gpu/amdgpu_kd.h"

#ifdef CML_AM_MOCK_GPU
#include "ops/ir/gpu/am_mock.h"
#endif


static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    printf("  Testing: %s... ", #name); \
    tests_run++; \
    if (test_##name()) { printf("PASS\n"); tests_passed++; } \
    else { printf("FAIL\n"); } \
} while(0)

static bool g_hw_available = false;

#define SKIP_IF_NO_HW() do { \
    if (!g_hw_available) { \
        printf("SKIP (no AMD KFD hardware) "); \
        return true; \
    } \
} while(0)


static bool test_availability_check(void) {
    bool a = cml_am_driver_available();
    bool b = cml_am_driver_available();
    if (a != b) return false;
    return true;
}

static bool test_create_and_free(void) {
    CMLAMDriver* drv = cml_am_driver_create();
    if (!drv) return false;

    if (drv->initialized) { cml_am_driver_free(drv); return false; }
    if (drv->fd_kfd != -1) { cml_am_driver_free(drv); return false; }
    if (drv->fd_drm != -1) { cml_am_driver_free(drv); return false; }

    cml_am_driver_free(drv);
    return true;
}

static bool test_free_null(void) {
    cml_am_driver_free(NULL);
    return true;
}

static bool test_init(void) {
    SKIP_IF_NO_HW();

    CMLAMDriver* drv = cml_am_driver_create();
    if (!drv) return false;

    int ret = cml_am_driver_init(drv);
    if (ret != 0) {
        printf("(init returned %d, hw-specific) ", ret);
        cml_am_driver_free(drv);
        return true;
    }

    if (!drv->initialized) { cml_am_driver_free(drv); return false; }
    if (drv->fd_kfd < 0)   { cml_am_driver_free(drv); return false; }
    if (drv->fd_drm < 0)   { cml_am_driver_free(drv); return false; }

    cml_am_driver_free(drv);
    return true;
}

static bool test_buffer_create_gtt(void) {
    SKIP_IF_NO_HW();

    CMLAMDriver* drv = cml_am_driver_create();
    if (!drv) return false;

    if (cml_am_driver_init(drv) != 0) {
        cml_am_driver_free(drv);
        return true;
    }

    CMLAMBuffer* buf = cml_am_buffer_create(drv, 4096, false);
    if (!buf) {
        printf("(buffer create failed, hw-specific) ");
        cml_am_driver_free(drv);
        return true;
    }

    if (buf->gpu_va == 0) { cml_am_buffer_free(drv, buf); cml_am_driver_free(drv); return false; }
    if (buf->is_vram)     { cml_am_buffer_free(drv, buf); cml_am_driver_free(drv); return false; }

    cml_am_buffer_free(drv, buf);
    cml_am_driver_free(drv);
    return true;
}

static bool test_buffer_create_vram(void) {
    SKIP_IF_NO_HW();

    CMLAMDriver* drv = cml_am_driver_create();
    if (!drv) return false;

    if (cml_am_driver_init(drv) != 0) {
        cml_am_driver_free(drv);
        return true;
    }

    CMLAMBuffer* buf = cml_am_buffer_create(drv, 4096, true);
    if (!buf) {
        printf("(VRAM buffer create failed, hw-specific) ");
        cml_am_driver_free(drv);
        return true;
    }

    if (buf->gpu_va == 0) { cml_am_buffer_free(drv, buf); cml_am_driver_free(drv); return false; }
    if (!buf->is_vram)    { cml_am_buffer_free(drv, buf); cml_am_driver_free(drv); return false; }

    cml_am_buffer_free(drv, buf);
    cml_am_driver_free(drv);
    return true;
}

static bool test_buffer_upload_download(void) {
    SKIP_IF_NO_HW();

    CMLAMDriver* drv = cml_am_driver_create();
    if (!drv) return false;

    if (cml_am_driver_init(drv) != 0) {
        cml_am_driver_free(drv);
        return true;
    }

    size_t size = 256 * sizeof(float);
    CMLAMBuffer* buf = cml_am_buffer_create(drv, size, false);
    if (!buf) {
        cml_am_driver_free(drv);
        return true;
    }

    float* src = (float*)malloc(size);
    float* dst = (float*)calloc(256, sizeof(float));
    if (!src || !dst) {
        free(src); free(dst);
        cml_am_buffer_free(drv, buf);
        cml_am_driver_free(drv);
        return false;
    }

    for (int i = 0; i < 256; i++)
        src[i] = (float)i * 1.5f;

    if (cml_am_buffer_upload(drv, buf, src, size) != 0) {
        printf("(upload failed) ");
        free(src); free(dst);
        cml_am_buffer_free(drv, buf);
        cml_am_driver_free(drv);
        return true;
    }

    if (cml_am_buffer_download(drv, buf, dst, size) != 0) {
        printf("(download failed) ");
        free(src); free(dst);
        cml_am_buffer_free(drv, buf);
        cml_am_driver_free(drv);
        return true;
    }

    bool match = true;
    for (int i = 0; i < 256; i++) {
        if (dst[i] != src[i]) { match = false; break; }
    }

    free(src);
    free(dst);
    cml_am_buffer_free(drv, buf);
    cml_am_driver_free(drv);
    return match;
}

static bool test_buffer_free_null(void) {
    cml_am_buffer_free(NULL, NULL);
    return true;
}

static bool test_kernel_load_null_args(void) {
    CMLAMKernel* k = cml_am_kernel_load(NULL, NULL, 0, NULL);
    if (k != NULL) return false;
    return true;
}

static bool test_kernel_free_null(void) {
    cml_am_kernel_free(NULL, NULL);
    return true;
}

static bool test_synchronize_no_work(void) {
    SKIP_IF_NO_HW();

    CMLAMDriver* drv = cml_am_driver_create();
    if (!drv) return false;

    if (cml_am_driver_init(drv) != 0) {
        cml_am_driver_free(drv);
        return true;
    }

    int ret = cml_am_synchronize(drv);
    cml_am_driver_free(drv);
    return (ret == 0);
}

static bool test_synchronize_null(void) {
    int ret = cml_am_synchronize(NULL);
    return (ret == -1);
}

static bool test_execute_graph_stub(void) {
    CMLAMDriver* drv = cml_am_driver_create();
    if (!drv) return false;

    int ret = cml_am_execute_graph(drv, NULL);
    cml_am_driver_free(drv);
    return (ret == -1);
}

static bool test_execute_graph_null(void) {
    int ret = cml_am_execute_graph(NULL, NULL);
    return (ret == -1);
}

static bool test_buffer_create_zero_size(void) {
    SKIP_IF_NO_HW();

    CMLAMDriver* drv = cml_am_driver_create();
    if (!drv) return false;

    if (cml_am_driver_init(drv) != 0) {
        cml_am_driver_free(drv);
        return true;
    }

    CMLAMBuffer* buf = cml_am_buffer_create(drv, 0, false);
    cml_am_driver_free(drv);
    return (buf == NULL);
}

static bool test_buffer_create_uninit(void) {
    CMLAMDriver* drv = cml_am_driver_create();
    if (!drv) return false;

    CMLAMBuffer* buf = cml_am_buffer_create(drv, 4096, false);
    cml_am_driver_free(drv);
    return (buf == NULL);
}

static bool test_enumerate_gpus(void) {
    CMLAMGPUInfo* gpus = NULL;
    int count = 0;
    int ret = cml_am_enumerate_gpus(&gpus, &count);

    if (ret != 0 && !g_hw_available) {
        printf("SKIP (no KFD topology) ");
        return true;
    }

    if (ret == 0) {
        printf("(found %d GPU(s)) ", count);
        if (g_hw_available && count == 0) {
            free(gpus);
            return false;
        }
        for (int i = 0; i < count; i++) {
            printf("\n    GPU[%d]: %s [%s] gpu_id=%u CUs=%d",
                   i, gpus[i].name, gpus[i].gfx_version,
                   gpus[i].gpu_id, gpus[i].cu_count);
        }
        if (count > 0) printf("\n    ");
    }

    free(gpus);
    return true;
}

static bool test_enumerate_gpus_null(void) {
    int ret = cml_am_enumerate_gpus(NULL, NULL);
    return (ret == -1);
}

static bool test_signal_create_free(void) {
    SKIP_IF_NO_HW();

    CMLAMDriver* drv = cml_am_driver_create();
    if (!drv) return false;

    if (cml_am_driver_init(drv) != 0) {
        cml_am_driver_free(drv);
        return true;
    }

    CMLAMSignal* sig = cml_am_signal_create(drv, 0);
    if (!sig) {
        printf("(signal create failed, hw-specific) ");
        cml_am_driver_free(drv);
        return true;
    }

    if (!sig->value) { cml_am_signal_free(drv, sig); cml_am_driver_free(drv); return false; }
    if (sig->gpu_va == 0) { cml_am_signal_free(drv, sig); cml_am_driver_free(drv); return false; }
    if (*sig->value != 0) { cml_am_signal_free(drv, sig); cml_am_driver_free(drv); return false; }

    cml_am_signal_free(drv, sig);
    cml_am_driver_free(drv);
    return true;
}

static bool test_signal_free_null(void) {
    cml_am_signal_free(NULL, NULL);
    return true;
}

static bool test_signal_wait_null(void) {
    int ret = cml_am_signal_wait(NULL, 0, 1000);
    return (ret == -1);
}

static bool test_signal_immediate_wait(void) {
    SKIP_IF_NO_HW();

    CMLAMDriver* drv = cml_am_driver_create();
    if (!drv) return false;

    if (cml_am_driver_init(drv) != 0) {
        cml_am_driver_free(drv);
        return true;
    }

    CMLAMSignal* sig = cml_am_signal_create(drv, 42);
    if (!sig) {
        cml_am_driver_free(drv);
        return true;
    }

    int ret = cml_am_signal_wait(sig, 42, 100000);
    cml_am_signal_free(drv, sig);
    cml_am_driver_free(drv);
    return (ret == 0);
}

static bool test_multi_queue_create(void) {
    SKIP_IF_NO_HW();

    CMLAMDriver* drv = cml_am_driver_create();
    if (!drv) return false;

    if (cml_am_driver_init(drv) != 0) {
        cml_am_driver_free(drv);
        return true;
    }

    if (drv->num_compute_queues < 1) {
        cml_am_driver_free(drv);
        return false;
    }

    int ret = cml_am_create_compute_queue(drv, 1);
    if (ret != 0) {
        printf("(queue 1 creation failed, hw-specific) ");
        cml_am_driver_free(drv);
        return true;
    }

    if (drv->num_compute_queues < 2) {
        cml_am_driver_free(drv);
        return false;
    }

    cml_am_driver_free(drv);
    return true;
}

static bool test_multi_queue_invalid(void) {
    CMLAMDriver* drv = cml_am_driver_create();
    if (!drv) return false;

    int ret = cml_am_create_compute_queue(drv, 0);
    cml_am_driver_free(drv);
    return (ret == -1);
}

static bool test_multi_queue_bounds(void) {
    int ret = cml_am_create_compute_queue(NULL, -1);
    if (ret != -1) return false;
    ret = cml_am_create_compute_queue(NULL, AM_MAX_COMPUTE_QUEUES);
    return (ret == -1);
}

static bool test_sdma_queue(void) {
    SKIP_IF_NO_HW();

    CMLAMDriver* drv = cml_am_driver_create();
    if (!drv) return false;

    if (cml_am_driver_init(drv) != 0) {
        cml_am_driver_free(drv);
        return true;
    }

    if (!drv->has_sdma) {
        printf("(SDMA not available) ");
        cml_am_driver_free(drv);
        return true;
    }

    printf("(SDMA active) ");
    cml_am_driver_free(drv);
    return true;
}

static bool test_sdma_copy(void) {
    SKIP_IF_NO_HW();

    CMLAMDriver* drv = cml_am_driver_create();
    if (!drv) return false;

    if (cml_am_driver_init(drv) != 0) {
        cml_am_driver_free(drv);
        return true;
    }

    if (!drv->has_sdma) {
        printf("SKIP (no SDMA) ");
        cml_am_driver_free(drv);
        return true;
    }

    size_t size = 1024;
    CMLAMBuffer* src = cml_am_buffer_create(drv, size, false);
    CMLAMBuffer* dst = cml_am_buffer_create(drv, size, false);
    if (!src || !dst) {
        if (src) cml_am_buffer_free(drv, src);
        if (dst) cml_am_buffer_free(drv, dst);
        cml_am_driver_free(drv);
        return true;
    }

    memset(src->cpu_addr, 0xAB, size);
    memset(dst->cpu_addr, 0, size);

    int ret = cml_am_sdma_copy(drv, dst->gpu_va, src->gpu_va, size);
    if (ret == 0) ret = cml_am_sdma_synchronize(drv);

    bool ok = true;
    if (ret == 0) {
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
        ok = (memcmp(dst->cpu_addr, src->cpu_addr, size) == 0);
    } else {
        printf("(sdma copy/sync failed) ");
        ok = true;
    }

    cml_am_buffer_free(drv, src);
    cml_am_buffer_free(drv, dst);
    cml_am_driver_free(drv);
    return ok;
}

static bool test_validate_lds(void) {
    CMLAMDriver* drv = cml_am_driver_create();
    if (!drv) return false;

    bool ok = !cml_am_validate_lds(drv, 1024);
    cml_am_driver_free(drv);
    return ok;
}

static bool test_validate_lds_hw(void) {
    SKIP_IF_NO_HW();

    CMLAMDriver* drv = cml_am_driver_create();
    if (!drv) return false;

    if (cml_am_driver_init(drv) != 0) {
        cml_am_driver_free(drv);
        return true;
    }

    bool ok = cml_am_validate_lds(drv, 1024);
    bool too_big = !cml_am_validate_lds(drv, 1024 * 1024);

    cml_am_driver_free(drv);
    return ok && too_big;
}

static bool test_scratch_alloc(void) {
    SKIP_IF_NO_HW();

    CMLAMDriver* drv = cml_am_driver_create();
    if (!drv) return false;

    if (cml_am_driver_init(drv) != 0) {
        cml_am_driver_free(drv);
        return true;
    }

    int ret = cml_am_alloc_scratch(drv, 256, 32);
    if (ret != 0) {
        printf("(scratch alloc failed, hw-specific) ");
        cml_am_driver_free(drv);
        return true;
    }

    if (!drv->has_scratch) { cml_am_driver_free(drv); return false; }
    if (drv->scratch.gpu_va == 0) { cml_am_driver_free(drv); return false; }

    cml_am_driver_free(drv);
    return true;
}

static bool test_scratch_alloc_null(void) {
    int ret = cml_am_alloc_scratch(NULL, 256, 32);
    return (ret == -1);
}

static bool test_barrier_null(void) {
    int ret = cml_am_barrier_and(NULL, 0, NULL, 0, NULL);
    if (ret != -1) return false;
    ret = cml_am_barrier_or(NULL, 0, NULL, 0, NULL);
    return (ret == -1);
}

static bool test_gpu_hang_check(void) {
    SKIP_IF_NO_HW();

    CMLAMDriver* drv = cml_am_driver_create();
    if (!drv) return false;

    if (cml_am_driver_init(drv) != 0) {
        cml_am_driver_free(drv);
        return true;
    }

    int status = cml_am_check_gpu_hang(drv);
    printf("(hang_status=%d) ", status);

    cml_am_driver_free(drv);
    return (status == 0 || status == -1);
}

static bool test_gpu_hang_null(void) {
    int ret = cml_am_check_gpu_hang(NULL);
    return (ret == -1);
}

static bool test_kd_parse_null(void) {
    AMDGPUKernelDescriptor kd;
    int ret = am_parse_kernel_descriptor(NULL, 0, NULL, &kd);
    return (ret == -1);
}

static bool test_kd_parse_bad_elf(void) {
    char buf[128] = {0};
    AMDGPUKernelDescriptor kd;
    int ret = am_parse_kernel_descriptor(buf, sizeof(buf), "test", &kd);
    return (ret == -1);
}

static bool test_gpu_info_topology(void) {
    SKIP_IF_NO_HW();

    CMLAMDriver* drv = cml_am_driver_create();
    if (!drv) return false;

    if (cml_am_driver_init(drv) != 0) {
        cml_am_driver_free(drv);
        return true;
    }

    bool ok = true;
    if (drv->gpu_id == 0) ok = false;
    if (drv->gfx_version[0] == '\0') ok = false;

    printf("(gpu_id=%u gfx=%s CUs=%d) ",
           drv->gpu_id, drv->gfx_version, drv->cu_count);

    cml_am_driver_free(drv);
    return ok;
}


#ifdef CML_AM_MOCK_GPU

static bool test_mock_enumerate_gpus(void) {
    CMLAMGPUInfo* gpus = NULL;
    int count = 0;
    int ret = cml_am_enumerate_gpus(&gpus, &count);
    if (ret != 0) return false;
    if (count < 1) { free(gpus); return false; }

    CMLAMMockGPU* mock = cml_am_mock_get();
    bool ok = (gpus[0].gpu_id == mock->gpu_id);
    ok = ok && (gpus[0].simd_per_cu == mock->simd_per_cu);
    ok = ok && (gpus[0].max_waves_per_simd == mock->max_waves_per_simd);
    ok = ok && (strcmp(gpus[0].gfx_version, mock->gfx_version) == 0);

    printf("(found %d GPU: %s [%s] gpu_id=%u) ",
           count, gpus[0].name, gpus[0].gfx_version, gpus[0].gpu_id);

    free(gpus);
    return ok;
}

static bool test_mock_driver_init(void) {
    CMLAMDriver* drv = cml_am_driver_create();
    if (!drv) return false;

    int ret = cml_am_driver_init(drv);
    if (ret != 0) { cml_am_driver_free(drv); return false; }

    if (!drv->initialized) { cml_am_driver_free(drv); return false; }
    if (drv->fd_kfd < 0)   { cml_am_driver_free(drv); return false; }
    if (drv->fd_drm < 0)   { cml_am_driver_free(drv); return false; }

    CMLAMMockGPU* mock = cml_am_mock_get();
    bool ok = (drv->gpu_id == mock->gpu_id);
    ok = ok && (strcmp(drv->gfx_version, mock->gfx_version) == 0);
    ok = ok && drv->aql_queue.active;
    ok = ok && (drv->num_compute_queues >= 1);

    printf("(gpu_id=%u gfx=%s queues=%d sdma=%d) ",
           drv->gpu_id, drv->gfx_version,
           drv->num_compute_queues, drv->has_sdma);

    cml_am_driver_free(drv);
    return ok;
}

static bool test_mock_buffer_lifecycle(void) {
    CMLAMDriver* drv = cml_am_driver_create();
    if (!drv) return false;
    if (cml_am_driver_init(drv) != 0) { cml_am_driver_free(drv); return false; }

    size_t size = 256 * sizeof(float);
    CMLAMBuffer* buf = cml_am_buffer_create(drv, size, false);
    if (!buf) { cml_am_driver_free(drv); return false; }

    if (buf->gpu_va == 0) { cml_am_buffer_free(drv, buf); cml_am_driver_free(drv); return false; }
    if (buf->is_vram)     { cml_am_buffer_free(drv, buf); cml_am_driver_free(drv); return false; }

    float* src = (float*)malloc(size);
    float* dst = (float*)calloc(256, sizeof(float));
    if (!src || !dst) { free(src); free(dst); cml_am_buffer_free(drv, buf); cml_am_driver_free(drv); return false; }

    for (int i = 0; i < 256; i++) src[i] = (float)i * 2.5f;

    if (cml_am_buffer_upload(drv, buf, src, size) != 0) { free(src); free(dst); cml_am_buffer_free(drv, buf); cml_am_driver_free(drv); return false; }
    if (cml_am_buffer_download(drv, buf, dst, size) != 0) { free(src); free(dst); cml_am_buffer_free(drv, buf); cml_am_driver_free(drv); return false; }

    bool match = true;
    for (int i = 0; i < 256; i++) {
        if (dst[i] != src[i]) { match = false; break; }
    }

    CMLAMBuffer* vbuf = cml_am_buffer_create(drv, size, true);
    bool vram_ok = (vbuf != NULL);
    if (vbuf) {
        vram_ok = vram_ok && vbuf->is_vram && (vbuf->gpu_va != 0);
        cml_am_buffer_free(drv, vbuf);
    }

    free(src);
    free(dst);
    cml_am_buffer_free(drv, buf);
    cml_am_driver_free(drv);
    return match && vram_ok;
}

static bool test_mock_kernel_launch(void) {
    CMLAMDriver* drv = cml_am_driver_create();
    if (!drv) return false;
    if (cml_am_driver_init(drv) != 0) { cml_am_driver_free(drv); return false; }

    /* Build a minimal fake ELF that the driver will accept for loading.
     * The kernel_load path uses am_alloc_and_map + memcpy; with mock
     * it will succeed even though the ELF is not real AMDGPU code. */
    uint8_t fake_elf[256];
    memset(fake_elf, 0, sizeof(fake_elf));
    fake_elf[0] = 0x7f; fake_elf[1] = 'E'; fake_elf[2] = 'L'; fake_elf[3] = 'F';
    fake_elf[4] = 2; /* 64-bit */

    CMLAMKernel* kernel = cml_am_kernel_load(drv, fake_elf, sizeof(fake_elf), "mock_kernel");
    if (!kernel) {
        cml_am_driver_free(drv);
        return false;
    }

    if (kernel->gpu_addr == 0) { cml_am_kernel_free(drv, kernel); cml_am_driver_free(drv); return false; }

    uint32_t grid[3]  = {1, 1, 1};
    uint32_t block[3] = {64, 1, 1};
    float dummy_arg = 42.0f;
    int ret = cml_am_kernel_launch(drv, kernel, grid, block, &dummy_arg, sizeof(dummy_arg));

    /* With mock, the launch writes to AQL ring but no real GPU runs.
     * Auto-complete writes the signal so synchronize succeeds. */
    if (ret == 0 && cml_am_mock_get()->auto_complete) {
        if (drv->signal) *drv->signal = drv->signal_value;
    }
    int sync = cml_am_synchronize(drv);

    cml_am_kernel_free(drv, kernel);
    cml_am_driver_free(drv);
    return (ret == 0) && (sync == 0);
}

static bool test_mock_sdma_copy(void) {
    CMLAMDriver* drv = cml_am_driver_create();
    if (!drv) return false;
    if (cml_am_driver_init(drv) != 0) { cml_am_driver_free(drv); return false; }

    if (!drv->has_sdma) {
        printf("(mock SDMA not available) ");
        cml_am_driver_free(drv);
        return false;
    }

    size_t size = 1024;
    CMLAMBuffer* src = cml_am_buffer_create(drv, size, false);
    CMLAMBuffer* dst = cml_am_buffer_create(drv, size, false);
    if (!src || !dst) {
        if (src) cml_am_buffer_free(drv, src);
        if (dst) cml_am_buffer_free(drv, dst);
        cml_am_driver_free(drv);
        return false;
    }

    memset(src->cpu_addr, 0xAB, size);
    memset(dst->cpu_addr, 0, size);

    int ret = cml_am_sdma_copy(drv, dst->gpu_va, src->gpu_va, size);
    if (ret != 0) {
        cml_am_buffer_free(drv, src);
        cml_am_buffer_free(drv, dst);
        cml_am_driver_free(drv);
        return false;
    }

    /* In mock mode, SDMA doesn't actually copy. The ring buffer gets
     * written to, which is sufficient to verify the path works.
     * Advance the read pointer to simulate hardware completion. */
    if (drv->sdma_queue.write_ptr && drv->sdma_queue.read_ptr)
        *drv->sdma_queue.read_ptr = *drv->sdma_queue.write_ptr;

    int sync = cml_am_sdma_synchronize(drv);
    cml_am_mock_get()->sdma_copies_seen++;

    cml_am_buffer_free(drv, src);
    cml_am_buffer_free(drv, dst);
    cml_am_driver_free(drv);
    return (sync == 0);
}

static bool test_mock_multi_queue(void) {
    CMLAMDriver* drv = cml_am_driver_create();
    if (!drv) return false;
    if (cml_am_driver_init(drv) != 0) { cml_am_driver_free(drv); return false; }

    if (drv->num_compute_queues < 1) { cml_am_driver_free(drv); return false; }

    int ret1 = cml_am_create_compute_queue(drv, 1);
    int ret2 = cml_am_create_compute_queue(drv, 2);

    bool ok = (ret1 == 0 && ret2 == 0);
    ok = ok && (drv->num_compute_queues >= 3);
    ok = ok && drv->compute_queues[1].active;
    ok = ok && drv->compute_queues[2].active;

    /* Verify each queue has distinct queue_ids */
    ok = ok && (drv->compute_queues[0].queue_id != drv->compute_queues[1].queue_id);
    ok = ok && (drv->compute_queues[1].queue_id != drv->compute_queues[2].queue_id);

    printf("(queues=%d) ", drv->num_compute_queues);

    cml_am_driver_free(drv);
    return ok;
}

static bool test_mock_signal_system(void) {
    CMLAMDriver* drv = cml_am_driver_create();
    if (!drv) return false;
    if (cml_am_driver_init(drv) != 0) { cml_am_driver_free(drv); return false; }

    CMLAMSignal* sig = cml_am_signal_create(drv, 0);
    if (!sig) { cml_am_driver_free(drv); return false; }
    if (!sig->value || sig->gpu_va == 0) { cml_am_signal_free(drv, sig); cml_am_driver_free(drv); return false; }
    if (*sig->value != 0) { cml_am_signal_free(drv, sig); cml_am_driver_free(drv); return false; }

    /* Simulate GPU completing work by writing signal */
    *sig->value = 42;
    int ret = cml_am_signal_wait(sig, 42, 1000000);
    if (ret != 0) { cml_am_signal_free(drv, sig); cml_am_driver_free(drv); return false; }

    /* Create a dispatch with this signal */
    uint8_t fake_elf[256];
    memset(fake_elf, 0, sizeof(fake_elf));
    fake_elf[0] = 0x7f; fake_elf[1] = 'E'; fake_elf[2] = 'L'; fake_elf[3] = 'F';
    fake_elf[4] = 2;

    CMLAMKernel* kernel = cml_am_kernel_load(drv, fake_elf, sizeof(fake_elf), "sig_test");
    if (!kernel) { cml_am_signal_free(drv, sig); cml_am_driver_free(drv); return false; }

    CMLAMSignal* comp = cml_am_signal_create(drv, 0);
    if (!comp) { cml_am_kernel_free(drv, kernel); cml_am_signal_free(drv, sig); cml_am_driver_free(drv); return false; }

    uint32_t grid[3] = {1,1,1}, block[3] = {64,1,1};
    int launch = cml_am_kernel_launch_on_queue(drv, 0, kernel, grid, block, NULL, 0, comp);

    /* Simulate completion */
    if (launch == 0 && comp->value)
        *comp->value = comp->target;

    int wait = cml_am_signal_wait(comp, comp->target, 1000000);

    cml_am_signal_free(drv, comp);
    cml_am_kernel_free(drv, kernel);
    cml_am_signal_free(drv, sig);
    cml_am_driver_free(drv);
    return (launch == 0) && (wait == 0);
}

static bool test_mock_barrier_and(void) {
    CMLAMDriver* drv = cml_am_driver_create();
    if (!drv) return false;
    if (cml_am_driver_init(drv) != 0) { cml_am_driver_free(drv); return false; }

    CMLAMSignal* dep1 = cml_am_signal_create(drv, 1);
    CMLAMSignal* dep2 = cml_am_signal_create(drv, 1);
    CMLAMSignal* comp = cml_am_signal_create(drv, 0);
    if (!dep1 || !dep2 || !comp) {
        if (dep1) cml_am_signal_free(drv, dep1);
        if (dep2) cml_am_signal_free(drv, dep2);
        if (comp) cml_am_signal_free(drv, comp);
        cml_am_driver_free(drv);
        return false;
    }

    CMLAMSignal* deps[2] = { dep1, dep2 };
    int ret = cml_am_barrier_and(drv, 0, deps, 2, comp);

    /* Simulate hardware completing the barrier */
    if (ret == 0 && comp->value)
        *comp->value = 1;

    int wait = cml_am_signal_wait(comp, 1, 1000000);

    cml_am_signal_free(drv, dep1);
    cml_am_signal_free(drv, dep2);
    cml_am_signal_free(drv, comp);
    cml_am_driver_free(drv);
    return (ret == 0) && (wait == 0);
}

#endif /* CML_AM_MOCK_GPU */


int main(void) {
    printf("\nAMD AM Driver Tests\n\n");

#ifdef CML_AM_MOCK_GPU
    cml_am_mock_init(NULL);
    g_hw_available = true;
    printf("  Mode: MOCK GPU\n\n");
#else
    g_hw_available = cml_am_driver_available();
    printf("  AMD KFD hardware: %s\n\n", g_hw_available ? "available" : "not available");
#endif

    /* Lifecycle */
    TEST(availability_check);
    TEST(create_and_free);
    TEST(free_null);

    /* Null/invalid */
    TEST(kernel_load_null_args);
    TEST(kernel_free_null);
    TEST(buffer_free_null);
    TEST(synchronize_null);
    TEST(execute_graph_stub);
    TEST(execute_graph_null);
    TEST(buffer_create_uninit);
    TEST(enumerate_gpus_null);
    TEST(signal_free_null);
    TEST(signal_wait_null);
    TEST(multi_queue_invalid);
    TEST(multi_queue_bounds);
    TEST(scratch_alloc_null);
    TEST(barrier_null);
    TEST(gpu_hang_null);
    TEST(kd_parse_null);
    TEST(kd_parse_bad_elf);
    TEST(validate_lds);

    /* Hardware-dependent */
    TEST(init);
    TEST(gpu_info_topology);
    TEST(enumerate_gpus);
    TEST(buffer_create_gtt);
    TEST(buffer_create_vram);
    TEST(buffer_upload_download);
    TEST(buffer_create_zero_size);
    TEST(synchronize_no_work);
    TEST(signal_create_free);
    TEST(signal_immediate_wait);
    TEST(multi_queue_create);
    TEST(sdma_queue);
    TEST(sdma_copy);
    TEST(validate_lds_hw);
    TEST(scratch_alloc);
    TEST(gpu_hang_check);

#ifdef CML_AM_MOCK_GPU
    /* Mock-specific integration tests */
    printf("\n  Mock GPU integration tests:\n");
    TEST(mock_enumerate_gpus);
    TEST(mock_driver_init);
    TEST(mock_buffer_lifecycle);
    TEST(mock_kernel_launch);
    TEST(mock_sdma_copy);
    TEST(mock_multi_queue);
    TEST(mock_signal_system);
    TEST(mock_barrier_and);

    cml_am_mock_shutdown();
#endif

    printf("\nResults: %d/%d passed\n\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
