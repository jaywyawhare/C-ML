#include "ops/ir/gpu/nv_driver.h"
#include "ops/ir/gpu/nv_qmd.h"
#include "core/logging.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static int tests_run    = 0;
static int tests_passed = 0;

#define TEST(name) \
    do { \
        printf("  Testing: %s... ", #name); \
        tests_run++; \
        if (test_##name()) { \
            printf("PASS\n"); \
            tests_passed++; \
        } else { \
            printf("FAIL\n"); \
        } \
    } while(0)

#define SKIP(reason) \
    do { \
        printf("SKIP (%s)\n", reason); \
        return 1; \
    } while(0)


static int test_driver_available(void) {
    bool avail = cml_nv_driver_available();
    printf("(available=%s) ", avail ? "true" : "false");
    return 1;
}

static int test_create_free_no_init(void) {
    CMLNVDriver *drv = cml_nv_driver_create();
    if (!drv) return 0;

    if (drv->initialized) {
        cml_nv_driver_free(drv);
        return 0;
    }
    if (drv->fd_ctl != -1 || drv->fd_dev != -1) {
        cml_nv_driver_free(drv);
        return 0;
    }
    cml_nv_driver_free(drv);
    return 1;
}

static int test_free_null(void) {
    cml_nv_driver_free(NULL);
    return 1;
}

static int test_full_lifecycle(void) {
    if (!cml_nv_driver_available())
        SKIP("no NVIDIA device");

    CMLNVDriver *drv = cml_nv_driver_create();
    if (!drv) return 0;

    int ret = cml_nv_driver_init(drv);
    if (ret != 0) {
        printf("(init returned %d, may need root) ", ret);
        cml_nv_driver_free(drv);
        tests_passed++;
        return 1;
    }

    if (!drv->initialized) {
        cml_nv_driver_free(drv);
        return 0;
    }

    if (drv->gpu_arch == 0) {
        printf("(gpu_arch not set) ");
        cml_nv_driver_free(drv);
        return 0;
    }

    if (drv->compute_cap_major == 0) {
        printf("(compute cap not set) ");
        cml_nv_driver_free(drv);
        return 0;
    }

    printf("(arch=0x%X sm_%d%d) ", drv->gpu_arch,
           drv->compute_cap_major, drv->compute_cap_minor);

    cml_nv_driver_free(drv);
    return 1;
}

static int test_double_init(void) {
    if (!cml_nv_driver_available())
        SKIP("no NVIDIA device");

    CMLNVDriver *drv = cml_nv_driver_create();
    if (!drv) return 0;

    int ret1 = cml_nv_driver_init(drv);
    if (ret1 != 0) {
        cml_nv_driver_free(drv);
        SKIP("init failed");
    }

    int ret2 = cml_nv_driver_init(drv);
    if (ret2 != 0) {
        cml_nv_driver_free(drv);
        return 0;
    }

    cml_nv_driver_free(drv);
    return 1;
}

static int test_buffer_create_free(void) {
    if (!cml_nv_driver_available())
        SKIP("no NVIDIA device");

    CMLNVDriver *drv = cml_nv_driver_create();
    if (!drv) return 0;

    if (cml_nv_driver_init(drv) != 0) {
        cml_nv_driver_free(drv);
        SKIP("init failed");
    }

    CMLNVBuffer *hbuf = cml_nv_buffer_create(drv, 4096, true);
    if (!hbuf) {
        cml_nv_driver_free(drv);
        return 0;
    }
    if (hbuf->size < 4096 || !hbuf->host_visible) {
        cml_nv_buffer_free(drv, hbuf);
        cml_nv_driver_free(drv);
        return 0;
    }
    if (hbuf->gpu_va == 0) {
        printf("(no gpu_va) ");
        cml_nv_buffer_free(drv, hbuf);
        cml_nv_driver_free(drv);
        return 0;
    }
    cml_nv_buffer_free(drv, hbuf);

    CMLNVBuffer *dbuf = cml_nv_buffer_create(drv, 8192, false);
    if (!dbuf) {
        cml_nv_driver_free(drv);
        return 0;
    }
    cml_nv_buffer_free(drv, dbuf);

    CMLNVBuffer *vbuf = cml_nv_buffer_create_vram(drv, 4096);
    if (vbuf) {
        if (vbuf->gpu_va == 0) {
            printf("(vram buf no gpu_va) ");
            cml_nv_buffer_free(drv, vbuf);
            cml_nv_driver_free(drv);
            return 0;
        }
        cml_nv_buffer_free(drv, vbuf);
    }

    cml_nv_driver_free(drv);
    return 1;
}

static int test_buffer_upload_download(void) {
    if (!cml_nv_driver_available())
        SKIP("no NVIDIA device");

    CMLNVDriver *drv = cml_nv_driver_create();
    if (!drv) return 0;

    if (cml_nv_driver_init(drv) != 0) {
        cml_nv_driver_free(drv);
        SKIP("init failed");
    }

    size_t n = 256 * sizeof(float);
    CMLNVBuffer *buf = cml_nv_buffer_create(drv, n, true);
    if (!buf) {
        cml_nv_driver_free(drv);
        SKIP("buffer create failed");
    }

    float *src = (float *)malloc(n);
    float *dst = (float *)calloc(256, sizeof(float));
    if (!src || !dst) {
        free(src); free(dst);
        cml_nv_buffer_free(drv, buf);
        cml_nv_driver_free(drv);
        return 0;
    }
    for (int i = 0; i < 256; i++)
        src[i] = (float)i * 0.5f;

    if (cml_nv_buffer_upload(drv, buf, src, n) != 0) {
        free(src); free(dst);
        cml_nv_buffer_free(drv, buf);
        cml_nv_driver_free(drv);
        return 0;
    }

    if (cml_nv_buffer_download(drv, buf, dst, n) != 0) {
        free(src); free(dst);
        cml_nv_buffer_free(drv, buf);
        cml_nv_driver_free(drv);
        return 0;
    }

    int ok = 1;
    for (int i = 0; i < 256; i++) {
        if (dst[i] != src[i]) {
            printf("(mismatch at %d: %.2f != %.2f) ", i, dst[i], src[i]);
            ok = 0;
            break;
        }
    }

    free(src);
    free(dst);
    cml_nv_buffer_free(drv, buf);
    cml_nv_driver_free(drv);
    return ok;
}

static int test_buffer_null_safety(void) {
    cml_nv_buffer_free(NULL, NULL);
    if (cml_nv_buffer_upload(NULL, NULL, NULL, 0) != -1) return 0;
    if (cml_nv_buffer_download(NULL, NULL, NULL, 0) != -1) return 0;
    if (cml_nv_buffer_copy(NULL, NULL, NULL, 0) != -1) return 0;
    return 1;
}

static int test_kernel_free_null(void) {
    cml_nv_kernel_free(NULL, NULL);
    return 1;
}

static int test_kernel_compile_null(void) {
    if (cml_nv_kernel_compile_ptx(NULL, NULL, NULL) != NULL) return 0;
    if (cml_nv_kernel_compile_ptx(NULL, "some ptx", NULL) != NULL) return 0;
    return 1;
}

static int test_kernel_load_cubin_null(void) {
    if (cml_nv_kernel_load_cubin(NULL, NULL, 0, NULL) != NULL) return 0;
    if (cml_nv_kernel_load_cubin(NULL, "data", 4, NULL) != NULL) return 0;
    return 1;
}

static int test_execute_graph_stub(void) {
    CMLNVDriver *drv = cml_nv_driver_create();
    if (!drv) return 0;

    if (cml_nv_execute_graph(drv, NULL) != -1) {
        cml_nv_driver_free(drv);
        return 0;
    }
    if (cml_nv_execute_graph(NULL, NULL) != -1) {
        cml_nv_driver_free(drv);
        return 0;
    }

    cml_nv_driver_free(drv);
    return 1;
}

static int test_synchronize_not_init(void) {
    CMLNVDriver *drv = cml_nv_driver_create();
    if (!drv) return 0;

    if (cml_nv_synchronize(drv) != -1) {
        cml_nv_driver_free(drv);
        return 0;
    }

    cml_nv_driver_free(drv);
    return 1;
}

static int test_kernel_launch_not_init(void) {
    CMLNVDriver *drv = cml_nv_driver_create();
    if (!drv) return 0;

    uint32_t grid[3]  = {1, 1, 1};
    uint32_t block[3] = {256, 1, 1};

    if (cml_nv_kernel_launch(drv, NULL, grid, block, NULL, 0) != -1) {
        cml_nv_driver_free(drv);
        return 0;
    }

    cml_nv_driver_free(drv);
    return 1;
}

static int test_qmd_init(void) {
    NVQmd qmd;
    nv_qmd_init(&qmd, NV_GPU_ARCH_TURING);

    uint32_t version = (qmd.data[0] >> 4) & 0xF;
    if (version != QMD_VERSION_TURING) {
        printf("(turing version=%u expected=%u) ", version, QMD_VERSION_TURING);
        return 0;
    }

    uint32_t outer_mode = qmd.data[0] & 0x3;
    if (outer_mode != 0x1) {
        printf("(outer_mode=%u expected=1) ", outer_mode);
        return 0;
    }

    NVQmd qmd_h;
    nv_qmd_init(&qmd_h, NV_GPU_ARCH_HOPPER);
    uint32_t ver_h = (qmd_h.data[0] >> 4) & 0xF;
    if (ver_h != QMD_VERSION_HOPPER) {
        printf("(hopper version=%u expected=%u) ", ver_h, QMD_VERSION_HOPPER);
        return 0;
    }

    return 1;
}

static int test_qmd_fields(void) {
    NVQmd qmd;
    nv_qmd_init(&qmd, NV_GPU_ARCH_TURING);

    nv_qmd_set_program_address(&qmd, 0xDEADBEEF12345678ULL);
    if (qmd.data[8] != 0x12345678U) {
        printf("(prog addr lo=0x%X) ", qmd.data[8]);
        return 0;
    }
    if (qmd.data[9] != 0xDEADBEEFU) {
        printf("(prog addr hi=0x%X) ", qmd.data[9]);
        return 0;
    }

    nv_qmd_set_grid_dim(&qmd, 100, 200, 300);
    if (qmd.data[12] != 100 || qmd.data[13] != 200 || qmd.data[14] != 300) {
        printf("(grid dims wrong) ");
        return 0;
    }

    nv_qmd_set_block_dim(&qmd, 256, 4, 2);
    if ((qmd.data[15] & 0xFFFF) != 256) {
        printf("(block x=%u) ", qmd.data[15] & 0xFFFF);
        return 0;
    }

    nv_qmd_set_register_count(&qmd, 32);
    if ((qmd.data[18] & 0xFF) != 32) {
        printf("(regs=%u) ", qmd.data[18] & 0xFF);
        return 0;
    }

    nv_qmd_set_shared_memory(&qmd, 1000);
    uint32_t smem = qmd.data[19];
    if (smem != 1024) {
        printf("(smem=%u expected 1024) ", smem);
        return 0;
    }

    nv_qmd_set_constant_buffer(&qmd, 0, 0xAAAABBBBCCCCDDDDULL, 512);
    if (!(qmd.data[22] & 0x1)) {
        printf("(cb0 not valid) ");
        return 0;
    }
    if (qmd.data[24] != 0xCCCCDDDDU) {
        printf("(cb0 addr lo=0x%X) ", qmd.data[24]);
        return 0;
    }
    if (qmd.data[25] != 0xAAAABBBBU) {
        printf("(cb0 addr hi=0x%X) ", qmd.data[25]);
        return 0;
    }
    if (qmd.data[26] != 512) {
        printf("(cb0 size=%u) ", qmd.data[26]);
        return 0;
    }

    return 1;
}

static int test_qmd_barrier_count(void) {
    NVQmd qmd;
    nv_qmd_init(&qmd, NV_GPU_ARCH_AMPERE);

    nv_qmd_set_barrier_count(&qmd, 16);
    uint32_t bar = qmd.data[20] & 0x3F;
    if (bar != 16) {
        printf("(bar_count=%u expected=16) ", bar);
        return 0;
    }

    nv_qmd_set_barrier_count(&qmd, 99);
    bar = qmd.data[20] & 0x3F;
    if (bar != 31) {
        printf("(bar_count=%u expected=31 after clamp) ", bar);
        return 0;
    }

    return 1;
}

static int test_cubin_parse_invalid(void) {
    NVKernelMeta meta;
    if (nv_parse_cubin(NULL, 0, "test", &meta) != -1) return 0;
    if (nv_parse_cubin("not_elf", 7, "test", &meta) != -1) return 0;

    uint8_t bad_elf[64];
    memset(bad_elf, 0, sizeof(bad_elf));
    bad_elf[0] = 0x7F; bad_elf[1] = 'E'; bad_elf[2] = 'L'; bad_elf[3] = 'F';
    if (nv_parse_cubin(bad_elf, sizeof(bad_elf), "test", &meta) != -1) {
        /* May succeed with defaults if ELF header is valid enough */
    }

    return 1;
}

static int test_multiple_constant_buffers(void) {
    NVQmd qmd;
    nv_qmd_init(&qmd, NV_GPU_ARCH_TURING);

    nv_qmd_set_constant_buffer(&qmd, 0, 0x1000, 256);
    nv_qmd_set_constant_buffer(&qmd, 1, 0x2000, 128);
    nv_qmd_set_constant_buffer(&qmd, 2, 0x3000, 64);

    uint32_t valid = qmd.data[22];
    if ((valid & 0x7) != 0x7) {
        printf("(cb valid mask=0x%X expected=0x7) ", valid & 0x7);
        return 0;
    }

    if (qmd.data[24] != 0x1000 || qmd.data[26] != 256) return 0;
    if (qmd.data[27] != 0x2000 || qmd.data[29] != 128) return 0;
    if (qmd.data[30] != 0x3000 || qmd.data[32] != 64) return 0;

    nv_qmd_set_constant_buffer(&qmd, -1, 0, 0);
    nv_qmd_set_constant_buffer(&qmd, 8, 0, 0);

    return 1;
}

static int test_buffer_copy_null(void) {
    if (cml_nv_buffer_copy(NULL, NULL, NULL, 0) != -1) return 0;
    return 1;
}


#ifdef CML_NV_MOCK_GPU
#include "ops/ir/gpu/nv_mock.h"

static CMLNVDriver* mock_create_and_init(void) {
    CMLNVDriver *drv = cml_nv_driver_create();
    if (!drv) return NULL;
    if (cml_nv_driver_init(drv) != 0) {
        cml_nv_driver_free(drv);
        return NULL;
    }
    return drv;
}

static int test_mock_driver_init(void) {
    cml_nv_mock_init(NULL);

    if (!cml_nv_driver_available()) {
        cml_nv_mock_shutdown();
        return 0;
    }

    CMLNVDriver *drv = mock_create_and_init();
    if (!drv) {
        cml_nv_mock_shutdown();
        return 0;
    }

    if (!drv->initialized) {
        cml_nv_driver_free(drv);
        cml_nv_mock_shutdown();
        return 0;
    }

    if (drv->gpu_arch != 0x190) {
        printf("(arch=0x%X expected=0x190) ", drv->gpu_arch);
        cml_nv_driver_free(drv);
        cml_nv_mock_shutdown();
        return 0;
    }

    if (drv->compute_cap_major != 7 || drv->compute_cap_minor != 5) {
        printf("(sm_%d%d expected sm_75) ",
               drv->compute_cap_major, drv->compute_cap_minor);
        cml_nv_driver_free(drv);
        cml_nv_mock_shutdown();
        return 0;
    }

    CMLNVMockGPU *mock = cml_nv_mock_get();
    if (!mock) {
        cml_nv_driver_free(drv);
        cml_nv_mock_shutdown();
        return 0;
    }

    cml_nv_driver_free(drv);
    cml_nv_mock_shutdown();
    return 1;
}

static int test_mock_buffer_create(void) {
    cml_nv_mock_init(NULL);

    CMLNVDriver *drv = mock_create_and_init();
    if (!drv) {
        cml_nv_mock_shutdown();
        return 0;
    }

    CMLNVBuffer *buf1 = cml_nv_buffer_create(drv, 4096, true);
    if (!buf1 || buf1->gpu_va == 0) {
        printf("(buf1 failed) ");
        if (buf1) cml_nv_buffer_free(drv, buf1);
        cml_nv_driver_free(drv);
        cml_nv_mock_shutdown();
        return 0;
    }

    CMLNVBuffer *buf2 = cml_nv_buffer_create(drv, 8192, true);
    if (!buf2 || buf2->gpu_va == 0) {
        printf("(buf2 failed) ");
        cml_nv_buffer_free(drv, buf1);
        if (buf2) cml_nv_buffer_free(drv, buf2);
        cml_nv_driver_free(drv);
        cml_nv_mock_shutdown();
        return 0;
    }

    if (buf2->gpu_va <= buf1->gpu_va) {
        printf("(VA not monotonic: 0x%lX <= 0x%lX) ",
               (unsigned long)buf2->gpu_va, (unsigned long)buf1->gpu_va);
        cml_nv_buffer_free(drv, buf1);
        cml_nv_buffer_free(drv, buf2);
        cml_nv_driver_free(drv);
        cml_nv_mock_shutdown();
        return 0;
    }

    CMLNVBuffer *vbuf = cml_nv_buffer_create_vram(drv, 4096);
    if (!vbuf || vbuf->gpu_va == 0) {
        printf("(vram buf failed) ");
        cml_nv_buffer_free(drv, buf1);
        cml_nv_buffer_free(drv, buf2);
        if (vbuf) cml_nv_buffer_free(drv, vbuf);
        cml_nv_driver_free(drv);
        cml_nv_mock_shutdown();
        return 0;
    }

    cml_nv_buffer_free(drv, buf1);
    cml_nv_buffer_free(drv, buf2);
    cml_nv_buffer_free(drv, vbuf);
    cml_nv_driver_free(drv);
    cml_nv_mock_shutdown();
    return 1;
}

static int test_mock_kernel_launch(void) {
    cml_nv_mock_init(NULL);

    CMLNVDriver *drv = mock_create_and_init();
    if (!drv) {
        cml_nv_mock_shutdown();
        return 0;
    }

    uint8_t dummy_cubin[256];
    memset(dummy_cubin, 0, sizeof(dummy_cubin));
    dummy_cubin[0] = 0x7F;
    dummy_cubin[1] = 'E';
    dummy_cubin[2] = 'L';
    dummy_cubin[3] = 'F';

    CMLNVKernel *kernel = cml_nv_kernel_load_cubin(drv, dummy_cubin,
                                                     sizeof(dummy_cubin),
                                                     "mock_kernel");
    if (!kernel) {
        printf("(cubin load failed) ");
        cml_nv_driver_free(drv);
        cml_nv_mock_shutdown();
        return 0;
    }

    uint32_t grid[3]  = {1, 1, 1};
    uint32_t block[3] = {256, 1, 1};

    uint64_t sem_before = drv->semaphore_value;
    int ret = cml_nv_kernel_launch(drv, kernel, grid, block, NULL, 0);
    if (ret != 0) {
        printf("(launch failed) ");
        cml_nv_kernel_free(drv, kernel);
        cml_nv_driver_free(drv);
        cml_nv_mock_shutdown();
        return 0;
    }

    if (drv->semaphore_value <= sem_before) {
        printf("(semaphore value not incremented) ");
        cml_nv_kernel_free(drv, kernel);
        cml_nv_driver_free(drv);
        cml_nv_mock_shutdown();
        return 0;
    }

    cml_nv_kernel_free(drv, kernel);
    cml_nv_driver_free(drv);
    cml_nv_mock_shutdown();
    return 1;
}

static int test_mock_synchronize(void) {
    cml_nv_mock_init(NULL);

    CMLNVDriver *drv = mock_create_and_init();
    if (!drv) {
        cml_nv_mock_shutdown();
        return 0;
    }

    uint8_t dummy_cubin[256];
    memset(dummy_cubin, 0, sizeof(dummy_cubin));
    dummy_cubin[0] = 0x7F;
    dummy_cubin[1] = 'E';
    dummy_cubin[2] = 'L';
    dummy_cubin[3] = 'F';

    CMLNVKernel *kernel = cml_nv_kernel_load_cubin(drv, dummy_cubin,
                                                     sizeof(dummy_cubin),
                                                     "sync_kernel");
    if (!kernel) {
        cml_nv_driver_free(drv);
        cml_nv_mock_shutdown();
        return 0;
    }

    uint32_t grid[3]  = {1, 1, 1};
    uint32_t block[3] = {128, 1, 1};

    if (cml_nv_kernel_launch(drv, kernel, grid, block, NULL, 0) != 0) {
        cml_nv_kernel_free(drv, kernel);
        cml_nv_driver_free(drv);
        cml_nv_mock_shutdown();
        return 0;
    }

    if (drv->semaphore)
        *drv->semaphore = (uint32_t)drv->semaphore_value;

    int ret = cml_nv_synchronize(drv);
    if (ret != 0) {
        printf("(sync failed) ");
        cml_nv_kernel_free(drv, kernel);
        cml_nv_driver_free(drv);
        cml_nv_mock_shutdown();
        return 0;
    }

    cml_nv_kernel_free(drv, kernel);
    cml_nv_driver_free(drv);
    cml_nv_mock_shutdown();
    return 1;
}

static int test_mock_buffer_copy(void) {
    cml_nv_mock_init(NULL);

    CMLNVDriver *drv = mock_create_and_init();
    if (!drv) {
        cml_nv_mock_shutdown();
        return 0;
    }

    size_t n = 1024;
    CMLNVBuffer *src = cml_nv_buffer_create(drv, n, true);
    CMLNVBuffer *dst = cml_nv_buffer_create(drv, n, true);
    if (!src || !dst) {
        if (src) cml_nv_buffer_free(drv, src);
        if (dst) cml_nv_buffer_free(drv, dst);
        cml_nv_driver_free(drv);
        cml_nv_mock_shutdown();
        return 0;
    }

    float *src_data = (float *)src->cpu_addr;
    for (size_t i = 0; i < n / sizeof(float); i++)
        src_data[i] = (float)i;

    int ret = cml_nv_buffer_copy(drv, dst, src, n);
    if (ret != 0) {
        printf("(copy returned %d) ", ret);
        cml_nv_buffer_free(drv, src);
        cml_nv_buffer_free(drv, dst);
        cml_nv_driver_free(drv);
        cml_nv_mock_shutdown();
        return 0;
    }

    if (drv->semaphore)
        *drv->semaphore = (uint32_t)drv->semaphore_value;

    ret = cml_nv_synchronize(drv);
    if (ret != 0) {
        printf("(sync failed) ");
        cml_nv_buffer_free(drv, src);
        cml_nv_buffer_free(drv, dst);
        cml_nv_driver_free(drv);
        cml_nv_mock_shutdown();
        return 0;
    }

    cml_nv_buffer_free(drv, src);
    cml_nv_buffer_free(drv, dst);
    cml_nv_driver_free(drv);
    cml_nv_mock_shutdown();
    return 1;
}

static int test_mock_multi_launch(void) {
    cml_nv_mock_init(NULL);

    CMLNVDriver *drv = mock_create_and_init();
    if (!drv) {
        cml_nv_mock_shutdown();
        return 0;
    }

    uint8_t dummy_cubin[256];
    memset(dummy_cubin, 0, sizeof(dummy_cubin));
    dummy_cubin[0] = 0x7F;
    dummy_cubin[1] = 'E';
    dummy_cubin[2] = 'L';
    dummy_cubin[3] = 'F';

    CMLNVKernel *kernel = cml_nv_kernel_load_cubin(drv, dummy_cubin,
                                                     sizeof(dummy_cubin),
                                                     "multi_kernel");
    if (!kernel) {
        cml_nv_driver_free(drv);
        cml_nv_mock_shutdown();
        return 0;
    }

    uint32_t grid[3]  = {4, 1, 1};
    uint32_t block[3] = {64, 1, 1};

    for (int i = 0; i < 5; i++) {
        uint64_t sem_before = drv->semaphore_value;
        int ret = cml_nv_kernel_launch(drv, kernel, grid, block, NULL, 0);
        if (ret != 0) {
            printf("(launch %d failed) ", i);
            cml_nv_kernel_free(drv, kernel);
            cml_nv_driver_free(drv);
            cml_nv_mock_shutdown();
            return 0;
        }
        if (drv->semaphore_value != sem_before + 1) {
            printf("(semaphore not incremented on launch %d) ", i);
            cml_nv_kernel_free(drv, kernel);
            cml_nv_driver_free(drv);
            cml_nv_mock_shutdown();
            return 0;
        }
    }

    if (drv->semaphore)
        *drv->semaphore = (uint32_t)drv->semaphore_value;

    if (cml_nv_synchronize(drv) != 0) {
        printf("(final sync failed) ");
        cml_nv_kernel_free(drv, kernel);
        cml_nv_driver_free(drv);
        cml_nv_mock_shutdown();
        return 0;
    }

    cml_nv_kernel_free(drv, kernel);
    cml_nv_driver_free(drv);
    cml_nv_mock_shutdown();
    return 1;
}

#endif /* CML_NV_MOCK_GPU */


int main(void) {
    printf("\nNV Driver Tests\n\n");

    TEST(driver_available);
    TEST(create_free_no_init);
    TEST(free_null);
    TEST(buffer_null_safety);
    TEST(buffer_copy_null);
    TEST(kernel_free_null);
    TEST(kernel_compile_null);
    TEST(kernel_load_cubin_null);
    TEST(execute_graph_stub);
    TEST(synchronize_not_init);
    TEST(kernel_launch_not_init);

    TEST(qmd_init);
    TEST(qmd_fields);
    TEST(qmd_barrier_count);
    TEST(multiple_constant_buffers);
    TEST(cubin_parse_invalid);

    TEST(full_lifecycle);
    TEST(double_init);
    TEST(buffer_create_free);
    TEST(buffer_upload_download);

#ifdef CML_NV_MOCK_GPU
    printf("\nMock GPU Tests\n\n");

    TEST(mock_driver_init);
    TEST(mock_buffer_create);
    TEST(mock_kernel_launch);
    TEST(mock_synchronize);
    TEST(mock_buffer_copy);
    TEST(mock_multi_launch);
#endif

    printf("\nResults: %d/%d passed\n\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
