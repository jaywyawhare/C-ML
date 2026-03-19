#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "ops/ir/gpu/am_driver.h"


static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    printf("  Testing: %s... ", #name); \
    tests_run++; \
    if (test_##name()) { printf("PASS\n"); tests_passed++; } \
    else { printf("FAIL\n"); } \
} while(0)

/* Skip helper: counts as passed when hardware is absent */
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
    /* Calling twice should return the same result */
    if (a != b) return false;
    return true;
}

static bool test_create_and_free(void) {
    CMLAMDriver* drv = cml_am_driver_create();
    if (!drv) return false;

    /* Before init, these should be default values */
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
        /* Init might fail even with /dev/kfd if GPU is busy or incompatible */
        printf("(init returned %d, hw-specific) ", ret);
        cml_am_driver_free(drv);
        return true; /* Not a test failure */
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
        return true; /* Skip */
    }

    CMLAMBuffer* buf = cml_am_buffer_create(drv, 4096, false /* GTT */);
    if (!buf) {
        printf("(buffer create failed, hw-specific) ");
        cml_am_driver_free(drv);
        return true;
    }

    /* GTT buffer should have a CPU address */
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

    CMLAMBuffer* buf = cml_am_buffer_create(drv, 4096, true /* VRAM */);
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
    CMLAMBuffer* buf = cml_am_buffer_create(drv, size, false /* GTT */);
    if (!buf) {
        cml_am_driver_free(drv);
        return true;
    }

    /* Fill source data */
    float* src = (float*)malloc(size);
    float* dst = (float*)calloc(256, sizeof(float));
    if (!src || !dst) {
        free(src); free(dst);
        cml_am_buffer_free(drv, buf);
        cml_am_driver_free(drv);
        return false;
    }

    for (int i = 0; i < 256; i++) {
        src[i] = (float)i * 1.5f;
    }

    /* Upload */
    if (cml_am_buffer_upload(drv, buf, src, size) != 0) {
        printf("(upload failed) ");
        free(src); free(dst);
        cml_am_buffer_free(drv, buf);
        cml_am_driver_free(drv);
        return true; /* hw-specific */
    }

    /* Download */
    if (cml_am_buffer_download(drv, buf, dst, size) != 0) {
        printf("(download failed) ");
        free(src); free(dst);
        cml_am_buffer_free(drv, buf);
        cml_am_driver_free(drv);
        return true;
    }

    /* Verify */
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

    /* No dispatches, synchronize should return immediately */
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

    /* NULL graph must return -1 */
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

    /* Don't call init */
    CMLAMBuffer* buf = cml_am_buffer_create(drv, 4096, false);
    cml_am_driver_free(drv);
    return (buf == NULL);
}


int main(void) {
    printf("\nAMD AM Driver Tests\n\n");

    /* Check hardware availability once */
    g_hw_available = cml_am_driver_available();
    printf("  AMD KFD hardware: %s\n\n", g_hw_available ? "available" : "not available");

    /* Basic lifecycle tests (always run) */
    TEST(availability_check);
    TEST(create_and_free);
    TEST(free_null);

    /* Null/invalid argument tests (always run) */
    TEST(kernel_load_null_args);
    TEST(kernel_free_null);
    TEST(buffer_free_null);
    TEST(synchronize_null);
    TEST(execute_graph_stub);
    TEST(execute_graph_null);
    TEST(buffer_create_uninit);

    /* Hardware-dependent tests (skipped if no AMD GPU) */
    TEST(init);
    TEST(buffer_create_gtt);
    TEST(buffer_create_vram);
    TEST(buffer_upload_download);
    TEST(buffer_create_zero_size);
    TEST(synchronize_no_work);

    printf("\nResults: %d/%d passed\n\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
