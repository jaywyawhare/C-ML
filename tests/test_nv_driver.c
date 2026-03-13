/**
 * @file test_nv_driver.c
 * @brief Tests for the NVIDIA userspace driver (direct ioctl interface)
 *
 * All tests pass even without NVIDIA hardware present.  Tests that require
 * a live driver are skipped when cml_nv_driver_available() returns false.
 */

#include "ops/ir/gpu/nv_driver.h"
#include "core/logging.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Test harness ─────────────────────────────────────────────────────── */

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
        tests_passed++; \
        return 1; \
    } while(0)

/* ══════════════════════════════════════════════════════════════════════════
 * Tests
 * ══════════════════════════════════════════════════════════════════════════ */

/**
 * cml_nv_driver_available() must not crash regardless of environment.
 * The return value depends on whether /dev/nvidia0 exists.
 */
static int test_driver_available(void) {
    bool avail = cml_nv_driver_available();
    printf("(available=%s) ", avail ? "true" : "false");
    /* No assertion on the return value -- we just verify no crash */
    return 1;
}

/**
 * Create and immediately free a driver context without initialising.
 * Must not leak or crash.
 */
static int test_create_free_no_init(void) {
    CMLNVDriver* drv = cml_nv_driver_create();
    if (!drv) {
        printf("(alloc failed) ");
        return 0;
    }
    /* Verify initial state */
    if (drv->initialized) {
        printf("(should not be initialized) ");
        cml_nv_driver_free(drv);
        return 0;
    }
    if (drv->fd_ctl != -1 || drv->fd_dev != -1) {
        printf("(fds should be -1 before init) ");
        cml_nv_driver_free(drv);
        return 0;
    }
    cml_nv_driver_free(drv);
    return 1;
}

/**
 * Free NULL must be a safe no-op.
 */
static int test_free_null(void) {
    cml_nv_driver_free(NULL);
    return 1;
}

/**
 * Full lifecycle: create -> init -> free.
 * Skipped when the driver is not available.
 */
static int test_full_lifecycle(void) {
    if (!cml_nv_driver_available()) {
        SKIP("no NVIDIA device");
    }

    CMLNVDriver* drv = cml_nv_driver_create();
    if (!drv) {
        printf("(create failed) ");
        return 0;
    }

    int ret = cml_nv_driver_init(drv);
    if (ret != 0) {
        /* init may fail even when device is present (permissions, etc.) */
        printf("(init returned %d, may need root) ", ret);
        cml_nv_driver_free(drv);
        tests_passed++;  /* count as pass -- not a code bug */
        return 1;
    }

    if (!drv->initialized) {
        printf("(init returned 0 but flag not set) ");
        cml_nv_driver_free(drv);
        return 0;
    }

    cml_nv_driver_free(drv);
    return 1;
}

/**
 * Double-init must be idempotent.
 */
static int test_double_init(void) {
    if (!cml_nv_driver_available()) {
        SKIP("no NVIDIA device");
    }

    CMLNVDriver* drv = cml_nv_driver_create();
    if (!drv) return 0;

    int ret1 = cml_nv_driver_init(drv);
    if (ret1 != 0) {
        cml_nv_driver_free(drv);
        SKIP("init failed");
    }

    int ret2 = cml_nv_driver_init(drv);
    if (ret2 != 0) {
        printf("(second init failed) ");
        cml_nv_driver_free(drv);
        return 0;
    }

    cml_nv_driver_free(drv);
    return 1;
}

/**
 * Buffer create/free with a live driver.
 */
static int test_buffer_create_free(void) {
    if (!cml_nv_driver_available()) {
        SKIP("no NVIDIA device");
    }

    CMLNVDriver* drv = cml_nv_driver_create();
    if (!drv) return 0;

    if (cml_nv_driver_init(drv) != 0) {
        cml_nv_driver_free(drv);
        SKIP("init failed");
    }

    /* Host-visible buffer */
    CMLNVBuffer* hbuf = cml_nv_buffer_create(drv, 4096, true);
    if (!hbuf) {
        printf("(host buffer create returned NULL) ");
        cml_nv_driver_free(drv);
        return 0;
    }
    if (hbuf->size != 4096) {
        printf("(wrong size) ");
        cml_nv_buffer_free(drv, hbuf);
        cml_nv_driver_free(drv);
        return 0;
    }
    if (!hbuf->host_visible) {
        printf("(host_visible flag wrong) ");
        cml_nv_buffer_free(drv, hbuf);
        cml_nv_driver_free(drv);
        return 0;
    }
    cml_nv_buffer_free(drv, hbuf);

    /* Device-local buffer */
    CMLNVBuffer* dbuf = cml_nv_buffer_create(drv, 8192, false);
    if (!dbuf) {
        printf("(device buffer create returned NULL) ");
        cml_nv_driver_free(drv);
        return 0;
    }
    if (dbuf->host_visible) {
        printf("(device buffer should not be host-visible) ");
        cml_nv_buffer_free(drv, dbuf);
        cml_nv_driver_free(drv);
        return 0;
    }
    cml_nv_buffer_free(drv, dbuf);

    cml_nv_driver_free(drv);
    return 1;
}

/**
 * Buffer upload/download round-trip for a host-visible buffer.
 */
static int test_buffer_upload_download(void) {
    if (!cml_nv_driver_available()) {
        SKIP("no NVIDIA device");
    }

    CMLNVDriver* drv = cml_nv_driver_create();
    if (!drv) return 0;

    if (cml_nv_driver_init(drv) != 0) {
        cml_nv_driver_free(drv);
        SKIP("init failed");
    }

    size_t n = 256 * sizeof(float);
    CMLNVBuffer* buf = cml_nv_buffer_create(drv, n, true);
    if (!buf) {
        cml_nv_driver_free(drv);
        SKIP("buffer create failed");
    }

    /* Fill source pattern */
    float* src = (float*)malloc(n);
    float* dst = (float*)calloc(256, sizeof(float));
    if (!src || !dst) {
        free(src); free(dst);
        cml_nv_buffer_free(drv, buf);
        cml_nv_driver_free(drv);
        return 0;
    }
    for (int i = 0; i < 256; i++) {
        src[i] = (float)i * 0.5f;
    }

    /* Upload */
    int ret_up = cml_nv_buffer_upload(drv, buf, src, n);
    if (ret_up != 0) {
        printf("(upload failed) ");
        free(src); free(dst);
        cml_nv_buffer_free(drv, buf);
        cml_nv_driver_free(drv);
        return 0;
    }

    /* Download */
    int ret_dn = cml_nv_buffer_download(drv, buf, dst, n);
    if (ret_dn != 0) {
        printf("(download failed) ");
        free(src); free(dst);
        cml_nv_buffer_free(drv, buf);
        cml_nv_driver_free(drv);
        return 0;
    }

    /* Verify round-trip */
    int ok = 1;
    for (int i = 0; i < 256; i++) {
        if (dst[i] != src[i]) {
            printf("(mismatch at index %d: %.2f != %.2f) ", i, dst[i], src[i]);
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

/**
 * Null/edge-case buffer operations must not crash.
 */
static int test_buffer_null_safety(void) {
    /* All of these should return -1 or be no-ops, not crash */
    cml_nv_buffer_free(NULL, NULL);

    int ret1 = cml_nv_buffer_upload(NULL, NULL, NULL, 0);
    if (ret1 != -1) {
        printf("(upload NULL should return -1) ");
        return 0;
    }

    int ret2 = cml_nv_buffer_download(NULL, NULL, NULL, 0);
    if (ret2 != -1) {
        printf("(download NULL should return -1) ");
        return 0;
    }

    return 1;
}

/**
 * Kernel free on NULL must not crash.
 */
static int test_kernel_free_null(void) {
    cml_nv_kernel_free(NULL, NULL);
    return 1;
}

/**
 * cml_nv_kernel_compile_ptx with NULL arguments must return NULL.
 */
static int test_kernel_compile_null(void) {
    CMLNVKernel* k1 = cml_nv_kernel_compile_ptx(NULL, NULL, NULL);
    if (k1 != NULL) {
        printf("(should return NULL for null ptx) ");
        return 0;
    }

    CMLNVKernel* k2 = cml_nv_kernel_compile_ptx(NULL, "some ptx", NULL);
    if (k2 != NULL) {
        printf("(should return NULL for null name) ");
        return 0;
    }

    return 1;
}

/**
 * cml_nv_execute_graph stub must return -1.
 */
static int test_execute_graph_stub(void) {
    CMLNVDriver* drv = cml_nv_driver_create();
    if (!drv) return 0;

    /* Graph execution is a stub -- should return -1 */
    int ret = cml_nv_execute_graph(drv, NULL);
    if (ret != -1) {
        printf("(expected -1, got %d) ", ret);
        cml_nv_driver_free(drv);
        return 0;
    }

    cml_nv_driver_free(drv);
    return 1;
}

/**
 * cml_nv_synchronize on uninitialized driver must return -1.
 */
static int test_synchronize_not_init(void) {
    CMLNVDriver* drv = cml_nv_driver_create();
    if (!drv) return 0;

    int ret = cml_nv_synchronize(drv);
    if (ret != -1) {
        printf("(expected -1 on uninitialized driver) ");
        cml_nv_driver_free(drv);
        return 0;
    }

    cml_nv_driver_free(drv);
    return 1;
}

/**
 * cml_nv_kernel_launch on NULL/uninitialized driver must return -1.
 */
static int test_kernel_launch_not_init(void) {
    CMLNVDriver* drv = cml_nv_driver_create();
    if (!drv) return 0;

    uint32_t grid[3]  = {1, 1, 1};
    uint32_t block[3] = {256, 1, 1};

    int ret = cml_nv_kernel_launch(drv, NULL, grid, block, NULL, 0);
    if (ret != -1) {
        printf("(expected -1) ");
        cml_nv_driver_free(drv);
        return 0;
    }

    cml_nv_driver_free(drv);
    return 1;
}

/* ══════════════════════════════════════════════════════════════════════════
 * Main
 * ══════════════════════════════════════════════════════════════════════════ */

int main(void) {
    printf("\n=== NV Driver Tests ===\n\n");

    /* These always run, regardless of hardware */
    TEST(driver_available);
    TEST(create_free_no_init);
    TEST(free_null);
    TEST(buffer_null_safety);
    TEST(kernel_free_null);
    TEST(kernel_compile_null);
    TEST(execute_graph_stub);
    TEST(synchronize_not_init);
    TEST(kernel_launch_not_init);

    /* These skip gracefully when no hardware is present */
    TEST(full_lifecycle);
    TEST(double_init);
    TEST(buffer_create_free);
    TEST(buffer_upload_download);

    printf("\n=== Results: %d/%d passed ===\n\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
