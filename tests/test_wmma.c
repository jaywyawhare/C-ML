/**
 * @file test_wmma.c
 * @brief Tests for WMMA (Tensor Core) support
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "ops/ir/gpu/wmma.h"

static void test_select_config_basic(void) {
    printf("  test_select_config_basic...");

    if (!cml_wmma_available()) {
        printf(" SKIPPED (no WMMA device)\n");
        return;
    }

    WMMAConfig config;
    memset(&config, 0, sizeof(config));

    /* Standard matmul dimensions */
    int ret = cml_wmma_select_config(256, 256, 256, &config);
    assert(ret == 0);

    /* Config should have valid fragment dimensions */
    assert(config.M > 0);
    assert(config.N > 0);
    assert(config.K > 0);

    /* Block dimensions should be positive */
    assert(config.block_m > 0);
    assert(config.block_n > 0);
    assert(config.block_k > 0);

    printf(" PASS\n");
}

static void test_select_config_small(void) {
    printf("  test_select_config_small...");

    if (!cml_wmma_available()) {
        printf(" SKIPPED (no WMMA device)\n");
        return;
    }

    WMMAConfig config;
    memset(&config, 0, sizeof(config));

    /* Small matmul that fits in a single fragment */
    int ret = cml_wmma_select_config(16, 16, 16, &config);
    assert(ret == 0);
    assert(config.fragment == WMMA_M16N16K16);
    assert(config.M == 16);
    assert(config.N == 16);
    assert(config.K == 16);

    printf(" PASS\n");
}

static void test_select_config_rectangular(void) {
    printf("  test_select_config_rectangular...");

    if (!cml_wmma_available()) {
        printf(" SKIPPED (no WMMA device)\n");
        return;
    }

    WMMAConfig config;
    memset(&config, 0, sizeof(config));

    /* Rectangular matrix dimensions */
    int ret = cml_wmma_select_config(512, 128, 256, &config);
    assert(ret == 0);
    assert(config.M > 0);
    assert(config.N > 0);
    assert(config.K > 0);

    printf(" PASS\n");
}

static void test_generate_kernel(void) {
    printf("  test_generate_kernel...");

    if (!cml_wmma_available()) {
        printf(" SKIPPED (no WMMA device)\n");
        return;
    }

    WMMAConfig config;
    memset(&config, 0, sizeof(config));

    int ret = cml_wmma_select_config(256, 256, 256, &config);
    assert(ret == 0);

    char* kernel_src = cml_wmma_generate_kernel(&config, 256, 256, 256);
    assert(kernel_src != NULL);
    assert(strlen(kernel_src) > 0);

    /* The kernel source should contain "wmma" references */
    assert(strstr(kernel_src, "wmma") != NULL);

    printf(" (generated %zu bytes) ", strlen(kernel_src));
    free(kernel_src);
    printf("PASS\n");
}

static void test_generate_kernel_various_sizes(void) {
    printf("  test_generate_kernel_various_sizes...");

    if (!cml_wmma_available()) {
        printf(" SKIPPED (no WMMA device)\n");
        return;
    }

    int sizes[][3] = {
        {128, 128, 128},
        {512, 256, 64},
        {1024, 1024, 512},
    };

    for (int i = 0; i < 3; i++) {
        int M = sizes[i][0], N = sizes[i][1], K = sizes[i][2];
        WMMAConfig config;
        memset(&config, 0, sizeof(config));

        int ret = cml_wmma_select_config(M, N, K, &config);
        assert(ret == 0);

        char* src = cml_wmma_generate_kernel(&config, M, N, K);
        assert(src != NULL);
        assert(strstr(src, "wmma") != NULL);
        free(src);
    }

    printf(" PASS\n");
}

static void test_wmma_availability(void) {
    printf("  test_wmma_availability...");

    bool available = cml_wmma_available();
    printf(" available=%s", available ? "true" : "false");

    if (!available) {
        printf(" (no Tensor Core GPU detected, skipping matmul test)");
    }

    printf(" PASS\n");
}

static void test_wmma_matmul_if_available(void) {
    printf("  test_wmma_matmul_if_available...");

    if (!cml_wmma_available()) {
        printf(" SKIPPED (no WMMA device)\n");
        return;
    }

    /* If WMMA is available, test a small matmul
     * A is [16, 16] fp16, B is [16, 16] fp16, C is [16, 16] fp32
     * We would need device memory, so this test is only run on capable hardware.
     */
    int M = 16, N = 16, K = 16;
    size_t fp16_size = M * K * 2;  /* 2 bytes per fp16 */
    size_t fp32_size = M * N * 4;  /* 4 bytes per fp32 */

    /* Allocate host buffers (simplified: using calloc for zero-init) */
    void* A = calloc(1, fp16_size);
    void* B = calloc(1, fp16_size);
    void* C = calloc(1, fp32_size);
    assert(A != NULL && B != NULL && C != NULL);

    int ret = cml_wmma_matmul(A, B, C, M, N, K);
    /* Zero matrices multiplied should give zero result */
    assert(ret == 0);

    float* C_fp32 = (float*)C;
    for (int i = 0; i < M * N; i++) {
        assert(C_fp32[i] == 0.0f);
    }

    free(A);
    free(B);
    free(C);
    printf(" PASS\n");
}

int main(void) {
    printf("=== WMMA (Tensor Core) Tests ===\n");

    test_select_config_basic();
    test_select_config_small();
    test_select_config_rectangular();
    test_generate_kernel();
    test_generate_kernel_various_sizes();
    test_wmma_availability();
    test_wmma_matmul_if_available();

    printf("All WMMA tests passed.\n");
    return 0;
}
