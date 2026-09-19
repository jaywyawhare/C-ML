#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "test_require.h"

#include "ops/ir/gpu/wmma.h"
#include "alloc/cml_allocator.h"

static void test_select_config_basic(void) {
    printf("  test_select_config_basic...");

    WMMAConfig config;
    memset(&config, 0, sizeof(config));

    /* Standard matmul dimensions */
    int ret = cml_wmma_select_config(256, 256, 256, &config);
    if (!cml_wmma_available()) {
        /* Stub contract: without a Tensor Core device the API must fail
         * loudly, not fake a config. */
        REQUIRE(ret == -1);
        printf(" PASS (stub contract: select_config == -1)\n");
        return;
    }
    REQUIRE(ret == 0);

    /* Config should have valid fragment dimensions */
    REQUIRE(config.M > 0);
    REQUIRE(config.N > 0);
    REQUIRE(config.K > 0);

    /* Block dimensions should be positive */
    REQUIRE(config.block_m > 0);
    REQUIRE(config.block_n > 0);
    REQUIRE(config.block_k > 0);

    printf(" PASS\n");
}

static void test_select_config_small(void) {
    printf("  test_select_config_small...");

    WMMAConfig config;
    memset(&config, 0, sizeof(config));

    /* Small matmul that fits in a single fragment */
    int ret = cml_wmma_select_config(16, 16, 16, &config);
    if (!cml_wmma_available()) {
        REQUIRE(ret == -1);
        printf(" PASS (stub contract: select_config == -1)\n");
        return;
    }
    REQUIRE(ret == 0);
    REQUIRE(config.fragment == WMMA_M16N16K16);
    REQUIRE(config.M == 16);
    REQUIRE(config.N == 16);
    REQUIRE(config.K == 16);

    printf(" PASS\n");
}

static void test_select_config_rectangular(void) {
    printf("  test_select_config_rectangular...");

    WMMAConfig config;
    memset(&config, 0, sizeof(config));

    /* Rectangular matrix dimensions */
    int ret = cml_wmma_select_config(512, 128, 256, &config);
    if (!cml_wmma_available()) {
        REQUIRE(ret == -1);
        printf(" PASS (stub contract: select_config == -1)\n");
        return;
    }
    REQUIRE(ret == 0);
    REQUIRE(config.M > 0);
    REQUIRE(config.N > 0);
    REQUIRE(config.K > 0);

    printf(" PASS\n");
}

static void test_generate_kernel(void) {
    printf("  test_generate_kernel...");

    WMMAConfig config;
    memset(&config, 0, sizeof(config));

    if (!cml_wmma_available()) {
        char* stub_src = cml_wmma_generate_kernel(&config, 256, 256, 256);
        REQUIRE(stub_src == NULL); /* stub contract: no fake kernel source */
        printf(" PASS (stub contract: generate_kernel == NULL)\n");
        return;
    }

    int ret = cml_wmma_select_config(256, 256, 256, &config);
    REQUIRE(ret == 0);

    char* kernel_src = cml_wmma_generate_kernel(&config, 256, 256, 256);
    REQUIRE(kernel_src != NULL);
    REQUIRE(strlen(kernel_src) > 0);

    /* The kernel source should contain "wmma" references */
    REQUIRE(strstr(kernel_src, "wmma") != NULL);

    printf(" (generated %zu bytes) ", strlen(kernel_src));
    cml_free(kernel_src);
    printf("PASS\n");
}

static void test_generate_kernel_various_sizes(void) {
    printf("  test_generate_kernel_various_sizes...");

    if (!cml_wmma_available()) {
        WMMAConfig stub_cfg;
        memset(&stub_cfg, 0, sizeof(stub_cfg));
        REQUIRE(cml_wmma_generate_kernel(&stub_cfg, 128, 128, 128) == NULL);
        printf(" PASS (stub contract: generate_kernel == NULL)\n");
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
        REQUIRE(ret == 0);

        char* src = cml_wmma_generate_kernel(&config, M, N, K);
        REQUIRE(src != NULL);
        REQUIRE(strstr(src, "wmma") != NULL);
        cml_free(src);
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
        /* Stub contract: matmul must refuse rather than fake success. */
        float a[4] = {0}, b[4] = {0}, c[4] = {0};
        REQUIRE(cml_wmma_matmul(a, b, c, 2, 2, 2) == -1);
        printf(" PASS (stub contract: matmul == -1)\n");
        return;
    }

    /* If WMMA is available, test a small matmul
     * A is [16, 16] fp16, B is [16, 16] fp16, C is [16, 16] fp32
     * We would need device memory, so this test is only run on capable hardware.
     */
    int M = 16, N = 16, K = 16;
    size_t fp16_size = M * K * 2; /* 2 bytes per fp16 */
    size_t fp32_size = M * N * 4; /* 4 bytes per fp32 */

    /* Allocate host buffers (simplified: using calloc for zero-init) */
    void* A = cml_calloc(1, fp16_size);
    void* B = cml_calloc(1, fp16_size);
    void* C = cml_calloc(1, fp32_size);
    REQUIRE(A != NULL && B != NULL && C != NULL);

    int ret = cml_wmma_matmul(A, B, C, M, N, K);
    /* Zero matrices multiplied should give zero result */
    REQUIRE(ret == 0);

    float* C_fp32 = (float*)C;
    for (int i = 0; i < M * N; i++) {
        REQUIRE(C_fp32[i] == 0.0f);
    }

    cml_free(A);
    cml_free(B);
    cml_free(C);
    printf(" PASS\n");
}

int main(void) {
    printf("WMMA (Tensor Core) Tests\n");

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
