#include "ops/ir/gpu/nv_driver.h"
#include "ops/ir/gpu/nv_qmd.h"
#include "ops/ir/gpu/am_driver.h"
#include "ops/ir/gpu/amx.h"
#include "ops/ir/gpu/xmx.h"
#include "ops/ir/tc_opt.h"
#include "core/logging.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

static int test_blackwell_defines(void) {
    if (NV_GPU_ARCH_BLACKWELL != 0x1D0) return 0;
    if (QMD_VERSION_BLACKWELL != 0x05) return 0;
    if (BLACKWELL_CHANNEL_GPFIFO_A != 0x0000C96F) return 0;
    if (BLACKWELL_COMPUTE_A != 0x0000CAC0) return 0;
    if (BLACKWELL_DMA_COPY_A != 0x0000C9B5) return 0;
    return 1;
}

static int test_blackwell_arch_ordering(void) {
    if (NV_GPU_ARCH_BLACKWELL <= NV_GPU_ARCH_HOPPER) return 0;
    if (NV_GPU_ARCH_HOPPER <= NV_GPU_ARCH_ADA) return 0;
    if (NV_GPU_ARCH_ADA <= NV_GPU_ARCH_AMPERE) return 0;
    if (NV_GPU_ARCH_AMPERE <= NV_GPU_ARCH_TURING) return 0;
    return 1;
}

static int test_qmd_blackwell_version(void) {
    NVQmd qmd;
    nv_qmd_init(&qmd, NV_GPU_ARCH_BLACKWELL);
    uint32_t version_field = (qmd.data[0] >> 4) & 0xF;
    if (version_field != QMD_VERSION_BLACKWELL) return 0;
    return 1;
}

static int test_qmd_hopper_version(void) {
    NVQmd qmd;
    nv_qmd_init(&qmd, NV_GPU_ARCH_HOPPER);
    uint32_t version_field = (qmd.data[0] >> 4) & 0xF;
    if (version_field != QMD_VERSION_HOPPER) return 0;
    return 1;
}

static int test_qmd_turing_version(void) {
    NVQmd qmd;
    nv_qmd_init(&qmd, NV_GPU_ARCH_TURING);
    uint32_t version_field = (qmd.data[0] >> 4) & 0xF;
    if (version_field != QMD_VERSION_TURING) return 0;
    return 1;
}

static int test_qmd_blackwell_grid_block(void) {
    NVQmd qmd;
    nv_qmd_init(&qmd, NV_GPU_ARCH_BLACKWELL);
    nv_qmd_set_grid_dim(&qmd, 128, 64, 32);
    nv_qmd_set_block_dim(&qmd, 256, 2, 1);

    if (qmd.data[12] != 128) return 0;
    if (qmd.data[13] != 64) return 0;
    if (qmd.data[14] != 32) return 0;
    if ((qmd.data[15] & 0xFFFF) != 256) return 0;
    if ((qmd.data[16] & 0xFFFF) != 2) return 0;
    if ((qmd.data[17] & 0xFFFF) != 1) return 0;
    return 1;
}

static int test_qmd_blackwell_program_address(void) {
    NVQmd qmd;
    nv_qmd_init(&qmd, NV_GPU_ARCH_BLACKWELL);
    uint64_t addr = 0xDEADBEEF12345678ULL;
    nv_qmd_set_program_address(&qmd, addr);
    if (qmd.data[8] != 0x12345678) return 0;
    if (qmd.data[9] != 0xDEADBEEF) return 0;
    return 1;
}

static int test_am_chiplet_config_mi300x(void) {
    CMLAMChipletConfig cfg = cml_am_get_chiplet_config("gfx942");
    if (cfg.num_xcd != 8) return 0;
    if (cfg.cu_per_xcd != 38) return 0;
    if (cfg.sdma_per_xcd != 1) return 0;
    if (!cfg.unified_memory) return 0;
    return 1;
}

static int test_am_chiplet_config_mi350(void) {
    CMLAMChipletConfig cfg = cml_am_get_chiplet_config("gfx950");
    if (cfg.num_xcd != 8) return 0;
    if (cfg.cu_per_xcd != 48) return 0;
    if (cfg.sdma_per_xcd != 2) return 0;
    if (!cfg.unified_memory) return 0;
    return 1;
}

static int test_am_chiplet_config_rdna4(void) {
    CMLAMChipletConfig cfg = cml_am_get_chiplet_config("gfx1200");
    if (cfg.num_xcd != 1) return 0;
    if (cfg.unified_memory) return 0;
    return 1;
}

static int test_am_chiplet_config_unknown(void) {
    CMLAMChipletConfig cfg = cml_am_get_chiplet_config("gfx803");
    if (cfg.num_xcd != 0) return 0;
    return 1;
}

static int test_am_chiplet_config_null(void) {
    CMLAMChipletConfig cfg = cml_am_get_chiplet_config(NULL);
    if (cfg.num_xcd != 0) return 0;
    return 1;
}

static int test_am_defines(void) {
    if (AM_GFX_MI300X != 0x120100) return 0;
    if (AM_GFX_MI350  != 0x120200) return 0;
    if (AM_GFX_RDNA4  != 0x120300) return 0;
    return 1;
}

static int test_amx_availability(void) {
    bool avail = cml_amx_available();
#if defined(__APPLE__) && defined(__aarch64__)
    if (!avail) return 0;
#else
    if (avail) return 0;
#endif
    return 1;
}

static int test_amx_null_args(void) {
    int ret = cml_amx_matmul_f32(NULL, NULL, NULL, 0, 0, 0, 0, 0, 0);
    if (ret != -1) return 0;
    return 1;
}

static int test_amx_f16_stub(void) {
    int ret = cml_amx_matmul_f16(NULL, NULL, NULL, 0, 0, 0, 0, 0, 0);
    if (ret != -1) return 0;
    return 1;
}

static int test_xmx_config(void) {
    CMLXMXConfig cfg = cml_xmx_get_config();
    if (cfg.dpas_depth != 8) return 0;
    if (cfg.exec_size != 16) return 0;
    if (cfg.ops_per_chan != 8) return 0;
    return 1;
}

static int test_tc_available(void) {
    cml_tc_available();
    return 1;
}

static int test_tc_config_roundtrip(void) {
    CMLTCConfig original = cml_tc_get_config();
    CMLTCConfig custom = {
        .min_m = 32, .min_n = 32, .min_k = 32,
        .allow_padding = false, .prefer_fp16 = true
    };
    cml_tc_set_config(&custom);
    CMLTCConfig got = cml_tc_get_config();
    if (got.min_m != 32 || got.min_n != 32 || got.min_k != 32) return 0;
    if (got.allow_padding != false || got.prefer_fp16 != true) return 0;
    cml_tc_set_config(&original);
    return 1;
}

int main(void) {
    printf("=== Hardware Support Tests ===\n\n");

    printf("[NVIDIA Blackwell]\n");
    TEST(blackwell_defines);
    TEST(blackwell_arch_ordering);
    TEST(qmd_blackwell_version);
    TEST(qmd_hopper_version);
    TEST(qmd_turing_version);
    TEST(qmd_blackwell_grid_block);
    TEST(qmd_blackwell_program_address);

    printf("\n[AMD MI300/MI350]\n");
    TEST(am_chiplet_config_mi300x);
    TEST(am_chiplet_config_mi350);
    TEST(am_chiplet_config_rdna4);
    TEST(am_chiplet_config_unknown);
    TEST(am_chiplet_config_null);
    TEST(am_defines);

    printf("\n[Apple AMX]\n");
    TEST(amx_availability);
    TEST(amx_null_args);
    TEST(amx_f16_stub);

    printf("\n[Intel XMX]\n");
    TEST(xmx_config);

    printf("\n[Tensor Core Optimization]\n");
    TEST(tc_available);
    TEST(tc_config_roundtrip);

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
