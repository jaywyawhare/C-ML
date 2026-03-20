#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "ops/ir/gpu/amd_profiling.h"

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    printf("  Testing: %s... ", #name); \
    tests_run++; \
    if (test_##name()) { printf("PASS\n"); tests_passed++; } \
    else { printf("FAIL\n"); } \
} while(0)

static bool test_profile_create_free(void) {
    CMLAMDProfile* prof = cml_amd_profile_create();
    if (!prof) return false;
    if (prof->capacity <= 0) { cml_amd_profile_free(prof); return false; }
    if (prof->num_entries != 0) { cml_amd_profile_free(prof); return false; }
    if (!prof->timestamps) { cml_amd_profile_free(prof); return false; }
    if (!prof->wave_counts) { cml_amd_profile_free(prof); return false; }
    if (!prof->busy_cycles) { cml_amd_profile_free(prof); return false; }
    if (!prof->mem_reads) { cml_amd_profile_free(prof); return false; }
    if (!prof->mem_writes) { cml_amd_profile_free(prof); return false; }
    cml_amd_profile_free(prof);
    return true;
}

static bool test_profile_free_null(void) {
    cml_amd_profile_free(NULL);
    return true;
}

static bool test_profile_start_null(void) {
    if (cml_amd_profile_start(NULL, NULL) == 0) return false;
    CMLAMDProfile* prof = cml_amd_profile_create();
    if (cml_amd_profile_start(prof, NULL) == 0) { cml_amd_profile_free(prof); return false; }
    cml_amd_profile_free(prof);
    return true;
}

static bool test_profile_stop_null(void) {
    if (cml_amd_profile_stop(NULL, NULL) == 0) return false;
    return true;
}

static bool test_pmc_read_null(void) {
    uint64_t val;
    if (cml_amd_pmc_read(NULL, 0, &val) == 0) return false;
    return true;
}

static bool test_sqtt_capture_null(void) {
    if (cml_amd_sqtt_capture(NULL, 1) != NULL) return false;
    return true;
}

static bool test_sqtt_capture_invalid(void) {
    CMLAMDriver drv;
    memset(&drv, 0, sizeof(drv));
    if (cml_amd_sqtt_capture(&drv, 0) != NULL) return false;
    if (cml_amd_sqtt_capture(&drv, -1) != NULL) return false;
    return true;
}

static bool test_sqtt_free_null(void) {
    cml_amd_sqtt_free(NULL);
    return true;
}

static bool test_profile_print_empty(void) {
    cml_amd_profile_print(NULL);

    CMLAMDProfile* prof = cml_amd_profile_create();
    cml_amd_profile_print(prof);
    cml_amd_profile_free(prof);
    return true;
}

static bool test_profile_print_with_data(void) {
    CMLAMDProfile* prof = cml_amd_profile_create();
    if (!prof) return false;

    prof->timestamps[0] = 1000000000ULL;
    prof->busy_cycles[0] = 0;
    prof->mem_reads[0] = 0;
    prof->mem_writes[0] = 0;
    prof->wave_counts[0] = 0;

    prof->timestamps[1] = 1001000000ULL;
    prof->busy_cycles[1] = 500;
    prof->mem_reads[1] = 1024;
    prof->mem_writes[1] = 512;
    prof->wave_counts[1] = 64;
    prof->num_entries = 2;

    cml_amd_profile_print(prof);
    cml_amd_profile_free(prof);
    return true;
}

int main(void) {
    printf("=== AMD SQTT/PMC Profiling Tests ===\n");

    TEST(profile_create_free);
    TEST(profile_free_null);
    TEST(profile_start_null);
    TEST(profile_stop_null);
    TEST(pmc_read_null);
    TEST(sqtt_capture_null);
    TEST(sqtt_capture_invalid);
    TEST(sqtt_free_null);
    TEST(profile_print_empty);
    TEST(profile_print_with_data);

    printf("\nResults: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
