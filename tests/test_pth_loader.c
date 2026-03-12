/**
 * @file test_pth_loader.c
 * @brief Tests for PyTorch .pth file loader
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "core/pth_loader.h"

static int tests_run = 0;
static int tests_passed = 0;

#define RUN_TEST(test) do { \
    tests_run++; \
    printf("  [%d] %-50s ", tests_run, #test); \
    if (test()) { tests_passed++; printf("PASS\n"); } \
    else { printf("FAIL\n"); } \
} while(0)

static int test_load_nonexistent(void) {
    CMLPthStateDict* sd = cml_pth_load("/nonexistent/path.pth");
    return sd == NULL;
}

static int test_free_null(void) {
    cml_pth_free(NULL);
    return 1;
}

static int test_get_tensor_null(void) {
    return cml_pth_get_tensor(NULL, "test") == NULL;
}

static int test_num_entries_null(void) {
    return cml_pth_num_entries(NULL) == 0;
}

static int test_get_key_null(void) {
    return cml_pth_get_key(NULL, 0) == NULL;
}

static int test_has_key_null(void) {
    return !cml_pth_has_key(NULL, "test");
}

static int test_total_params_null(void) {
    return cml_pth_total_params(NULL) == 0;
}

static int test_total_bytes_null(void) {
    return cml_pth_total_bytes(NULL) == 0;
}

static int test_print_null(void) {
    cml_pth_print(NULL);
    return 1;
}

static int test_list_keys_null(void) {
    int count;
    const char** keys = cml_pth_list_keys(NULL, &count);
    return keys == NULL;
}

int main(void) {
    printf("=== PTH Loader Tests ===\n");

    RUN_TEST(test_load_nonexistent);
    RUN_TEST(test_free_null);
    RUN_TEST(test_get_tensor_null);
    RUN_TEST(test_num_entries_null);
    RUN_TEST(test_get_key_null);
    RUN_TEST(test_has_key_null);
    RUN_TEST(test_total_params_null);
    RUN_TEST(test_total_bytes_null);
    RUN_TEST(test_print_null);
    RUN_TEST(test_list_keys_null);

    printf("\nResults: %d/%d passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
