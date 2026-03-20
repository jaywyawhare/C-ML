#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>

#include "ops/ir/process_replay.h"

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    printf("  Testing: %s... ", #name); \
    tests_run++; \
    if (test_##name()) { \
        printf("PASS\n"); \
        tests_passed++; \
    } else { \
        printf("FAIL\n"); \
    } \
} while(0)

static char g_tmpdir[256];

static void make_tmpdir(const char* suffix) {
    snprintf(g_tmpdir, sizeof(g_tmpdir), "/tmp/cml_pr_test_%s_XXXXXX", suffix);
    if (!mkdtemp(g_tmpdir)) {
        fprintf(stderr, "Failed to create temp dir\n");
        exit(1);
    }
}

static void rmdir_recursive(const char* path) {
    DIR* d = opendir(path);
    if (!d) return;

    struct dirent* ent;
    while ((ent = readdir(d)) != NULL) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
            continue;
        char full[4224];
        snprintf(full, sizeof(full), "%s/%s", path, ent->d_name);
        struct stat st;
        if (stat(full, &st) == 0 && S_ISDIR(st.st_mode)) {
            rmdir_recursive(full);
        } else {
            unlink(full);
        }
    }
    closedir(d);
    rmdir(path);
}

static int count_kernel_files(const char* dir) {
    DIR* d = opendir(dir);
    if (!d) return 0;

    int count = 0;
    struct dirent* ent;
    while ((ent = readdir(d)) != NULL) {
        size_t len = strlen(ent->d_name);
        if (len > 7 && strcmp(ent->d_name + len - 7, ".kernel") == 0)
            count++;
    }
    closedir(d);
    return count;
}

static int test_record_and_count(void) {
    make_tmpdir("rec");
    cml_process_replay_enable(g_tmpdir);

    cml_process_replay_record("matmul_f32", "kernel void matmul() {}", 23);
    cml_process_replay_record("relu_f32", "kernel void relu() {}", 22);
    cml_process_replay_record("add_f32", "kernel void add() {}", 21);

    cml_process_replay_disable();

    int n = count_kernel_files(g_tmpdir);
    rmdir_recursive(g_tmpdir);
    return n == 3;
}

static int test_identical_source_same_file(void) {
    make_tmpdir("dup");
    cml_process_replay_enable(g_tmpdir);

    cml_process_replay_record("relu_v1", "kernel void relu() {}", 22);
    cml_process_replay_record("relu_v2", "kernel void relu() {}", 22);

    cml_process_replay_disable();

    int n = count_kernel_files(g_tmpdir);
    rmdir_recursive(g_tmpdir);
    return n == 1;
}

static int test_compare_match(void) {
    char dir_a[256], dir_b[256];

    snprintf(dir_a, sizeof(dir_a), "/tmp/cml_pr_cmp_a_XXXXXX");
    snprintf(dir_b, sizeof(dir_b), "/tmp/cml_pr_cmp_b_XXXXXX");
    if (!mkdtemp(dir_a) || !mkdtemp(dir_b)) return 0;

    cml_process_replay_enable(dir_a);
    cml_process_replay_record("k1", "source_a", 8);
    cml_process_replay_record("k2", "source_b", 8);
    cml_process_replay_disable();

    cml_process_replay_enable(dir_b);
    cml_process_replay_record("k1", "source_a", 8);
    cml_process_replay_record("k2", "source_b", 8);
    cml_process_replay_disable();

    int result = cml_process_replay_compare(dir_b, dir_a);

    rmdir_recursive(dir_a);
    rmdir_recursive(dir_b);
    return result == 0;
}

static int test_compare_mismatch(void) {
    char dir_a[256], dir_b[256];

    snprintf(dir_a, sizeof(dir_a), "/tmp/cml_pr_mis_a_XXXXXX");
    snprintf(dir_b, sizeof(dir_b), "/tmp/cml_pr_mis_b_XXXXXX");
    if (!mkdtemp(dir_a) || !mkdtemp(dir_b)) return 0;

    cml_process_replay_enable(dir_a);
    cml_process_replay_record("k1", "source_a", 8);
    cml_process_replay_disable();

    cml_process_replay_enable(dir_b);
    cml_process_replay_record("k1", "source_X", 8);
    cml_process_replay_disable();

    int result = cml_process_replay_compare(dir_b, dir_a);

    rmdir_recursive(dir_a);
    rmdir_recursive(dir_b);
    return result > 0;
}

static int test_compare_missing_output(void) {
    char dir_a[256], dir_b[256];

    snprintf(dir_a, sizeof(dir_a), "/tmp/cml_pr_mso_a_XXXXXX");
    snprintf(dir_b, sizeof(dir_b), "/tmp/cml_pr_mso_b_XXXXXX");
    if (!mkdtemp(dir_a) || !mkdtemp(dir_b)) return 0;

    cml_process_replay_enable(dir_a);
    cml_process_replay_record("k1", "source_a", 8);
    cml_process_replay_record("k2", "source_b", 8);
    cml_process_replay_disable();

    cml_process_replay_enable(dir_b);
    cml_process_replay_record("k1", "source_a", 8);
    cml_process_replay_disable();

    int result = cml_process_replay_compare(dir_b, dir_a);

    rmdir_recursive(dir_a);
    rmdir_recursive(dir_b);
    return result > 0;
}

static int test_enable_disable(void) {
    make_tmpdir("endis");
    cml_process_replay_enable(g_tmpdir);
    cml_process_replay_record("k1", "aaa", 3);
    cml_process_replay_disable();
    cml_process_replay_record("k2", "bbb", 3);

    int n = count_kernel_files(g_tmpdir);
    rmdir_recursive(g_tmpdir);
    return n == 1;
}

int main(void) {
    printf("=== Process Replay Tests ===\n");

    TEST(record_and_count);
    TEST(identical_source_same_file);
    TEST(compare_match);
    TEST(compare_mismatch);
    TEST(compare_missing_output);
    TEST(enable_disable);

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
