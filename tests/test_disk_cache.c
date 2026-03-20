#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>

#include "ops/ir/disk_cache.h"

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

static char tmp_path[256];

static void make_tmp_path(void) {
    snprintf(tmp_path, sizeof(tmp_path), "/tmp/cml_test_disk_cache_%d.db", getpid());
}

static int test_open_close(void) {
    CMLDiskCache* cache = cml_disk_cache_open(tmp_path);
    if (!cache) {
        printf("(sqlite3 not available, skipping) ");
        return 1;
    }
    cml_disk_cache_close(cache);
    unlink(tmp_path);
    return 1;
}

static int test_put_get(void) {
    CMLDiskCache* cache = cml_disk_cache_open(tmp_path);
    if (!cache) return 1;

    uint8_t data[] = {0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE};
    uint64_t hash = 0x123456789ABCDEF0ULL;

    if (cml_disk_cache_put(cache, hash, data, sizeof(data)) != 0) {
        cml_disk_cache_close(cache);
        unlink(tmp_path);
        return 0;
    }

    void* out = NULL;
    size_t out_size = 0;
    if (cml_disk_cache_get(cache, hash, &out, &out_size) != 0) {
        cml_disk_cache_close(cache);
        unlink(tmp_path);
        return 0;
    }

    int ok = (out_size == sizeof(data)) && (memcmp(out, data, sizeof(data)) == 0);
    free(out);
    cml_disk_cache_close(cache);
    unlink(tmp_path);
    return ok;
}

static int test_has(void) {
    CMLDiskCache* cache = cml_disk_cache_open(tmp_path);
    if (!cache) return 1;

    uint8_t data[] = {1, 2, 3};
    cml_disk_cache_put(cache, 42, data, sizeof(data));

    int ok = cml_disk_cache_has(cache, 42) && !cml_disk_cache_has(cache, 99);
    cml_disk_cache_close(cache);
    unlink(tmp_path);
    return ok;
}

static int test_count(void) {
    CMLDiskCache* cache = cml_disk_cache_open(tmp_path);
    if (!cache) return 1;

    uint8_t data[] = {0xFF};
    cml_disk_cache_put(cache, 1, data, 1);
    cml_disk_cache_put(cache, 2, data, 1);
    cml_disk_cache_put(cache, 3, data, 1);

    int ok = (cml_disk_cache_count(cache) == 3);
    cml_disk_cache_close(cache);
    unlink(tmp_path);
    return ok;
}

static int test_clear(void) {
    CMLDiskCache* cache = cml_disk_cache_open(tmp_path);
    if (!cache) return 1;

    uint8_t data[] = {0xAA};
    cml_disk_cache_put(cache, 10, data, 1);
    cml_disk_cache_put(cache, 20, data, 1);

    cml_disk_cache_clear(cache);
    int ok = (cml_disk_cache_count(cache) == 0) && !cml_disk_cache_has(cache, 10);
    cml_disk_cache_close(cache);
    unlink(tmp_path);
    return ok;
}

static int test_overwrite(void) {
    CMLDiskCache* cache = cml_disk_cache_open(tmp_path);
    if (!cache) return 1;

    uint8_t data1[] = {1, 2, 3};
    uint8_t data2[] = {4, 5, 6, 7, 8};

    cml_disk_cache_put(cache, 100, data1, sizeof(data1));
    cml_disk_cache_put(cache, 100, data2, sizeof(data2));

    void* out = NULL;
    size_t out_size = 0;
    cml_disk_cache_get(cache, 100, &out, &out_size);

    int ok = (out_size == sizeof(data2)) && (memcmp(out, data2, sizeof(data2)) == 0);
    free(out);
    cml_disk_cache_close(cache);
    unlink(tmp_path);
    return ok;
}

static int test_persistence(void) {
    CMLDiskCache* cache = cml_disk_cache_open(tmp_path);
    if (!cache) return 1;

    uint8_t data[] = {0xBE, 0xEF};
    cml_disk_cache_put(cache, 555, data, sizeof(data));
    cml_disk_cache_close(cache);

    cache = cml_disk_cache_open(tmp_path);
    if (!cache) return 0;

    void* out = NULL;
    size_t out_size = 0;
    int ok = (cml_disk_cache_get(cache, 555, &out, &out_size) == 0) &&
             (out_size == sizeof(data)) &&
             (memcmp(out, data, sizeof(data)) == 0);
    free(out);
    cml_disk_cache_close(cache);
    unlink(tmp_path);
    return ok;
}

static int test_get_missing(void) {
    CMLDiskCache* cache = cml_disk_cache_open(tmp_path);
    if (!cache) return 1;

    void* out = NULL;
    size_t out_size = 0;
    int ok = (cml_disk_cache_get(cache, 999, &out, &out_size) != 0) && (out == NULL);
    cml_disk_cache_close(cache);
    unlink(tmp_path);
    return ok;
}

static int test_large_blob(void) {
    CMLDiskCache* cache = cml_disk_cache_open(tmp_path);
    if (!cache) return 1;

    size_t size = 1024 * 1024;
    uint8_t* data = malloc(size);
    for (size_t i = 0; i < size; i++)
        data[i] = (uint8_t)(i & 0xFF);

    cml_disk_cache_put(cache, 7777, data, size);

    void* out = NULL;
    size_t out_size = 0;
    cml_disk_cache_get(cache, 7777, &out, &out_size);

    int ok = (out_size == size) && (memcmp(out, data, size) == 0);
    free(out);
    free(data);
    cml_disk_cache_close(cache);
    unlink(tmp_path);
    return ok;
}

static int test_enabled_env(void) {
    int ok = !cml_disk_cache_enabled();
    return ok;
}

int main(void) {
    printf("=== Disk Cache Tests ===\n");
    make_tmp_path();

    TEST(open_close);
    TEST(put_get);
    TEST(has);
    TEST(count);
    TEST(clear);
    TEST(overwrite);
    TEST(persistence);
    TEST(get_missing);
    TEST(large_blob);
    TEST(enabled_env);

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
