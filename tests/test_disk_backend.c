#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "backend/disk_backend.h"
#include "tensor/tensor.h"

static int tests_run = 0;
static int tests_passed = 0;

#define RUN_TEST(test) do { \
    tests_run++; \
    printf("  [%d] %-50s ", tests_run, #test); \
    if (test()) { tests_passed++; printf("PASS\n"); } \
    else { printf("FAIL\n"); } \
} while(0)

static const char* TEST_DIR = "/tmp/cml_test_disk";

static void setup_test_dir(void) {
    mkdir(TEST_DIR, 0755);
}

static int test_create_destroy(void) {
    CMLDiskBackend* b = cml_disk_backend_create(TEST_DIR, CML_DISK_SYNC);
    if (!b) return 0;
    cml_disk_backend_free(b);
    return 1;
}

static int test_save_load_tensor(void) {
    setup_test_dir();
    CMLDiskBackend* b = cml_disk_backend_create(TEST_DIR, CML_DISK_SYNC);
    if (!b) return 0;

    int shape[] = {3, 4};
    TensorConfig tc = {0};
    Tensor* t = tensor_empty(shape, 2, &tc);
    if (!t) { cml_disk_backend_free(b); return 0; }

    /* Fill with known data */
    for (size_t i = 0; i < t->numel; i++)
        ((float*)t->data)[i] = (float)i * 0.5f;

    int ret = cml_disk_save_tensor(b, "test_tensor", t);
    if (ret != 0) { tensor_free(t); cml_disk_backend_free(b); return 0; }

    Tensor* loaded = cml_disk_load_tensor(b, "test_tensor");
    if (!loaded) { tensor_free(t); cml_disk_backend_free(b); return 0; }

    /* Verify data */
    int ok = (loaded->numel == t->numel);
    for (size_t i = 0; i < t->numel && ok; i++) {
        if (((float*)loaded->data)[i] != ((float*)t->data)[i]) ok = 0;
    }

    tensor_free(loaded);
    tensor_free(t);
    cml_disk_backend_free(b);
    return ok;
}

static int test_load_nonexistent(void) {
    CMLDiskBackend* b = cml_disk_backend_create(TEST_DIR, CML_DISK_SYNC);
    if (!b) return 0;
    Tensor* t = cml_disk_load_tensor(b, "nonexistent_xyz");
    int ok = (t == NULL);
    cml_disk_backend_free(b);
    return ok;
}

static int test_stats(void) {
    setup_test_dir();
    CMLDiskBackend* b = cml_disk_backend_create(TEST_DIR, CML_DISK_SYNC);
    if (!b) return 0;

    int shape[] = {2};
    TensorConfig tc = {0};
    Tensor* t = tensor_empty(shape, 1, &tc);
    if (!t) { cml_disk_backend_free(b); return 0; }

    cml_disk_save_tensor(b, "stats_test", t);
    cml_disk_load_tensor(b, "stats_test");

    uint64_t rd, wr, nr, nw;
    cml_disk_backend_stats(b, &rd, &wr, &nr, &nw);
    int ok = (nw >= 1) && (nr >= 1) && (wr > 0) && (rd > 0);

    tensor_free(t);
    cml_disk_backend_free(b);
    return ok;
}

static int test_mmap_tensor(void) {
    setup_test_dir();
    CMLDiskBackend* b = cml_disk_backend_create(TEST_DIR, CML_DISK_MMAP);
    if (!b) return 0;

    int shape[] = {4};
    TensorConfig tc = {0};
    Tensor* t = tensor_empty(shape, 1, &tc);
    if (!t) { cml_disk_backend_free(b); return 0; }
    for (size_t i = 0; i < t->numel; i++)
        ((float*)t->data)[i] = (float)i;

    cml_disk_save_tensor(b, "mmap_test", t);

    CMLDiskTensor* dt = cml_disk_mmap_tensor(b, "mmap_test");
    int ok = (dt != NULL);

    if (ok && dt->is_mapped) {
        float buf[4];
        int ret = cml_disk_tensor_read(dt, buf, 0, sizeof(buf));
        ok = (ret == 0);
        for (int i = 0; i < 4 && ok; i++) {
            if (buf[i] != (float)i) ok = 0;
        }
    }

    if (dt) cml_disk_tensor_free(dt);
    tensor_free(t);
    cml_disk_backend_free(b);
    return ok;
}

static int test_disk_tensor_to_tensor(void) {
    setup_test_dir();
    CMLDiskBackend* b = cml_disk_backend_create(TEST_DIR, CML_DISK_SYNC);
    if (!b) return 0;

    int shape[] = {3};
    TensorConfig tc = {0};
    Tensor* t = tensor_empty(shape, 1, &tc);
    if (!t) { cml_disk_backend_free(b); return 0; }
    ((float*)t->data)[0] = 1.0f; ((float*)t->data)[1] = 2.0f; ((float*)t->data)[2] = 3.0f;

    cml_disk_save_tensor(b, "convert_test", t);

    CMLDiskTensor* dt = cml_disk_mmap_tensor(b, "convert_test");
    int ok = (dt != NULL);

    if (ok) {
        Tensor* converted = cml_disk_tensor_to_tensor(dt);
        ok = (converted != NULL);
        if (ok) {
            ok = (((float*)converted->data)[0] == 1.0f && ((float*)converted->data)[2] == 3.0f);
            tensor_free(converted);
        }
        cml_disk_tensor_free(dt);
    }

    tensor_free(t);
    cml_disk_backend_free(b);
    return ok;
}

static int test_print_no_crash(void) {
    CMLDiskBackend* b = cml_disk_backend_create(TEST_DIR, CML_DISK_SYNC);
    cml_disk_backend_print(b);
    cml_disk_backend_print(NULL);
    cml_disk_backend_free(b);
    return 1;
}

static int test_free_null(void) {
    cml_disk_backend_free(NULL);
    cml_disk_tensor_free(NULL);
    return 1;
}

int main(void) {
    printf("Disk Backend Tests\n");
    setup_test_dir();

    RUN_TEST(test_create_destroy);
    RUN_TEST(test_save_load_tensor);
    RUN_TEST(test_load_nonexistent);
    RUN_TEST(test_stats);
    RUN_TEST(test_mmap_tensor);
    RUN_TEST(test_disk_tensor_to_tensor);
    RUN_TEST(test_print_no_crash);
    RUN_TEST(test_free_null);

    printf("\nResults: %d/%d passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
