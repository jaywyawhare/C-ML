/* CIFAR binary record parsing (shared by the CIFAR-10 and CIFAR-100 loaders),
 * verified on synthetic in-memory records: label selection for 1- and 2-byte layouts,
 * [0,1] pixel normalization, the max cap, and EOF handling. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include "alloc/cml_allocator.h"
#include "datasets/datasets.h"
#include "test_harness.h"

/* Write one record: `nlabels` label bytes then 3072 pixel bytes all equal to `pix`. */
static void write_record(FILE* f, const uint8_t* labels, int nlabels, uint8_t pix) {
    fwrite(labels, 1, (size_t)nlabels, f);
    uint8_t px[3072];
    for (int i = 0; i < 3072; i++)
        px[i] = pix;
    fwrite(px, 1, 3072, f);
}

static int test_cifar10_layout(void) {
    FILE* f = tmpfile();
    if (!f)
        return 0;
    uint8_t l0[1] = {3}, l1[1] = {7}, l2[1] = {1};
    write_record(f, l0, 1, 30);
    write_record(f, l1, 1, 70);
    write_record(f, l2, 1, 255);
    rewind(f);

    float* X = cml_malloc(sizeof(float) * 3 * 3072);
    float* y = cml_malloc(sizeof(float) * 3);
    int off  = 0;
    cml_cifar_read_batch(f, 1, 0, X, y, 3, &off);
    fclose(f);

    int ok = off == 3 && y[0] == 3.0f && y[1] == 7.0f && y[2] == 1.0f;
    ok &= fabsf(X[0] - 30.0f / 255.0f) < 1e-6f;
    ok &= fabsf(X[3072] - 70.0f / 255.0f) < 1e-6f;
    ok &= fabsf(X[2 * 3072] - 1.0f) < 1e-6f; /* 255/255 */
    cml_free(X);
    cml_free(y);
    return ok;
}

static int test_cifar100_layout(void) {
    FILE* f = tmpfile();
    if (!f)
        return 0;
    uint8_t r0[2] = {5, 42}, r1[2] = {1, 99}; /* coarse, fine */
    write_record(f, r0, 2, 10);
    write_record(f, r1, 2, 20);
    rewind(f);

    float* X = cml_malloc(sizeof(float) * 2 * 3072);
    float* y = cml_malloc(sizeof(float) * 2);
    int off  = 0;
    cml_cifar_read_batch(f, 2, 1, X, y, 2, &off); /* keep the fine label */
    fclose(f);

    int ok = off == 2 && y[0] == 42.0f && y[1] == 99.0f;
    ok &= fabsf(X[0] - 10.0f / 255.0f) < 1e-6f && fabsf(X[3072] - 20.0f / 255.0f) < 1e-6f;
    cml_free(X);
    cml_free(y);
    return ok;
}

static int test_max_cap_and_eof(void) {
    FILE* f = tmpfile();
    if (!f)
        return 0;
    uint8_t l[1] = {2};
    for (int i = 0; i < 5; i++)
        write_record(f, l, 1, 2);
    rewind(f);

    float* X = cml_malloc(sizeof(float) * 10 * 3072);
    float* y = cml_malloc(sizeof(float) * 10);
    int off  = 0;
    cml_cifar_read_batch(f, 1, 0, X, y, 3, &off); /* cap at 3 though 5 records exist */
    int ok = off == 3;
    cml_cifar_read_batch(f, 1, 0, X, y, 10, &off); /* read the remaining 2, then EOF */
    ok &= off == 5;
    fclose(f);
    cml_free(X);
    cml_free(y);
    return ok;
}

int main(void) {
    printf("=== CIFAR record parsing ===\n");
    TEST(cifar10_layout);
    TEST(cifar100_layout);
    TEST(max_cap_and_eof);
    return TEST_SUMMARY();
}
