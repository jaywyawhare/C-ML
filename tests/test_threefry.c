#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "core/threefry.h"

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

static int test_deterministic(void) {
    CMLRNGState s1, s2;
    cml_rng_init(&s1, 42);
    cml_rng_init(&s2, 42);

    float a[100], b[100];
    cml_rng_uniform(&s1, a, 100);
    cml_rng_uniform(&s2, b, 100);
    return memcmp(a, b, sizeof(a)) == 0;
}

static int test_different_seeds(void) {
    CMLRNGState s1, s2;
    cml_rng_init(&s1, 42);
    cml_rng_init(&s2, 43);

    float a[100], b[100];
    cml_rng_uniform(&s1, a, 100);
    cml_rng_uniform(&s2, b, 100);
    return memcmp(a, b, sizeof(a)) != 0;
}

static int test_uniform_range(void) {
    CMLRNGState state;
    cml_rng_init(&state, 123);

    float buf[10000];
    cml_rng_uniform(&state, buf, 10000);

    for (int i = 0; i < 10000; i++) {
        if (buf[i] < 0.0f || buf[i] >= 1.0f) return 0;
    }
    return 1;
}

static int test_uniform_mean(void) {
    CMLRNGState state;
    cml_rng_init(&state, 7);

    float buf[50000];
    cml_rng_uniform(&state, buf, 50000);

    double sum = 0;
    for (int i = 0; i < 50000; i++) sum += buf[i];
    double mean = sum / 50000.0;
    return fabs(mean - 0.5) < 0.01;
}

static int test_normal_moments(void) {
    CMLRNGState state;
    cml_rng_init(&state, 99);

    float buf[50000];
    cml_rng_normal(&state, buf, 50000);

    double sum = 0, sum2 = 0;
    for (int i = 0; i < 50000; i++) {
        sum += buf[i];
        sum2 += (double)buf[i] * buf[i];
    }
    double mean = sum / 50000.0;
    double var = sum2 / 50000.0 - mean * mean;

    return fabs(mean) < 0.02 && fabs(var - 1.0) < 0.05;
}

static int test_uint32_output(void) {
    CMLRNGState state;
    cml_rng_init(&state, 55);

    uint32_t buf[1000];
    cml_rng_uint32(&state, buf, 1000);

    int has_high = 0, has_low = 0;
    for (int i = 0; i < 1000; i++) {
        if (buf[i] > 0x80000000U) has_high = 1;
        if (buf[i] < 0x80000000U) has_low = 1;
    }
    return has_high && has_low;
}

static int test_fork_independence(void) {
    CMLRNGState parent;
    cml_rng_init(&parent, 42);

    CMLRNGState child1 = cml_rng_fork(&parent);
    CMLRNGState child2 = cml_rng_fork(&parent);

    float a[100], b[100];
    cml_rng_uniform(&child1, a, 100);
    cml_rng_uniform(&child2, b, 100);
    return memcmp(a, b, sizeof(a)) != 0;
}

static int test_global_state(void) {
    cml_rng_set_global_seed(12345);
    float a[50];
    cml_rng_uniform(cml_rng_get_global(), a, 50);

    cml_rng_set_global_seed(12345);
    float b[50];
    cml_rng_uniform(cml_rng_get_global(), b, 50);

    return memcmp(a, b, sizeof(a)) == 0;
}

static int test_single_element(void) {
    CMLRNGState state;
    cml_rng_init(&state, 1);
    float val;
    cml_rng_uniform(&state, &val, 1);
    return val >= 0.0f && val < 1.0f;
}

static int test_normal_single(void) {
    CMLRNGState state;
    cml_rng_init(&state, 1);
    float val;
    cml_rng_normal(&state, &val, 1);
    return fabsf(val) < 10.0f;
}

int main(void) {
    printf("Threefry PRNG Tests\n");

    TEST(deterministic);
    TEST(different_seeds);
    TEST(uniform_range);
    TEST(uniform_mean);
    TEST(normal_moments);
    TEST(uint32_output);
    TEST(fork_independence);
    TEST(global_state);
    TEST(single_element);
    TEST(normal_single);

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
