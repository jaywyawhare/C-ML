#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cml.h"
#include "tensor/shard.h"

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

#define APPROX_EQ(a, b) (fabsf((a) - (b)) < 1e-5f)

static int test_even_shard(void) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    int shape[] = {4};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    Tensor* t = tensor_from_data(data, shape, 1, &cfg);
    if (!t) return 0;

    DeviceType devs[] = {DEVICE_CPU, DEVICE_CPU};
    CMLShardedTensor* st = tensor_shard(t, devs, 2, 0);
    if (!st) { tensor_free(t); return 0; }

    int ok = (st->num_shards == 2);
    ok = ok && (st->shards[0]->shape[0] == 2);
    ok = ok && (st->shards[1]->shape[0] == 2);

    tensor_ensure_executed(st->shards[0]);
    tensor_ensure_executed(st->shards[1]);
    float* d0 = (float*)tensor_data_ptr(st->shards[0]);
    float* d1 = (float*)tensor_data_ptr(st->shards[1]);
    ok = ok && d0 && d1;
    ok = ok && APPROX_EQ(d0[0], 1.0f) && APPROX_EQ(d0[1], 2.0f);
    ok = ok && APPROX_EQ(d1[0], 3.0f) && APPROX_EQ(d1[1], 4.0f);

    sharded_tensor_free(st);
    tensor_free(t);
    return ok;
}

static int test_uneven_shard(void) {
    float data[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
    int shape[] = {5};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    Tensor* t = tensor_from_data(data, shape, 1, &cfg);
    if (!t) return 0;

    DeviceType devs[] = {DEVICE_CPU, DEVICE_CPU};
    CMLShardedTensor* st = tensor_shard(t, devs, 2, 0);
    if (!st) { tensor_free(t); return 0; }

    int ok = (st->shards[0]->shape[0] == 3);
    ok = ok && (st->shards[1]->shape[0] == 2);

    tensor_ensure_executed(st->shards[0]);
    tensor_ensure_executed(st->shards[1]);
    float* d0 = (float*)tensor_data_ptr(st->shards[0]);
    float* d1 = (float*)tensor_data_ptr(st->shards[1]);
    ok = ok && d0 && d1;
    ok = ok && APPROX_EQ(d0[0], 10.0f) && APPROX_EQ(d0[1], 20.0f) && APPROX_EQ(d0[2], 30.0f);
    ok = ok && APPROX_EQ(d1[0], 40.0f) && APPROX_EQ(d1[1], 50.0f);

    sharded_tensor_free(st);
    tensor_free(t);
    return ok;
}

static int test_shard_unshard_roundtrip(void) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    int shape[] = {2, 3};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    Tensor* t = tensor_from_data(data, shape, 2, &cfg);
    if (!t) return 0;

    DeviceType devs[] = {DEVICE_CPU, DEVICE_CPU};
    CMLShardedTensor* st = tensor_shard(t, devs, 2, 1);
    if (!st) { tensor_free(t); return 0; }

    Tensor* restored = tensor_unshard(st);
    if (!restored) { sharded_tensor_free(st); tensor_free(t); return 0; }

    tensor_ensure_executed(restored);
    float* rd = (float*)tensor_data_ptr(restored);
    int ok = rd != NULL;
    ok = ok && (restored->ndim == 2);
    ok = ok && (restored->shape[0] == 2) && (restored->shape[1] == 3);

    for (int i = 0; i < 6 && ok; i++) {
        ok = ok && APPROX_EQ(rd[i], data[i]);
    }

    tensor_free(restored);
    sharded_tensor_free(st);
    tensor_free(t);
    return ok;
}

static int test_shard_unshard_roundtrip_axis0(void) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    int shape[] = {3, 3};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    Tensor* t = tensor_from_data(data, shape, 2, &cfg);
    if (!t) return 0;

    DeviceType devs[] = {DEVICE_CPU, DEVICE_CPU, DEVICE_CPU};
    CMLShardedTensor* st = tensor_shard(t, devs, 3, 0);
    if (!st) { tensor_free(t); return 0; }

    Tensor* restored = tensor_unshard(st);
    if (!restored) { sharded_tensor_free(st); tensor_free(t); return 0; }

    tensor_ensure_executed(restored);
    float* rd = (float*)tensor_data_ptr(restored);
    int ok = rd != NULL;

    for (int i = 0; i < 9 && ok; i++) {
        ok = ok && APPROX_EQ(rd[i], data[i]);
    }

    tensor_free(restored);
    sharded_tensor_free(st);
    tensor_free(t);
    return ok;
}

static int test_sharded_add(void) {
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {10.0f, 20.0f, 30.0f, 40.0f};
    int shape[] = {4};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    Tensor* a = tensor_from_data(a_data, shape, 1, &cfg);
    Tensor* b = tensor_from_data(b_data, shape, 1, &cfg);
    if (!a || !b) return 0;

    DeviceType devs[] = {DEVICE_CPU, DEVICE_CPU};
    CMLShardedTensor* sa = tensor_shard(a, devs, 2, 0);
    CMLShardedTensor* sb = tensor_shard(b, devs, 2, 0);
    if (!sa || !sb) {
        if (sa) sharded_tensor_free(sa);
        if (sb) sharded_tensor_free(sb);
        tensor_free(a); tensor_free(b);
        return 0;
    }

    CMLShardedTensor* sc = sharded_add(sa, sb);
    if (!sc) {
        sharded_tensor_free(sa); sharded_tensor_free(sb);
        tensor_free(a); tensor_free(b);
        return 0;
    }

    Tensor* result = tensor_unshard(sc);
    if (!result) {
        sharded_tensor_free(sc); sharded_tensor_free(sa); sharded_tensor_free(sb);
        tensor_free(a); tensor_free(b);
        return 0;
    }

    tensor_ensure_executed(result);
    float* rd = (float*)tensor_data_ptr(result);
    int ok = rd != NULL;
    ok = ok && APPROX_EQ(rd[0], 11.0f) && APPROX_EQ(rd[1], 22.0f);
    ok = ok && APPROX_EQ(rd[2], 33.0f) && APPROX_EQ(rd[3], 44.0f);

    tensor_free(result);
    sharded_tensor_free(sc);
    sharded_tensor_free(sa);
    sharded_tensor_free(sb);
    tensor_free(a);
    tensor_free(b);
    return ok;
}

static int test_replicate(void) {
    float data[] = {1.0f, 2.0f, 3.0f};
    int shape[] = {3};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    Tensor* t = tensor_from_data(data, shape, 1, &cfg);
    if (!t) return 0;

    DeviceType devs[] = {DEVICE_CPU, DEVICE_CPU, DEVICE_CPU};
    CMLShardedTensor* st = tensor_replicate(t, devs, 3);
    if (!st) { tensor_free(t); return 0; }

    int ok = (st->num_shards == 3);
    for (int s = 0; s < 3 && ok; s++) {
        tensor_ensure_executed(st->shards[s]);
        float* sd = (float*)tensor_data_ptr(st->shards[s]);
        ok = ok && sd;
        ok = ok && (st->shards[s]->shape[0] == 3);
        ok = ok && APPROX_EQ(sd[0], 1.0f) && APPROX_EQ(sd[1], 2.0f) && APPROX_EQ(sd[2], 3.0f);
    }

    sharded_tensor_free(st);
    tensor_free(t);
    return ok;
}

int main(void) {
    cml_init();

    printf("\nTensor Sharding Tests\n\n");

    TEST(even_shard);
    TEST(uneven_shard);
    TEST(shard_unshard_roundtrip);
    TEST(shard_unshard_roundtrip_axis0);
    TEST(sharded_add);
    TEST(replicate);

    printf("\nResults: %d/%d passed\n\n", tests_passed, tests_run);

    cml_cleanup();
    return tests_passed == tests_run ? 0 : 1;
}
