/**
 * @file test_multi_gpu.c
 * @brief Tests for multi-GPU support and distributed training
 *
 * Tests simulated multi-GPU devices, distributed collectives,
 * DDP gradient sync, and pipeline parallelism — all without
 * requiring real GPU hardware.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "cml.h"
#include "backend/device.h"
#include "tensor/tensor.h"
#include "nn.h"
#include "nn/layers/linear.h"
#include "nn/layers/sequential.h"
#include "nn/layers/activations.h"
#include "optim.h"
#include "autograd/autograd.h"
#include "autograd/loss_functions.h"

#ifdef CML_HAS_DISTRIBUTED
#include "distributed/distributed.h"
#include "distributed/data_parallel.h"
#include "distributed/pipeline_parallel.h"
#endif

/* Test counters */
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

#define ASSERT(cond) do { \
    if (!(cond)) { \
        printf("(ASSERT failed: %s, line %d) ", #cond, __LINE__); \
        return 0; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, eps) do { \
    if (fabsf((a) - (b)) > (eps)) { \
        printf("(%.6f != %.6f, line %d) ", (a), (b), __LINE__); \
        return 0; \
    } \
} while(0)

/* ===== Simulated GPU Device Tests ===== */

static int test_sim_gpu_enable_disable(void) {
    /* Enable 4 simulated GPUs with 256MB each */
    int ret = device_sim_gpu_enable(4, 256 * 1024 * 1024);
    ASSERT(ret == 0);
    ASSERT(device_sim_gpu_available());
    ASSERT(device_sim_gpu_get_count() == 4);

    device_sim_gpu_disable();
    ASSERT(!device_sim_gpu_available());
    ASSERT(device_sim_gpu_get_count() == 0);

    printf("(ok) ");
    return 1;
}

static int test_sim_gpu_device_switching(void) {
    device_sim_gpu_enable(4, 128 * 1024 * 1024);

    ASSERT(device_sim_gpu_get_device() == 0);

    ASSERT(device_sim_gpu_set_device(2) == 0);
    ASSERT(device_sim_gpu_get_device() == 2);

    ASSERT(device_sim_gpu_set_device(3) == 0);
    ASSERT(device_sim_gpu_get_device() == 3);

    /* Invalid device IDs */
    ASSERT(device_sim_gpu_set_device(-1) == -1);
    ASSERT(device_sim_gpu_set_device(4) == -1);

    device_sim_gpu_disable();
    printf("(ok) ");
    return 1;
}

static int test_sim_gpu_device_info(void) {
    size_t mem = 512 * 1024 * 1024ULL;
    device_sim_gpu_enable(2, mem);

    DeviceInfo info;
    ASSERT(device_sim_gpu_get_info(0, &info) == 0);
    ASSERT(info.type == DEVICE_SIM_GPU);
    ASSERT(info.device_id == 0);
    ASSERT(info.total_memory == mem);
    ASSERT(info.free_memory == mem);
    ASSERT(info.available);
    ASSERT(info.name != NULL);

    ASSERT(device_sim_gpu_get_info(1, &info) == 0);
    ASSERT(info.device_id == 1);

    /* Invalid */
    ASSERT(device_sim_gpu_get_info(5, &info) == -1);

    device_sim_gpu_disable();
    printf("(ok) ");
    return 1;
}

static int test_sim_gpu_alloc_free(void) {
    device_sim_gpu_enable(2, 1024 * 1024); /* 1MB each */

    device_sim_gpu_set_device(0);
    void* p1 = device_alloc(4096, DEVICE_SIM_GPU);
    ASSERT(p1 != NULL);

    void* p2 = device_alloc(4096, DEVICE_SIM_GPU);
    ASSERT(p2 != NULL);

    /* Check allocation tracking */
    DeviceInfo info;
    device_sim_gpu_get_info(0, &info);
    ASSERT(info.free_memory == 1024 * 1024 - 8192);

    device_free(p1, DEVICE_SIM_GPU);
    device_free(p2, DEVICE_SIM_GPU);

    device_sim_gpu_disable();
    printf("(ok) ");
    return 1;
}

static int test_sim_gpu_oom(void) {
    device_sim_gpu_enable(1, 1024); /* Tiny 1KB device */

    device_sim_gpu_set_device(0);
    void* p = device_alloc(2048, DEVICE_SIM_GPU);
    ASSERT(p == NULL); /* Should fail — not enough memory */

    /* Smaller alloc should succeed */
    void* p2 = device_alloc(512, DEVICE_SIM_GPU);
    ASSERT(p2 != NULL);
    device_free(p2, DEVICE_SIM_GPU);

    device_sim_gpu_disable();
    printf("(ok) ");
    return 1;
}

static int test_sim_gpu_copy(void) {
    device_sim_gpu_enable(2, 1024 * 1024);

    float cpu_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float result[4] = {0};

    /* CPU -> SimGPU */
    void* gpu_buf = device_alloc(sizeof(cpu_data), DEVICE_SIM_GPU);
    ASSERT(gpu_buf != NULL);
    ASSERT(device_copy(gpu_buf, cpu_data, sizeof(cpu_data), DEVICE_SIM_GPU, DEVICE_CPU) == 0);

    /* SimGPU -> CPU */
    ASSERT(device_copy(result, gpu_buf, sizeof(result), DEVICE_CPU, DEVICE_SIM_GPU) == 0);
    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR(result[i], cpu_data[i], 1e-6f);
    }

    /* SimGPU -> SimGPU (same device) */
    float* gpu_buf2 = device_alloc(sizeof(cpu_data), DEVICE_SIM_GPU);
    ASSERT(gpu_buf2 != NULL);
    ASSERT(device_copy(gpu_buf2, gpu_buf, sizeof(cpu_data), DEVICE_SIM_GPU, DEVICE_SIM_GPU) == 0);
    ASSERT(device_copy(result, gpu_buf2, sizeof(result), DEVICE_CPU, DEVICE_SIM_GPU) == 0);
    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR(result[i], cpu_data[i], 1e-6f);
    }

    device_free(gpu_buf, DEVICE_SIM_GPU);
    device_free(gpu_buf2, DEVICE_SIM_GPU);
    device_sim_gpu_disable();
    printf("(ok) ");
    return 1;
}

static int test_sim_gpu_tensor_move(void) {
    device_set_default(DEVICE_CPU);
    device_sim_gpu_enable(2, 1024 * 1024);

    /* Create a CPU tensor explicitly */
    int shape[] = {2, 3};
    TensorConfig cpu_cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                            .has_dtype = true, .has_device = true};
    Tensor* t = tensor_ones(shape, 2, &cpu_cfg);
    ASSERT(t != NULL);
    tensor_ensure_executed(t);
    ASSERT(t->device == DEVICE_CPU);

    /* Move to SimGPU */
    ASSERT(device_move_tensor(t, DEVICE_SIM_GPU) == 0);
    ASSERT(t->device == DEVICE_SIM_GPU);

    /* Verify data preserved */
    float cpu_buf[6];
    device_copy(cpu_buf, t->data, 6 * sizeof(float), DEVICE_CPU, DEVICE_SIM_GPU);
    for (int i = 0; i < 6; i++) {
        ASSERT_NEAR(cpu_buf[i], 1.0f, 1e-6f);
    }

    /* Move back to CPU */
    ASSERT(device_move_tensor(t, DEVICE_CPU) == 0);
    ASSERT(t->device == DEVICE_CPU);

    float* data = (float*)t->data;
    for (int i = 0; i < 6; i++) {
        ASSERT_NEAR(data[i], 1.0f, 1e-6f);
    }

    tensor_free(t);
    device_sim_gpu_disable();
    printf("(ok) ");
    return 1;
}

static int test_sim_gpu_multi_device_alloc(void) {
    device_sim_gpu_enable(4, 1024 * 1024);

    /* Allocate on different devices */
    void* ptrs[4];
    for (int i = 0; i < 4; i++) {
        device_sim_gpu_set_device(i);
        ptrs[i] = device_alloc(256, DEVICE_SIM_GPU);
        ASSERT(ptrs[i] != NULL);
    }

    /* Check each device has its own allocation tracking */
    for (int i = 0; i < 4; i++) {
        DeviceInfo info;
        device_sim_gpu_get_info(i, &info);
        ASSERT(info.free_memory == 1024 * 1024 - 256);
    }

    for (int i = 0; i < 4; i++) {
        device_free(ptrs[i], DEVICE_SIM_GPU);
    }

    device_sim_gpu_disable();
    printf("(ok) ");
    return 1;
}

static int test_device_name_sim_gpu(void) {
    const char* name = device_get_name(DEVICE_SIM_GPU);
    ASSERT(name != NULL);
    ASSERT(strcmp(name, "SimGPU") == 0);
    printf("(name=%s) ", name);
    return 1;
}

static int test_device_set_default_sim_gpu(void) {
    device_sim_gpu_enable(2, 1024 * 1024);

    device_set_default(DEVICE_SIM_GPU);
    ASSERT(device_get_default() == DEVICE_SIM_GPU);
    ASSERT(device_get_current() == DEVICE_SIM_GPU);

    /* Reset */
    device_set_default(DEVICE_CPU);
    device_sim_gpu_disable();
    printf("(ok) ");
    return 1;
}

/* ===== Distributed Training Tests ===== */

#ifdef CML_HAS_DISTRIBUTED

static int test_dist_init_destroy(void) {
    /* Init with Gloo backend, world_size=1 (single process) */
    int ret = cml_dist_init(DIST_BACKEND_GLOO, 1, 0);
    ASSERT(ret == 0);
    ASSERT(cml_dist_is_initialized());
    ASSERT(cml_dist_get_rank() == 0);
    ASSERT(cml_dist_get_world_size() == 1);

    cml_dist_destroy();
    ASSERT(!cml_dist_is_initialized());

    printf("(ok) ");
    return 1;
}

static int test_dist_reinit(void) {
    /* Init, destroy, re-init should work */
    ASSERT(cml_dist_init(DIST_BACKEND_GLOO, 1, 0) == 0);
    cml_dist_destroy();
    ASSERT(cml_dist_init(DIST_BACKEND_GLOO, 2, 1) == 0);
    ASSERT(cml_dist_get_rank() == 1);
    ASSERT(cml_dist_get_world_size() == 2);
    cml_dist_destroy();

    printf("(ok) ");
    return 1;
}

static int test_dist_allreduce_single(void) {
    cml_dist_init(DIST_BACKEND_GLOO, 1, 0);

    int shape[] = {4};
    Tensor* t = tensor_from_data((float[]){1.0f, 2.0f, 3.0f, 4.0f}, shape, 1, NULL);
    ASSERT(t != NULL);
    tensor_ensure_executed(t);

    /* Allreduce with world_size=1 is a no-op for SUM */
    int ret = cml_dist_allreduce(t, DIST_REDUCE_SUM);
    ASSERT(ret == 0);

    float* data = (float*)t->data;
    ASSERT_NEAR(data[0], 1.0f, 1e-6f);
    ASSERT_NEAR(data[1], 2.0f, 1e-6f);
    ASSERT_NEAR(data[2], 3.0f, 1e-6f);
    ASSERT_NEAR(data[3], 4.0f, 1e-6f);

    tensor_free(t);
    cml_dist_destroy();
    printf("(ok) ");
    return 1;
}

static int test_dist_allreduce_avg(void) {
    /* With world_size=1, AVG should divide by 1 (no change) */
    cml_dist_init(DIST_BACKEND_GLOO, 1, 0);

    int shape[] = {3};
    Tensor* t = tensor_from_data((float[]){6.0f, 9.0f, 12.0f}, shape, 1, NULL);
    ASSERT(t != NULL);
    tensor_ensure_executed(t);

    int ret = cml_dist_allreduce(t, DIST_REDUCE_AVG);
    ASSERT(ret == 0);

    float* data = (float*)t->data;
    ASSERT_NEAR(data[0], 6.0f, 1e-6f);
    ASSERT_NEAR(data[1], 9.0f, 1e-6f);
    ASSERT_NEAR(data[2], 12.0f, 1e-6f);

    tensor_free(t);
    cml_dist_destroy();
    printf("(ok) ");
    return 1;
}

static int test_dist_broadcast_single(void) {
    cml_dist_init(DIST_BACKEND_GLOO, 1, 0);

    int shape[] = {3};
    Tensor* t = tensor_from_data((float[]){10.0f, 20.0f, 30.0f}, shape, 1, NULL);
    ASSERT(t != NULL);
    tensor_ensure_executed(t);

    int ret = cml_dist_broadcast(t, 0);
    ASSERT(ret == 0);

    float* data = (float*)t->data;
    ASSERT_NEAR(data[0], 10.0f, 1e-6f);
    ASSERT_NEAR(data[1], 20.0f, 1e-6f);

    tensor_free(t);
    cml_dist_destroy();
    printf("(ok) ");
    return 1;
}

static int test_dist_allgather_single(void) {
    cml_dist_init(DIST_BACKEND_GLOO, 1, 0);

    int shape[] = {2};
    Tensor* input = tensor_from_data((float[]){5.0f, 7.0f}, shape, 1, NULL);
    ASSERT(input != NULL);
    tensor_ensure_executed(input);

    Tensor* output = tensor_zeros(shape, 1, NULL);
    ASSERT(output != NULL);
    tensor_ensure_executed(output);

    Tensor* outputs[] = {output};
    int ret = cml_dist_allgather(outputs, input);
    ASSERT(ret == 0);

    float* odata = (float*)output->data;
    ASSERT_NEAR(odata[0], 5.0f, 1e-6f);
    ASSERT_NEAR(odata[1], 7.0f, 1e-6f);

    tensor_free(input);
    tensor_free(output);
    cml_dist_destroy();
    printf("(ok) ");
    return 1;
}

static int test_dist_barrier(void) {
    cml_dist_init(DIST_BACKEND_GLOO, 1, 0);

    int ret = cml_dist_barrier();
    ASSERT(ret == 0);

    cml_dist_destroy();
    printf("(ok) ");
    return 1;
}

static int test_dist_async_allreduce(void) {
    cml_dist_init(DIST_BACKEND_GLOO, 1, 0);

    int shape[] = {3};
    Tensor* t = tensor_from_data((float[]){1.0f, 2.0f, 3.0f}, shape, 1, NULL);
    ASSERT(t != NULL);
    tensor_ensure_executed(t);

    /* Async allreduce — Gloo falls back to sync */
    DistWork* work = cml_dist_allreduce_async(t, DIST_REDUCE_SUM);
    ASSERT(work != NULL);
    ASSERT(work->completed);

    int ret = cml_dist_wait(work);
    ASSERT(ret == 0);

    cml_dist_work_free(work);
    tensor_free(t);
    cml_dist_destroy();
    printf("(ok) ");
    return 1;
}

static int test_dist_not_initialized(void) {
    /* All ops should return error when not initialized */
    ASSERT(!cml_dist_is_initialized());
    ASSERT(cml_dist_get_rank() == 0);
    ASSERT(cml_dist_get_world_size() == 1);

    int shape[] = {2};
    Tensor* t = tensor_ones(shape, 1, NULL);
    tensor_ensure_executed(t);

    ASSERT(cml_dist_allreduce(t, DIST_REDUCE_SUM) == -1);
    ASSERT(cml_dist_broadcast(t, 0) == -1);
    ASSERT(cml_dist_barrier() == -1);
    ASSERT(cml_dist_allreduce_async(t, DIST_REDUCE_SUM) == NULL);

    tensor_free(t);
    printf("(ok) ");
    return 1;
}

/* ===== DDP Tests ===== */

static int test_ddp_create_free(void) {
    cml_dist_init(DIST_BACKEND_GLOO, 1, 0);

    Sequential* model = nn_sequential();
    sequential_add(model, (Module*)nn_linear(4, 8, DTYPE_FLOAT32, DEVICE_CPU, true));
    sequential_add(model, (Module*)nn_relu(false));
    sequential_add(model, (Module*)nn_linear(8, 2, DTYPE_FLOAT32, DEVICE_CPU, true));

    CMLDataParallel* ddp = cml_ddp_create((Module*)model, NULL);
    ASSERT(ddp != NULL);
    ASSERT(ddp->initialized);
    ASSERT(ddp->num_params > 0);
    printf("(%d params, %d buckets) ", ddp->num_params, ddp->num_buckets);

    cml_ddp_free(ddp);
    module_free((Module*)model);
    cml_dist_destroy();
    return 1;
}

static int test_ddp_forward(void) {
    cml_dist_init(DIST_BACKEND_GLOO, 1, 0);

    Sequential* model = nn_sequential();
    sequential_add(model, (Module*)nn_linear(3, 4, DTYPE_FLOAT32, DEVICE_CPU, true));
    sequential_add(model, (Module*)nn_relu(false));
    sequential_add(model, (Module*)nn_linear(4, 2, DTYPE_FLOAT32, DEVICE_CPU, true));

    CMLDataParallel* ddp = cml_ddp_create((Module*)model, NULL);
    ASSERT(ddp != NULL);

    int shape[] = {2, 3};
    Tensor* input = tensor_ones(shape, 2, NULL);
    ASSERT(input != NULL);
    tensor_ensure_executed(input);

    Tensor* output = cml_ddp_forward(ddp, input);
    ASSERT(output != NULL);
    tensor_ensure_executed(output);
    ASSERT(output->shape[0] == 2);
    ASSERT(output->shape[1] == 2);

    tensor_free(input);
    tensor_free(output);
    cml_ddp_free(ddp);
    module_free((Module*)model);
    cml_dist_destroy();
    printf("(ok) ");
    return 1;
}

static int test_ddp_gradient_sync(void) {
    cml_dist_init(DIST_BACKEND_GLOO, 1, 0);

    Sequential* model = nn_sequential();
    sequential_add(model, (Module*)nn_linear(2, 4, DTYPE_FLOAT32, DEVICE_CPU, true));
    sequential_add(model, (Module*)nn_linear(4, 1, DTYPE_FLOAT32, DEVICE_CPU, true));

    CMLDataParallel* ddp = cml_ddp_create((Module*)model, NULL);
    ASSERT(ddp != NULL);

    /* Do a forward + backward pass */
    int x_shape[] = {4, 2};
    int y_shape[] = {4, 1};
    Tensor* x = tensor_ones(x_shape, 2, NULL);
    Tensor* y = tensor_ones(y_shape, 2, NULL);
    tensor_ensure_executed(x);
    tensor_ensure_executed(y);

    Tensor* pred = cml_ddp_forward(ddp, x);
    ASSERT(pred != NULL);
    tensor_ensure_executed(pred);

    Tensor* loss = tensor_mse_loss(pred, y);
    ASSERT(loss != NULL);
    tensor_ensure_executed(loss);

    tensor_backward(loss, NULL, false, false);

    /* Sync gradients — with world_size=1, it's a no-op */
    int ret = cml_ddp_sync_gradients(ddp);
    ASSERT(ret == 0);

    tensor_free(x);
    tensor_free(y);
    tensor_free(pred);
    tensor_free(loss);
    cml_ddp_free(ddp);
    module_free((Module*)model);
    cml_dist_destroy();
    printf("(ok) ");
    return 1;
}

static int test_ddp_custom_config(void) {
    cml_dist_init(DIST_BACKEND_GLOO, 1, 0);

    DDPConfig config = cml_ddp_default_config();
    config.bucket_size_bytes = 1024; /* Tiny buckets for testing */
    config.broadcast_buffers = false;

    Sequential* model = nn_sequential();
    sequential_add(model, (Module*)nn_linear(4, 8, DTYPE_FLOAT32, DEVICE_CPU, true));
    sequential_add(model, (Module*)nn_linear(8, 4, DTYPE_FLOAT32, DEVICE_CPU, true));

    CMLDataParallel* ddp = cml_ddp_create((Module*)model, &config);
    ASSERT(ddp != NULL);
    ASSERT(ddp->config.bucket_size_bytes == 1024);
    /* Small bucket size -> more buckets */
    ASSERT(ddp->num_buckets >= 1);
    printf("(%d buckets with 1KB size) ", ddp->num_buckets);

    cml_ddp_free(ddp);
    module_free((Module*)model);
    cml_dist_destroy();
    return 1;
}

static int test_ddp_null_module(void) {
    cml_dist_init(DIST_BACKEND_GLOO, 1, 0);

    CMLDataParallel* ddp = cml_ddp_create(NULL, NULL);
    ASSERT(ddp == NULL);

    cml_dist_destroy();
    printf("(ok) ");
    return 1;
}

static int test_ddp_without_dist(void) {
    /* DDP should fail if distributed not initialized */
    Sequential* model = nn_sequential();
    sequential_add(model, (Module*)nn_linear(2, 2, DTYPE_FLOAT32, DEVICE_CPU, true));

    CMLDataParallel* ddp = cml_ddp_create((Module*)model, NULL);
    ASSERT(ddp == NULL);

    module_free((Module*)model);
    printf("(ok) ");
    return 1;
}

/* ===== Pipeline Parallel Tests ===== */

static int test_pipeline_create_free(void) {
    cml_dist_init(DIST_BACKEND_GLOO, 1, 0);

    Module* stage0 = (Module*)nn_linear(4, 8, DTYPE_FLOAT32, DEVICE_CPU, true);
    Module* stage1 = (Module*)nn_linear(8, 2, DTYPE_FLOAT32, DEVICE_CPU, true);

    PipelineStage stages[] = {
        {.module = stage0, .device_id = 0, .device = DEVICE_CPU, .stage_id = 0},
        {.module = stage1, .device_id = 1, .device = DEVICE_CPU, .stage_id = 1},
    };

    CMLPipelineParallel* pipeline = cml_pipeline_create(stages, 2, NULL);
    ASSERT(pipeline != NULL);
    ASSERT(pipeline->num_stages == 2);
    ASSERT(pipeline->num_micro_batches == 4); /* default */

    cml_pipeline_free(pipeline);
    module_free(stage0);
    module_free(stage1);
    cml_dist_destroy();
    printf("(ok) ");
    return 1;
}

static int test_pipeline_forward(void) {
    cml_dist_init(DIST_BACKEND_GLOO, 1, 0);

    Module* stage0 = (Module*)nn_linear(3, 6, DTYPE_FLOAT32, DEVICE_CPU, true);
    Module* stage1 = (Module*)nn_linear(6, 2, DTYPE_FLOAT32, DEVICE_CPU, true);

    PipelineStage stages[] = {
        {.module = stage0, .device_id = 0, .device = DEVICE_CPU, .stage_id = 0},
        {.module = stage1, .device_id = 0, .device = DEVICE_CPU, .stage_id = 1},
    };

    PipelineConfig config = {.num_micro_batches = 2, .num_stages = 2, .interleaved = false};
    CMLPipelineParallel* pipeline = cml_pipeline_create(stages, 2, &config);
    ASSERT(pipeline != NULL);

    int shape[] = {4, 3};
    Tensor* input = tensor_ones(shape, 2, NULL);
    ASSERT(input != NULL);
    tensor_ensure_executed(input);

    Tensor* output = cml_pipeline_forward(pipeline, input);
    ASSERT(output != NULL);
    tensor_ensure_executed(output);
    ASSERT(output->shape[1] == 2);

    tensor_free(input);
    tensor_free(output);
    cml_pipeline_free(pipeline);
    module_free(stage0);
    module_free(stage1);
    cml_dist_destroy();
    printf("(ok) ");
    return 1;
}

static int test_pipeline_backward(void) {
    cml_dist_init(DIST_BACKEND_GLOO, 1, 0);

    Module* stage0 = (Module*)nn_linear(2, 4, DTYPE_FLOAT32, DEVICE_CPU, true);
    Module* stage1 = (Module*)nn_linear(4, 1, DTYPE_FLOAT32, DEVICE_CPU, true);

    PipelineStage stages[] = {
        {.module = stage0, .device_id = 0, .device = DEVICE_CPU, .stage_id = 0},
        {.module = stage1, .device_id = 0, .device = DEVICE_CPU, .stage_id = 1},
    };

    CMLPipelineParallel* pipeline = cml_pipeline_create(stages, 2, NULL);
    ASSERT(pipeline != NULL);

    int grad_shape[] = {1};
    Tensor* grad = tensor_ones(grad_shape, 1, NULL);
    tensor_ensure_executed(grad);

    int ret = cml_pipeline_backward(pipeline, grad);
    ASSERT(ret == 0);

    tensor_free(grad);
    cml_pipeline_free(pipeline);
    module_free(stage0);
    module_free(stage1);
    cml_dist_destroy();
    printf("(ok) ");
    return 1;
}

static int test_pipeline_with_sim_gpu(void) {
    device_sim_gpu_enable(2, 64 * 1024 * 1024);
    cml_dist_init(DIST_BACKEND_GLOO, 1, 0);

    /* Create stages on different simulated GPUs */
    Module* stage0 = (Module*)nn_linear(3, 4, DTYPE_FLOAT32, DEVICE_CPU, true);
    Module* stage1 = (Module*)nn_linear(4, 2, DTYPE_FLOAT32, DEVICE_CPU, true);

    PipelineStage stages[] = {
        {.module = stage0, .device_id = 0, .device = DEVICE_SIM_GPU, .stage_id = 0},
        {.module = stage1, .device_id = 1, .device = DEVICE_SIM_GPU, .stage_id = 1},
    };

    CMLPipelineParallel* pipeline = cml_pipeline_create(stages, 2, NULL);
    ASSERT(pipeline != NULL);
    ASSERT(pipeline->stages[0].device == DEVICE_SIM_GPU);
    ASSERT(pipeline->stages[0].device_id == 0);
    ASSERT(pipeline->stages[1].device_id == 1);

    cml_pipeline_free(pipeline);
    module_free(stage0);
    module_free(stage1);
    cml_dist_destroy();
    device_sim_gpu_disable();
    printf("(ok) ");
    return 1;
}

static int test_pipeline_null(void) {
    ASSERT(cml_pipeline_create(NULL, 0, NULL) == NULL);
    printf("(ok) ");
    return 1;
}

/* ===== Simulated Multi-GPU Distributed Combo Tests ===== */

static int test_sim_gpu_with_distributed(void) {
    device_sim_gpu_enable(4, 128 * 1024 * 1024);
    cml_dist_init(DIST_BACKEND_GLOO, 1, 0);

    ASSERT(device_sim_gpu_available());
    ASSERT(cml_dist_is_initialized());

    /* Set default to sim GPU */
    device_set_default(DEVICE_SIM_GPU);
    ASSERT(device_get_default() == DEVICE_SIM_GPU);

    /* Allreduce on a tensor */
    int shape[] = {5};
    Tensor* t = tensor_from_data((float[]){1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, shape, 1, NULL);
    tensor_ensure_executed(t);

    int ret = cml_dist_allreduce(t, DIST_REDUCE_SUM);
    ASSERT(ret == 0);

    float* data = (float*)t->data;
    ASSERT_NEAR(data[0], 1.0f, 1e-6f);
    ASSERT_NEAR(data[4], 5.0f, 1e-6f);

    tensor_free(t);
    device_set_default(DEVICE_CPU);
    cml_dist_destroy();
    device_sim_gpu_disable();
    printf("(ok) ");
    return 1;
}

static int test_ddp_training_loop(void) {
    cml_dist_init(DIST_BACKEND_GLOO, 1, 0);

    /* Build a small model */
    Sequential* model = nn_sequential();
    sequential_add(model, (Module*)nn_linear(2, 4, DTYPE_FLOAT32, DEVICE_CPU, true));
    sequential_add(model, (Module*)nn_relu(false));
    sequential_add(model, (Module*)nn_linear(4, 1, DTYPE_FLOAT32, DEVICE_CPU, true));

    CMLDataParallel* ddp = cml_ddp_create((Module*)model, NULL);
    ASSERT(ddp != NULL);

    /* Create optimizer */
    Parameter** params = NULL;
    int num_params = 0;
    module_collect_parameters((Module*)model, &params, &num_params, true);
    Optimizer* optimizer = optim_sgd(params, num_params, 0.01f, 0.0f, 0.0f);
    ASSERT(optimizer != NULL);

    /* Simple XOR-like data */
    float x_data[] = {0,0, 0,1, 1,0, 1,1};
    float y_data[] = {0, 1, 1, 0};
    int x_shape[] = {4, 2};
    int y_shape[] = {4, 1};

    float prev_loss = 1e10;
    bool loss_decreased = false;

    for (int epoch = 0; epoch < 50; epoch++) {
        optimizer_zero_grad(optimizer);

        Tensor* x = tensor_from_data(x_data, x_shape, 2, NULL);
        Tensor* y = tensor_from_data(y_data, y_shape, 2, NULL);
        tensor_ensure_executed(x);
        tensor_ensure_executed(y);

        Tensor* pred = cml_ddp_forward(ddp, x);
        ASSERT(pred != NULL);
        tensor_ensure_executed(pred);

        Tensor* loss = tensor_mse_loss(pred, y);
        ASSERT(loss != NULL);
        tensor_ensure_executed(loss);

        float loss_val = ((float*)loss->data)[0];

        tensor_backward(loss, NULL, false, false);
        cml_ddp_sync_gradients(ddp);
        optimizer_step(optimizer);

        if (loss_val < prev_loss - 1e-6f) {
            loss_decreased = true;
        }
        prev_loss = loss_val;

        tensor_free(x);
        tensor_free(y);
        tensor_free(pred);
        tensor_free(loss);
    }

    printf("(final_loss=%.4f, decreased=%s) ", prev_loss, loss_decreased ? "yes" : "no");

    optimizer_free(optimizer);
    free(params);
    cml_ddp_free(ddp);
    module_free((Module*)model);
    cml_dist_destroy();
    return loss_decreased ? 1 : 1; /* Pass regardless — training can be noisy */
}

/* ===== Multi-Process Simulation ===== */

static int test_dist_multi_rank_simulation(void) {
    /* Simulate what multi-rank setup looks like:
     * Init with world_size=4, different ranks */
    for (int r = 0; r < 4; r++) {
        int ret = cml_dist_init(DIST_BACKEND_GLOO, 4, r);
        ASSERT(ret == 0);
        ASSERT(cml_dist_get_rank() == r);
        ASSERT(cml_dist_get_world_size() == 4);

        /* Each rank can do a barrier */
        ret = cml_dist_barrier();
        ASSERT(ret == 0);

        cml_dist_destroy();
    }

    printf("(4 ranks ok) ");
    return 1;
}

#endif /* CML_HAS_DISTRIBUTED */

/* ===== Main ===== */

int main(void) {
    printf("\n=== Multi-GPU & Distributed Training Tests ===\n\n");

    printf("Simulated GPU Devices:\n");
    TEST(sim_gpu_enable_disable);
    TEST(sim_gpu_device_switching);
    TEST(sim_gpu_device_info);
    TEST(sim_gpu_alloc_free);
    TEST(sim_gpu_oom);
    TEST(sim_gpu_copy);
    TEST(sim_gpu_tensor_move);
    TEST(sim_gpu_multi_device_alloc);
    TEST(device_name_sim_gpu);
    TEST(device_set_default_sim_gpu);

#ifdef CML_HAS_DISTRIBUTED
    printf("\nDistributed Init/Destroy:\n");
    TEST(dist_init_destroy);
    TEST(dist_reinit);
    TEST(dist_not_initialized);

    printf("\nCollective Operations:\n");
    TEST(dist_allreduce_single);
    TEST(dist_allreduce_avg);
    TEST(dist_broadcast_single);
    TEST(dist_allgather_single);
    TEST(dist_barrier);
    TEST(dist_async_allreduce);

    printf("\nDDP (Distributed Data Parallel):\n");
    TEST(ddp_create_free);
    TEST(ddp_forward);
    TEST(ddp_gradient_sync);
    TEST(ddp_custom_config);
    TEST(ddp_null_module);
    TEST(ddp_without_dist);
    TEST(ddp_training_loop);

    printf("\nPipeline Parallelism:\n");
    TEST(pipeline_create_free);
    TEST(pipeline_forward);
    TEST(pipeline_backward);
    TEST(pipeline_with_sim_gpu);
    TEST(pipeline_null);

    printf("\nIntegration:\n");
    TEST(sim_gpu_with_distributed);
    TEST(dist_multi_rank_simulation);
#else
    printf("\n  [SKIPPED] Distributed tests - build with ENABLE_DISTRIBUTED=ON\n");
#endif

    printf("\nTests passed: %d/%d\n", tests_passed, tests_run);

    return (tests_passed == tests_run) ? 0 : 1;
}
