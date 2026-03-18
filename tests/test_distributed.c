#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "cml.h"
#include "nn/layers/linear.h"
#include "distributed/distributed.h"
#include "distributed/comm_backend.h"
#include "distributed/data_parallel.h"
#include "distributed/pipeline_parallel.h"
#include "distributed/tensor_parallel.h"

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  %-55s ", #name); \
    fflush(stdout); \
    if (test_##name()) { \
        tests_passed++; \
        printf("[PASS]\n"); \
    } else { \
        printf("[FAIL]\n"); \
    } \
} while(0)

#define EPSILON 1e-4f

static bool float_eq(float a, float b) {
    return fabsf(a - b) < EPSILON;
}

static Tensor* make_tensor_2d(const float* data, int rows, int cols) {
    int shape[2] = {rows, cols};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    return tensor_from_data(data, shape, 2, &cfg);
}

static Tensor* make_tensor_1d(const float* data, int len) {
    int shape[1] = {len};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    return tensor_from_data(data, shape, 1, &cfg);
}


static bool test_dist_init_destroy(void) {
    /* Should not be initialized yet */
    if (cml_dist_is_initialized())
        return false;

    int ret = cml_dist_init(DIST_BACKEND_GLOO, 1, 0);
    if (ret != 0)
        return false;

    if (!cml_dist_is_initialized())
        return false;

    if (cml_dist_get_rank() != 0)
        return false;

    if (cml_dist_get_world_size() != 1)
        return false;

    DistProcessGroup* group = cml_dist_get_default_group();
    if (!group || !group->initialized)
        return false;

    cml_dist_destroy();

    if (cml_dist_is_initialized())
        return false;

    return true;
}

static bool test_dist_reinit(void) {
    /* Init, destroy, re-init should work */
    int ret = cml_dist_init(DIST_BACKEND_GLOO, 1, 0);
    if (ret != 0) return false;
    cml_dist_destroy();

    /* Re-init with single process (multi-process would need real peers) */
    ret = cml_dist_init(DIST_BACKEND_GLOO, 1, 0);
    if (ret != 0) return false;

    if (cml_dist_get_rank() != 0) { cml_dist_destroy(); return false; }
    if (cml_dist_get_world_size() != 1) { cml_dist_destroy(); return false; }

    cml_dist_destroy();
    return true;
}

static bool test_dist_double_init(void) {
    int ret = cml_dist_init(DIST_BACKEND_GLOO, 1, 0);
    if (ret != 0) return false;

    /* Double init should succeed (no-op) */
    ret = cml_dist_init(DIST_BACKEND_GLOO, 1, 0);
    if (ret != 0) { cml_dist_destroy(); return false; }

    cml_dist_destroy();
    return true;
}


static bool test_backend_name(void) {
    if (strcmp(cml_dist_backend_name(DIST_BACKEND_NCCL), "NCCL") != 0)
        return false;
    if (strcmp(cml_dist_backend_name(DIST_BACKEND_MPI), "MPI") != 0)
        return false;
    if (strcmp(cml_dist_backend_name(DIST_BACKEND_GLOO), "Gloo") != 0)
        return false;
    if (strcmp(cml_dist_backend_name(99), "Unknown") != 0)
        return false;
    return true;
}


static bool test_allreduce_single(void) {
    int ret = cml_dist_init(DIST_BACKEND_GLOO, 1, 0);
    if (ret != 0) return false;

    float data[] = {1.0f, 2.0f, 3.0f};
    Tensor* t = make_tensor_1d(data, 3);
    if (!t) { cml_dist_destroy(); return false; }

    ret = cml_dist_allreduce(t, DIST_REDUCE_SUM);
    if (ret != 0) { tensor_free(t); cml_dist_destroy(); return false; }

    /* Single process: SUM should be identity */
    tensor_ensure_executed(t);
    const float* result = (const float*)tensor_data_ptr(t);
    bool ok = float_eq(result[0], 1.0f) && float_eq(result[1], 2.0f) &&
              float_eq(result[2], 3.0f);

    tensor_free(t);
    cml_dist_destroy();
    return ok;
}

static bool test_allreduce_avg_single(void) {
    int ret = cml_dist_init(DIST_BACKEND_GLOO, 1, 0);
    if (ret != 0) return false;

    float data[] = {4.0f, 8.0f};
    Tensor* t = make_tensor_1d(data, 2);
    if (!t) { cml_dist_destroy(); return false; }

    ret = cml_dist_allreduce(t, DIST_REDUCE_AVG);
    if (ret != 0) { tensor_free(t); cml_dist_destroy(); return false; }

    /* AVG of single process: value / 1 = value */
    tensor_ensure_executed(t);
    const float* result = (const float*)tensor_data_ptr(t);
    bool ok = float_eq(result[0], 4.0f) && float_eq(result[1], 8.0f);

    tensor_free(t);
    cml_dist_destroy();
    return ok;
}

static bool test_broadcast_single(void) {
    int ret = cml_dist_init(DIST_BACKEND_GLOO, 1, 0);
    if (ret != 0) return false;

    float data[] = {5.0f, 10.0f};
    Tensor* t = make_tensor_1d(data, 2);
    if (!t) { cml_dist_destroy(); return false; }

    ret = cml_dist_broadcast(t, 0);
    if (ret != 0) { tensor_free(t); cml_dist_destroy(); return false; }

    /* Single process: broadcast is identity */
    tensor_ensure_executed(t);
    const float* result = (const float*)tensor_data_ptr(t);
    bool ok = float_eq(result[0], 5.0f) && float_eq(result[1], 10.0f);

    tensor_free(t);
    cml_dist_destroy();
    return ok;
}

static bool test_barrier_single(void) {
    int ret = cml_dist_init(DIST_BACKEND_GLOO, 1, 0);
    if (ret != 0) return false;

    ret = cml_dist_barrier();
    cml_dist_destroy();
    return ret == 0;
}

static bool test_allgather_single(void) {
    int ret = cml_dist_init(DIST_BACKEND_GLOO, 1, 0);
    if (ret != 0) return false;

    float data[] = {7.0f, 8.0f, 9.0f};
    Tensor* input = make_tensor_1d(data, 3);
    if (!input) { cml_dist_destroy(); return false; }

    float out_data[] = {0.0f, 0.0f, 0.0f};
    Tensor* output = make_tensor_1d(out_data, 3);
    if (!output) { tensor_free(input); cml_dist_destroy(); return false; }

    Tensor* outputs[] = {output};
    ret = cml_dist_allgather(outputs, input);
    if (ret != 0) {
        tensor_free(input); tensor_free(output);
        cml_dist_destroy();
        return false;
    }

    /* Single process: output[0] = input */
    tensor_ensure_executed(output);
    const float* result = (const float*)tensor_data_ptr(output);
    bool ok = float_eq(result[0], 7.0f) && float_eq(result[1], 8.0f) &&
              float_eq(result[2], 9.0f);

    tensor_free(input);
    tensor_free(output);
    cml_dist_destroy();
    return ok;
}


static bool test_allreduce_async_single(void) {
    int ret = cml_dist_init(DIST_BACKEND_GLOO, 1, 0);
    if (ret != 0) return false;

    float data[] = {1.0f, 2.0f};
    Tensor* t = make_tensor_1d(data, 2);
    if (!t) { cml_dist_destroy(); return false; }

    DistWork* work = cml_dist_allreduce_async(t, DIST_REDUCE_SUM);
    if (!work) { tensor_free(t); cml_dist_destroy(); return false; }

    ret = cml_dist_wait(work);
    if (ret != 0) {
        cml_dist_work_free(work);
        tensor_free(t);
        cml_dist_destroy();
        return false;
    }

    /* Should be completed */
    if (!work->completed) {
        cml_dist_work_free(work);
        tensor_free(t);
        cml_dist_destroy();
        return false;
    }

    cml_dist_work_free(work);
    tensor_free(t);
    cml_dist_destroy();
    return true;
}


static bool test_gloo_backend_create(void) {
    DistCommOps* ops = cml_dist_create_gloo_backend();
    if (!ops) return false;

    /* Verify all ops are wired */
    bool ok = ops->allreduce != NULL &&
              ops->broadcast != NULL &&
              ops->allgather != NULL &&
              ops->reduce_scatter != NULL &&
              ops->barrier != NULL &&
              ops->send != NULL &&
              ops->recv != NULL &&
              ops->allreduce_async != NULL &&
              ops->wait != NULL &&
              ops->init != NULL &&
              ops->destroy != NULL;

    cml_dist_free_backend(ops);
    return ok;
}


static bool test_pipeline_create_free(void) {
    /* Create a simple 2-stage pipeline with linear modules */
    Linear* l1 = nn_linear(4, 4, DTYPE_FLOAT32, DEVICE_CPU, true);
    Linear* l2 = nn_linear(4, 2, DTYPE_FLOAT32, DEVICE_CPU, true);
    if (!l1 || !l2) {
        if (l1) module_free(&l1->base);
        if (l2) module_free(&l2->base);
        return false;
    }
    Module* m1 = &l1->base;
    Module* m2 = &l2->base;

    PipelineStage stages[2] = {
        {.module = m1, .device_id = 0, .device = DEVICE_CPU, .stage_id = 0},
        {.module = m2, .device_id = 0, .device = DEVICE_CPU, .stage_id = 1}
    };

    CMLPipelineParallel* pipeline = cml_pipeline_create(stages, 2, NULL);
    if (!pipeline) {
        module_free(m1); module_free(m2);
        return false;
    }

    /* Check defaults */
    bool ok = pipeline->num_stages == 2 &&
              pipeline->num_micro_batches == 4;

    cml_pipeline_free(pipeline);
    module_free(m1);
    module_free(m2);
    return ok;
}

static bool test_pipeline_forward(void) {
    /* Need distributed init for pipeline */
    int ret = cml_dist_init(DIST_BACKEND_GLOO, 1, 0);
    if (ret != 0) return false;

    Linear* l1 = nn_linear(4, 4, DTYPE_FLOAT32, DEVICE_CPU, true);
    Linear* l2 = nn_linear(4, 2, DTYPE_FLOAT32, DEVICE_CPU, true);
    if (!l1 || !l2) {
        if (l1) module_free(&l1->base);
        if (l2) module_free(&l2->base);
        cml_dist_destroy();
        return false;
    }
    Module* m1 = &l1->base;
    Module* m2 = &l2->base;

    PipelineStage stages[2] = {
        {.module = m1, .device_id = 0, .device = DEVICE_CPU, .stage_id = 0},
        {.module = m2, .device_id = 0, .device = DEVICE_CPU, .stage_id = 1}
    };

    PipelineConfig config = {.num_micro_batches = 2, .num_stages = 2, .interleaved = false};
    CMLPipelineParallel* pipeline = cml_pipeline_create(stages, 2, &config);
    if (!pipeline) {
        module_free(m1); module_free(m2);
        cml_dist_destroy();
        return false;
    }

    /* Create input [4, 4] (batch=4, features=4) */
    float x_data[16];
    for (int i = 0; i < 16; i++) x_data[i] = (float)(i + 1) * 0.1f;
    Tensor* input = make_tensor_2d(x_data, 4, 4);
    if (!input) {
        cml_pipeline_free(pipeline);
        module_free(m1); module_free(m2);
        cml_dist_destroy();
        return false;
    }

    Tensor* output = cml_pipeline_forward(pipeline, input);
    bool ok = output != NULL;
    if (ok) {
        /* Output should be [4, 2] (batch=4, out_features=2) */
        ok = output->ndim == 2 && output->shape[0] == 4 && output->shape[1] == 2;
    }

    if (output) tensor_free(output);
    tensor_free(input);
    cml_pipeline_free(pipeline);
    module_free(m1);
    module_free(m2);
    cml_dist_destroy();
    return ok;
}


static bool test_ddp_create_free(void) {
    int ret = cml_dist_init(DIST_BACKEND_GLOO, 1, 0);
    if (ret != 0) return false;

    Linear* lin = nn_linear(4, 2, DTYPE_FLOAT32, DEVICE_CPU, true);
    if (!lin) { cml_dist_destroy(); return false; }
    Module* m = &lin->base;

    DDPConfig config = cml_ddp_default_config();
    CMLDataParallel* ddp = cml_ddp_create(m, &config);
    if (!ddp) {
        module_free(m);
        cml_dist_destroy();
        return false;
    }

    bool ok = ddp->initialized && ddp->num_params > 0 && ddp->num_buckets >= 1;

    cml_ddp_free(ddp);
    module_free(m);
    cml_dist_destroy();
    return ok;
}

static bool test_ddp_forward(void) {
    int ret = cml_dist_init(DIST_BACKEND_GLOO, 1, 0);
    if (ret != 0) return false;

    Linear* lin = nn_linear(4, 2, DTYPE_FLOAT32, DEVICE_CPU, true);
    if (!lin) { cml_dist_destroy(); return false; }
    Module* m = &lin->base;

    CMLDataParallel* ddp = cml_ddp_create(m, NULL);
    if (!ddp) {
        module_free(m);
        cml_dist_destroy();
        return false;
    }

    float x_data[] = {1, 2, 3, 4};
    Tensor* input = make_tensor_2d(x_data, 1, 4);
    if (!input) {
        cml_ddp_free(ddp); module_free(m);
        cml_dist_destroy();
        return false;
    }

    Tensor* output = cml_ddp_forward(ddp, input);
    bool ok = output != NULL && output->shape[0] == 1 && output->shape[1] == 2;

    if (output) tensor_free(output);
    tensor_free(input);
    cml_ddp_free(ddp);
    module_free(m);
    cml_dist_destroy();
    return ok;
}

static bool test_ddp_sync_gradients_single(void) {
    int ret = cml_dist_init(DIST_BACKEND_GLOO, 1, 0);
    if (ret != 0) return false;

    Linear* lin = nn_linear(4, 2, DTYPE_FLOAT32, DEVICE_CPU, true);
    if (!lin) { cml_dist_destroy(); return false; }
    Module* m = &lin->base;

    CMLDataParallel* ddp = cml_ddp_create(m, NULL);
    if (!ddp) {
        module_free(m);
        cml_dist_destroy();
        return false;
    }

    /* Sync gradients in single-process mode should be a no-op */
    ret = cml_ddp_sync_gradients(ddp);

    cml_ddp_free(ddp);
    module_free(m);
    cml_dist_destroy();
    return ret == 0;
}


static bool test_ddp_default_config(void) {
    DDPConfig config = cml_ddp_default_config();
    return config.bucket_size_bytes == 25 * 1024 * 1024 &&
           config.broadcast_buffers == true &&
           config.find_unused_parameters == false;
}


static bool test_allreduce_without_init(void) {
    /* Should fail gracefully when not initialized */
    float data[] = {1.0f};
    Tensor* t = make_tensor_1d(data, 1);
    if (!t) return false;

    int ret = cml_dist_allreduce(t, DIST_REDUCE_SUM);
    tensor_free(t);
    return ret != 0; /* Should fail */
}

static bool test_broadcast_without_init(void) {
    float data[] = {1.0f};
    Tensor* t = make_tensor_1d(data, 1);
    if (!t) return false;

    int ret = cml_dist_broadcast(t, 0);
    tensor_free(t);
    return ret != 0;
}

static bool test_ddp_without_init(void) {
    Linear* lin = nn_linear(4, 2, DTYPE_FLOAT32, DEVICE_CPU, true);
    if (!lin) return false;
    Module* m = &lin->base;

    CMLDataParallel* ddp = cml_ddp_create(m, NULL);
    module_free(m);

    /* Should fail because distributed is not initialized */
    if (ddp) {
        cml_ddp_free(ddp);
        return false;
    }
    return true;
}

static bool test_pipeline_null_stages(void) {
    CMLPipelineParallel* p = cml_pipeline_create(NULL, 0, NULL);
    return p == NULL; /* Should fail */
}


int main(void) {
    printf("=== Distributed Training Tests ===\n\n");

    printf("Process group lifecycle:\n");
    TEST(dist_init_destroy);
    TEST(dist_reinit);
    TEST(dist_double_init);

    printf("\nBackend utilities:\n");
    TEST(backend_name);
    TEST(gloo_backend_create);

    printf("\nCollective operations (single-process):\n");
    TEST(allreduce_single);
    TEST(allreduce_avg_single);
    TEST(broadcast_single);
    TEST(barrier_single);
    TEST(allgather_single);

    printf("\nAsync operations:\n");
    TEST(allreduce_async_single);

    printf("\nPipeline parallel:\n");
    TEST(pipeline_create_free);
    TEST(pipeline_forward);

    printf("\nData parallel (DDP):\n");
    TEST(ddp_default_config);
    TEST(ddp_create_free);
    TEST(ddp_forward);
    TEST(ddp_sync_gradients_single);

    printf("\nError handling:\n");
    TEST(allreduce_without_init);
    TEST(broadcast_without_init);
    TEST(ddp_without_init);
    TEST(pipeline_null_stages);

    printf("\n=== Results: %d/%d tests passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
