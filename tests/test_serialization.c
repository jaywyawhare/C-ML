#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <unistd.h>

#include "cml.h"
#include "alloc/cml_allocator.h"
#include "test_harness.h"

#define APPROX_EQ(a, b) (fabsf((a) - (b)) < 1e-4f)

static int test_model_save_load(void) {
    const char* filepath = "/tmp/cml_test_model.bin";

    /* Create model */
    Sequential* model = cml_nn_sequential();
    sequential_add(model, (Module*)cml_nn_linear(4, 8, DTYPE_FLOAT32, DEVICE_CPU, true));
    sequential_add(model, (Module*)cml_nn_relu(false));
    sequential_add(model, (Module*)cml_nn_linear(8, 2, DTYPE_FLOAT32, DEVICE_CPU, true));

    /* Save */
    int ret = model_save((Module*)model, filepath);
    if (ret != 0) {
        printf("(save failed) ");
        module_free((Module*)model);
        return 0;
    }

    /* Create new model with same architecture */
    Sequential* loaded = cml_nn_sequential();
    sequential_add(loaded, (Module*)cml_nn_linear(4, 8, DTYPE_FLOAT32, DEVICE_CPU, true));
    sequential_add(loaded, (Module*)cml_nn_relu(false));
    sequential_add(loaded, (Module*)cml_nn_linear(8, 2, DTYPE_FLOAT32, DEVICE_CPU, true));

    /* Load */
    ret = model_load((Module*)loaded, filepath);
    if (ret != 0) {
        printf("(load failed) ");
        module_free((Module*)model);
        module_free((Module*)loaded);
        unlink(filepath);
        return 0;
    }

    /* Verify parameters match */
    Parameter** orig_params   = NULL;
    Parameter** loaded_params = NULL;
    int n_orig = 0, n_loaded = 0;
    module_collect_parameters((Module*)model, &orig_params, &n_orig, true);
    module_collect_parameters((Module*)loaded, &loaded_params, &n_loaded, true);

    int ok = (n_orig == n_loaded);
    if (ok) {
        for (int i = 0; i < n_orig && ok; i++) {
            if (orig_params[i] && loaded_params[i] && orig_params[i]->tensor &&
                loaded_params[i]->tensor) {
                float* od = (float*)orig_params[i]->tensor->data;
                float* ld = (float*)loaded_params[i]->tensor->data;
                for (size_t j = 0; j < orig_params[i]->tensor->numel && ok; j++) {
                    if (!APPROX_EQ(od[j], ld[j]))
                        ok = 0;
                }
            }
        }
    }

    printf("(params_match=%s) ", ok ? "yes" : "no");

    if (orig_params)
        cml_free(orig_params);
    if (loaded_params)
        cml_free(loaded_params);
    module_free((Module*)model);
    module_free((Module*)loaded);
    unlink(filepath);
    return ok;
}

static int test_checkpoint_save_load(void) {
    const char* filepath = "/tmp/cml_test_checkpoint.bin";

    Sequential* model = cml_nn_sequential();
    sequential_add(model, (Module*)cml_nn_linear(2, 4, DTYPE_FLOAT32, DEVICE_CPU, true));

    Parameter** params = NULL;
    int num_params     = 0;
    module_collect_parameters((Module*)model, &params, &num_params, true);

    Optimizer* opt = cml_optim_sgd(params, num_params, 0.01f, 0.0f, 0.0f);
    if (!opt) {
        cml_free(params);
        module_free((Module*)model);
        return 0;
    }

    int save_epoch  = 5;
    float save_loss = 0.123f;

    int ret = model_save_checkpoint((Module*)model, opt, save_epoch, save_loss, filepath);
    if (ret != 0) {
        printf("(checkpoint save failed) ");
        optimizer_free(opt);
        cml_free(params);
        module_free((Module*)model);
        return 0;
    }

    /* Create matching model for load */
    Sequential* loaded = cml_nn_sequential();
    sequential_add(loaded, (Module*)cml_nn_linear(2, 4, DTYPE_FLOAT32, DEVICE_CPU, true));

    Parameter** loaded_params = NULL;
    int num_loaded            = 0;
    module_collect_parameters((Module*)loaded, &loaded_params, &num_loaded, true);

    Optimizer* loaded_opt = cml_optim_sgd(loaded_params, num_loaded, 0.01f, 0.0f, 0.0f);

    int load_epoch  = 0;
    float load_loss = 0.0f;

    ret = model_load_checkpoint((Module*)loaded, loaded_opt, &load_epoch, &load_loss, filepath);
    if (ret != 0) {
        printf("(checkpoint load failed) ");
        optimizer_free(opt);
        optimizer_free(loaded_opt);
        cml_free(params);
        cml_free(loaded_params);
        module_free((Module*)model);
        module_free((Module*)loaded);
        unlink(filepath);
        return 0;
    }

    printf("(epoch=%d, loss=%.3f) ", load_epoch, load_loss);
    int ok = (load_epoch == save_epoch) && APPROX_EQ(load_loss, save_loss);

    optimizer_free(opt);
    optimizer_free(loaded_opt);
    cml_free(params);
    cml_free(loaded_params);
    module_free((Module*)model);
    module_free((Module*)loaded);
    unlink(filepath);
    return ok;
}

/* Non-trainable module state has to survive a save/load, and neither format used
 * to carry it: both walked parameters only, so a trained BatchNorm reloaded with
 * running_mean 0 / running_var 1 and eval-mode inference silently normalized
 * against the fresh module's stats. Both calls reported success.
 *
 * `safetensors` selects the format so the two paths share one body. */
static int buffers_round_trip(bool safetensors) {
    const char* path   = safetensors ? "/tmp/cml_test_buffers.st" : "/tmp/cml_test_buffers.bin";
    const float marker = 7.0f;

    BatchNorm1d* src = cml_nn_batchnorm1d(4, 1e-5f, 0.1f, true, true, DTYPE_FLOAT32, DEVICE_CPU);
    if (!src)
        return 0;
    Module* sm = (Module*)src;
    if (sm->num_buffers < 2) {
        printf("(expected >=2 buffers, got %d) ", sm->num_buffers);
        module_free(sm);
        return 0;
    }

    /* Stamp every buffer with a value neither default init produces. */
    for (int i = 0; i < sm->num_buffers; i++) {
        if (tensor_ensure_executed(sm->buffers[i]) != 0) {
            module_free(sm);
            return 0;
        }
        float* d = (float*)tensor_data_ptr(sm->buffers[i]);
        for (size_t j = 0; j < sm->buffers[i]->numel; j++)
            d[j] = marker;
    }

    int saved = safetensors ? module_save_safetensors(sm, path) : module_save(sm, path);

    BatchNorm1d* dst = cml_nn_batchnorm1d(4, 1e-5f, 0.1f, true, true, DTYPE_FLOAT32, DEVICE_CPU);
    if (!dst) {
        module_free(sm);
        return 0;
    }
    Module* dm = (Module*)dst;
    int loaded = safetensors ? module_load_safetensors(dm, path) : module_load(dm, path);

    int ok = (saved == 0 && loaded == 0 && dm->num_buffers == sm->num_buffers);
    for (int i = 0; i < dm->num_buffers && ok; i++) {
        if (tensor_ensure_executed(dm->buffers[i]) != 0) {
            ok = 0;
            break;
        }
        const float* d = (const float*)tensor_data_ptr(dm->buffers[i]);
        if (!d) {
            ok = 0;
            break;
        }
        for (size_t j = 0; j < dm->buffers[i]->numel && ok; j++)
            if (!APPROX_EQ(d[j], marker)) {
                printf("(buffer %d elem %zu: %g, want %g) ", i, j, (double)d[j], (double)marker);
                ok = 0;
            }
    }

    module_free(sm);
    module_free(dm);
    unlink(path);
    return ok;
}

static int test_buffers_survive_native_round_trip(void) { return buffers_round_trip(false); }
static int test_buffers_survive_safetensors_round_trip(void) { return buffers_round_trip(true); }

static int test_save_null(void) {
    int ret = model_save(NULL, "/tmp/test.bin");
    return (ret != 0);
}

static int test_load_nonexistent(void) {
    Sequential* model = cml_nn_sequential();
    sequential_add(model, (Module*)cml_nn_linear(2, 4, DTYPE_FLOAT32, DEVICE_CPU, true));

    int ret = model_load((Module*)model, "/tmp/nonexistent_cml_model_12345.bin");
    module_free((Module*)model);
    return (ret != 0);
}

static int test_save_bad_path(void) {
    Sequential* model = cml_nn_sequential();
    sequential_add(model, (Module*)cml_nn_linear(2, 4, DTYPE_FLOAT32, DEVICE_CPU, true));

    int ret = model_save((Module*)model, "/nonexistent_dir_12345/model.bin");
    module_free((Module*)model);
    return (ret != 0);
}

int main(void) {
    cml_init();

    printf("\nSerialization Unit Tests\n\n");

    printf("Model I/O:\n");
    TEST(model_save_load);
    TEST(checkpoint_save_load);
    TEST(buffers_survive_native_round_trip);
    TEST(buffers_survive_safetensors_round_trip);

    printf("\nError Handling:\n");
    TEST(save_null);
    TEST(load_nonexistent);
    TEST(save_bad_path);

    printf("\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_run);
    printf("\n");

    return (tests_passed == tests_run) ? 0 : 1;
}
