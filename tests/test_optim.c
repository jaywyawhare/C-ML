#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "cml.h"

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    printf("  %-30s ", #name); \
    tests_run++; \
    if (test_##name()) { \
        printf("PASS\n"); \
        tests_passed++; \
    } else { \
        printf("FAIL\n"); \
    } \
} while(0)

#define APPROX_EQ(a, b) (fabsf((a) - (b)) < 1e-4f)

static void create_test_model(Sequential** model_out, Parameter*** params_out, int* num_params_out) {
    Sequential* model = cml_nn_sequential();
    sequential_add(model, (Module*)cml_nn_linear(2, 4, DTYPE_FLOAT32, DEVICE_CPU, true));
    sequential_add(model, (Module*)cml_nn_relu(false));
    sequential_add(model, (Module*)cml_nn_linear(4, 1, DTYPE_FLOAT32, DEVICE_CPU, true));

    Parameter** params = NULL;
    int num_params = 0;
    module_collect_parameters((Module*)model, &params, &num_params, true);

    *model_out = model;
    *params_out = params;
    *num_params_out = num_params;
}

static int test_sgd(void) {
    Sequential* model; Parameter** params; int num_params;
    create_test_model(&model, &params, &num_params);

    Optimizer* opt = cml_optim_sgd(params, num_params, 0.01f, 0.0f, 0.0f);
    if (!opt) return 0;

    for (int i = 0; i < num_params; i++) {
        if (params[i] && params[i]->tensor && params[i]->tensor->grad) {
            float* grad = (float*)params[i]->tensor->grad->data;
            for (size_t j = 0; j < params[i]->tensor->numel; j++)
                grad[j] = 1.0f;
        }
    }

    optimizer_step(opt);
    printf("(ok) ");

    optimizer_free(opt);
    free(params);
    module_free((Module*)model);
    return 1;
}

static int test_adam(void) {
    Sequential* model; Parameter** params; int num_params;
    create_test_model(&model, &params, &num_params);

    Optimizer* opt = cml_optim_adam(params, num_params, 0.001f, 0.0f, 0.9f, 0.999f, 1e-8f);
    if (!opt) return 0;

    optimizer_step(opt);
    printf("(ok) ");

    optimizer_free(opt);
    free(params);
    module_free((Module*)model);
    return 1;
}

static int test_adamw(void) {
    Sequential* model; Parameter** params; int num_params;
    create_test_model(&model, &params, &num_params);

    Optimizer* opt = optim_adamw(params, num_params, 0.001f, 0.01f, 0.9f, 0.999f, 1e-8f);
    if (!opt) return 0;

    optimizer_step(opt);
    printf("(ok) ");

    optimizer_free(opt);
    free(params);
    module_free((Module*)model);
    return 1;
}

static int test_rmsprop(void) {
    Sequential* model; Parameter** params; int num_params;
    create_test_model(&model, &params, &num_params);

    Optimizer* opt = cml_optim_rmsprop(params, num_params, 0.01f, 0.0f, 0.99f, 1e-8f);
    if (!opt) return 0;

    optimizer_step(opt);
    printf("(ok) ");

    optimizer_free(opt);
    free(params);
    module_free((Module*)model);
    return 1;
}

static int test_adagrad(void) {
    Sequential* model; Parameter** params; int num_params;
    create_test_model(&model, &params, &num_params);

    Optimizer* opt = cml_optim_adagrad(params, num_params, 0.01f, 0.0f, 1e-8f);
    if (!opt) return 0;

    optimizer_step(opt);
    printf("(ok) ");

    optimizer_free(opt);
    free(params);
    module_free((Module*)model);
    return 1;
}

static int test_adadelta(void) {
    Sequential* model; Parameter** params; int num_params;
    create_test_model(&model, &params, &num_params);

    Optimizer* opt = optim_adadelta(params, num_params, 0.9f, 0.0f, 1e-6f);
    if (!opt) return 0;

    optimizer_step(opt);
    printf("(ok) ");

    optimizer_free(opt);
    free(params);
    module_free((Module*)model);
    return 1;
}

static int test_lamb(void) {
    Sequential* model; Parameter** params; int num_params;
    create_test_model(&model, &params, &num_params);

    Optimizer* opt = optim_lamb(params, num_params, 0.001f, 0.01f, 0.9f, 0.999f, 1e-6f);
    if (!opt) { free(params); module_free((Module*)model); return 0; }

    optimizer_step(opt);
    printf("(ok) ");

    optimizer_free(opt);
    free(params);
    module_free((Module*)model);
    return 1;
}

static int test_lars(void) {
    Sequential* model; Parameter** params; int num_params;
    create_test_model(&model, &params, &num_params);

    Optimizer* opt = optim_lars(params, num_params, 0.01f, 0.9f, 0.0f, 0.02f);
    if (!opt) { free(params); module_free((Module*)model); return 0; }

    optimizer_step(opt);
    printf("(ok) ");

    optimizer_free(opt);
    free(params);
    module_free((Module*)model);
    return 1;
}

static int test_zero_grad(void) {
    Sequential* model; Parameter** params; int num_params;
    create_test_model(&model, &params, &num_params);

    Optimizer* opt = cml_optim_sgd(params, num_params, 0.01f, 0.0f, 0.0f);
    if (!opt) return 0;

    cml_optim_zero_grad(opt);
    printf("(ok) ");

    optimizer_free(opt);
    free(params);
    module_free((Module*)model);
    return 1;
}

static int test_lr_scheduler_step(void) {
    Sequential* model; Parameter** params; int num_params;
    create_test_model(&model, &params, &num_params);

    Optimizer* opt = cml_optim_sgd(params, num_params, 0.1f, 0.0f, 0.0f);
    if (!opt) return 0;

    LRScheduler* sched = lr_scheduler_step(opt, 10, 0.1f);
    if (!sched) { optimizer_free(opt); free(params); module_free((Module*)model); return 0; }

    float lr_before = lr_scheduler_get_lr(sched);
    for (int i = 0; i < 10; i++)
        lr_scheduler_update(sched, 0.0f);
    float lr_after = lr_scheduler_get_lr(sched);

    printf("(lr: %.4f -> %.4f) ", lr_before, lr_after);
    int ok = (lr_after < lr_before);

    lr_scheduler_free(sched);
    optimizer_free(opt);
    free(params);
    module_free((Module*)model);
    return ok;
}

static int test_optim_for_model(void) {
    Sequential* model = cml_nn_sequential();
    sequential_add(model, (Module*)cml_nn_linear(2, 4, DTYPE_FLOAT32, DEVICE_CPU, true));

    Optimizer* opt = cml_optim_adam_for_model((Module*)model, 0.001f, 0.0f, 0.9f, 0.999f, 1e-8f);
    if (!opt) { module_free((Module*)model); return 0; }

    printf("(ok) ");
    optimizer_free(opt);
    module_free((Module*)model);
    return 1;
}

int main(void) {
    cml_init();

    printf("test_optim\n\n");

    TEST(sgd);
    TEST(adam);
    TEST(adamw);
    TEST(rmsprop);
    TEST(adagrad);
    TEST(adadelta);
    TEST(lamb);
    TEST(lars);
    TEST(zero_grad);
    TEST(optim_for_model);
    TEST(lr_scheduler_step);

    printf("\n%d/%d passed\n", tests_passed, tests_run);

    return (tests_passed == tests_run) ? 0 : 1;
}
