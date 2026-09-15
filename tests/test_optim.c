#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "cml.h"
#include "alloc/cml_allocator.h"
#include "core/cml_flags.h"
#include "autograd/autograd.h"
#include "test_harness.h"

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
    cml_free(params);
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
    cml_free(params);
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
    cml_free(params);
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
    cml_free(params);
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
    cml_free(params);
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
    cml_free(params);
    module_free((Module*)model);
    return 1;
}

static int test_lamb(void) {
    Sequential* model; Parameter** params; int num_params;
    create_test_model(&model, &params, &num_params);

    Optimizer* opt = optim_lamb(params, num_params, 0.001f, 0.01f, 0.9f, 0.999f, 1e-6f);
    if (!opt) { cml_free(params); module_free((Module*)model); return 0; }

    optimizer_step(opt);
    printf("(ok) ");

    optimizer_free(opt);
    cml_free(params);
    module_free((Module*)model);
    return 1;
}

static int test_lars(void) {
    Sequential* model; Parameter** params; int num_params;
    create_test_model(&model, &params, &num_params);

    Optimizer* opt = optim_lars(params, num_params, 0.01f, 0.9f, 0.0f, 0.02f);
    if (!opt) { cml_free(params); module_free((Module*)model); return 0; }

    optimizer_step(opt);
    printf("(ok) ");

    optimizer_free(opt);
    cml_free(params);
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
    cml_free(params);
    module_free((Module*)model);
    return 1;
}

static int test_lr_scheduler_step(void) {
    Sequential* model; Parameter** params; int num_params;
    create_test_model(&model, &params, &num_params);

    Optimizer* opt = cml_optim_sgd(params, num_params, 0.1f, 0.0f, 0.0f);
    if (!opt) return 0;

    LRScheduler* sched = lr_scheduler_step(opt, 10, 0.1f);
    if (!sched) { optimizer_free(opt); cml_free(params); module_free((Module*)model); return 0; }

    float lr_before = lr_scheduler_get_lr(sched);
    for (int i = 0; i < 10; i++)
        lr_scheduler_update(sched, 0.0f);
    float lr_after = lr_scheduler_get_lr(sched);

    printf("(lr: %.4f -> %.4f) ", lr_before, lr_after);
    int ok = (lr_after < lr_before);

    lr_scheduler_free(sched);
    optimizer_free(opt);
    cml_free(params);
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

/* FUSE_OPTIM only changes how the SGD updates are scheduled (all emitted then
 * realized in one pass), not the math. Train two identical models on identical
 * data -- one default, one with FUSE_OPTIM -- and require the weights to match.
 * A double-applied momentum update (the main risk of the batched realize) would
 * diverge here. */
static int test_fuse_optim_matches_default(void) {
    Sequential *ma, *mb; Parameter **pa, **pb; int na, nb;
    create_test_model(&ma, &pa, &na);
    create_test_model(&mb, &pb, &nb);
    if (na != nb || na == 0) return 0;

    /* Align B's initial weights to A's (both were randomly initialized). */
    for (int i = 0; i < na; i++) {
        Tensor *ta = pa[i]->tensor, *tb = pb[i]->tensor;
        if (ta->numel != tb->numel) return 0;
        memcpy(tb->data, ta->data, ta->numel * sizeof(float));
    }

    Optimizer* oa = cml_optim_sgd(pa, na, 0.05f, 0.0f, 0.9f); /* momentum on */
    Optimizer* ob = cml_optim_sgd(pb, nb, 0.05f, 0.0f, 0.9f);
    if (!oa || !ob) return 0;

    cml_nn_module_set_training((Module*)ma, true);
    cml_nn_module_set_training((Module*)mb, true);

    float xin[8] = {0.5f, -0.3f, 0.1f, 0.9f, -0.7f, 0.2f, 0.4f, -0.6f}; /* [4,2] */
    float yt[4]  = {1.0f, 0.0f, -1.0f, 0.5f};                            /* [4,1] */
    int xs[2] = {4, 2}, ys[2] = {4, 1};

    for (int step = 0; step < 4; step++) {
        /* Default path */
        Tensor* xa = tensor_from_data(xin, xs, 2, NULL);
        Tensor* ya = tensor_from_data(yt, ys, 2, NULL);
        optimizer_zero_grad(oa);
        Tensor* outa  = cml_nn_module_forward((Module*)ma, xa);
        Tensor* lossa = cml_nn_mse_loss(outa, ya);
        tensor_backward(lossa, NULL, false, false);
        optimizer_step(oa);
        tensor_free(lossa); tensor_free(outa); tensor_free(xa); tensor_free(ya);
        cml_reset_ir_context();

        /* FUSE_OPTIM path */
        int prev = cml_flag_push(CML_FLAG_FUSE_OPTIM, 1);
        Tensor* xb = tensor_from_data(xin, xs, 2, NULL);
        Tensor* yb = tensor_from_data(yt, ys, 2, NULL);
        optimizer_zero_grad(ob);
        Tensor* outb  = cml_nn_module_forward((Module*)mb, xb);
        Tensor* lossb = cml_nn_mse_loss(outb, yb);
        tensor_backward(lossb, NULL, false, false);
        optimizer_step(ob);
        tensor_free(lossb); tensor_free(outb); tensor_free(xb); tensor_free(yb);
        cml_reset_ir_context();
        cml_flag_pop(CML_FLAG_FUSE_OPTIM, prev);
    }

    int ok = 1;
    for (int i = 0; i < na && ok; i++) {
        const float* a = (const float*)pa[i]->tensor->data;
        const float* b = (const float*)pb[i]->tensor->data;
        for (size_t j = 0; j < pa[i]->tensor->numel; j++) {
            if (fabsf(a[j] - b[j]) > 1e-5f) {
                printf("(param %d[%zu]: default=%.6f fused=%.6f) ", i, j, a[j], b[j]);
                ok = 0; break;
            }
        }
    }

    optimizer_free(oa); optimizer_free(ob);
    cml_free(pa); cml_free(pb);
    module_free((Module*)ma); module_free((Module*)mb);
    return ok;
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
    TEST(fuse_optim_matches_default);

    return TEST_SUMMARY();
}
