#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "cml.h"
#include "test_harness.h"
#include "tensor/realize.h"
#include "alloc/cml_allocator.h"
#include "datasets/datasets.h"

static float get_scalar(Tensor* t) {
    float* p = (float*)tensor_data_ptr(t);
    return p ? p[0] : INFINITY;
}

static int test_xor_mlp(void) {
    printf("  [XOR MLP] Sequential(Linear(2,4)+Tanh+Linear(4,1)+Sigmoid), Adam lr=0.05, 2000 "
           "epochs\n");

    Sequential* model = nn_sequential();
    sequential_add(model, (Module*)nn_linear(2, 4, DTYPE_FLOAT32, DEVICE_CPU, true));
    sequential_add(model, (Module*)nn_tanh());
    sequential_add(model, (Module*)nn_linear(4, 1, DTYPE_FLOAT32, DEVICE_CPU, true));
    sequential_add(model, (Module*)nn_sigmoid());
    module_set_training((Module*)model, true);

    Parameter** params = NULL;
    int num_params     = 0;
    module_collect_parameters((Module*)model, &params, &num_params, true);
    Optimizer* optimizer = optim_adam(params, num_params, 0.05f, 0.0f, 0.9f, 0.999f, 1e-8f);
    if (!optimizer) {
        printf("    ERROR: failed to create optimizer\n");
        cml_free(params);
        module_free((Module*)model);
        return 0;
    }

    float xor_inputs[]  = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f};
    float xor_targets[] = {0.0f, 1.0f, 1.0f, 0.0f};

    Tensor* X = cml_tensor_2d(xor_inputs, 4, 2);
    Tensor* Y = cml_tensor_2d(xor_targets, 4, 1);
    if (X)
        tensor_realize(X);
    if (Y)
        tensor_realize(Y);

    float final_loss = INFINITY;

    for (int epoch = 0; epoch < 2000; epoch++) {
        optimizer_zero_grad(optimizer);

        Tensor* out  = module_forward((Module*)model, X);
        Tensor* loss = tensor_bce_loss(out, Y);
        final_loss   = get_scalar(loss);

        tensor_backward(loss, NULL, false, false);
        optimizer_step(optimizer);

        if ((epoch + 1) % 500 == 0) {
            printf("    epoch %4d  loss = %.6f\n", epoch + 1, (double)final_loss);
        }

        tensor_free(loss);
        tensor_free(out);
        cml_ir_reset_global_context();
    }

    int pass = (final_loss < 0.01f);
    printf("    final loss = %.6f  (threshold < 0.01) => %s\n", (double)final_loss,
           pass ? "PASS" : "FAIL");

    tensor_free(X);
    tensor_free(Y);
    optimizer_free(optimizer);
    cml_free(params);
    module_free((Module*)model);

    return pass;
}

static int test_linear_regression(void) {
    printf("  [Linear Regression] Linear(1,1), SGD lr=0.01, 500 epochs, y=2x+1\n");

    Linear* layer = nn_linear(1, 1, DTYPE_FLOAT32, DEVICE_CPU, true);
    Module* model = (Module*)layer;
    module_set_training(model, true);

    Parameter** params = NULL;
    int num_params     = 0;
    module_collect_parameters(model, &params, &num_params, true);
    Optimizer* optimizer = optim_sgd(params, num_params, 0.01f, 0.0f, 0.0f);
    if (!optimizer) {
        printf("    ERROR: failed to create optimizer\n");
        cml_free(params);
        module_free(model);
        return 0;
    }

#define LR_N 20
    float x_data[LR_N];
    float y_data[LR_N];
    for (int i = 0; i < LR_N; i++) {
        x_data[i] = (float)i / (float)(LR_N - 1) * 2.0f - 1.0f;
        y_data[i] = 2.0f * x_data[i] + 1.0f;
    }

    Tensor* X = cml_tensor_2d(x_data, LR_N, 1);
    Tensor* Y = cml_tensor_2d(y_data, LR_N, 1);
    if (X)
        tensor_realize(X);
    if (Y)
        tensor_realize(Y);

    float final_loss = INFINITY;

    for (int epoch = 0; epoch < 500; epoch++) {
        optimizer_zero_grad(optimizer);

        Tensor* out  = module_forward(model, X);
        Tensor* loss = tensor_mse_loss(out, Y);
        final_loss   = get_scalar(loss);

        tensor_backward(loss, NULL, false, false);
        optimizer_step(optimizer);

        if ((epoch + 1) % 100 == 0) {
            printf("    epoch %4d  loss = %.6f\n", epoch + 1, (double)final_loss);
        }

        tensor_free(loss);
        tensor_free(out);
        cml_ir_reset_global_context();
    }

    float learned_weight = 0.0f;
    float learned_bias   = 0.0f;

    Parameter* w = linear_get_weight(layer);
    Parameter* b = linear_get_bias(layer);
    if (w && w->tensor) {
        float* wd = (float*)tensor_data_ptr(w->tensor);
        if (wd)
            learned_weight = wd[0];
    }
    if (b && b->tensor) {
        float* bd = (float*)tensor_data_ptr(b->tensor);
        if (bd)
            learned_bias = bd[0];
    }

    printf("    learned weight = %.4f (target ~2.0), bias = %.4f (target ~1.0)\n",
           (double)learned_weight, (double)learned_bias);
    printf("    final loss = %.6f\n", (double)final_loss);

    int weight_ok = (fabsf(learned_weight - 2.0f) < 0.5f);
    int bias_ok   = (fabsf(learned_bias - 1.0f) < 0.5f);
    int loss_ok   = (final_loss < 0.1f);
    int pass      = weight_ok && bias_ok && loss_ok;

    printf("    weight close to 2.0 (tol 0.5): %s\n", weight_ok ? "PASS" : "FAIL");
    printf("    bias   close to 1.0 (tol 0.5): %s\n", bias_ok ? "PASS" : "FAIL");
    printf("    loss < 0.1:                     %s\n", loss_ok ? "PASS" : "FAIL");
    printf("    => %s\n", pass ? "PASS" : "FAIL");

    tensor_free(X);
    tensor_free(Y);
    optimizer_free(optimizer);
    cml_free(params);
    module_free(model);

    return pass;
}

static int test_conv2d_pattern(void) {
    printf("  [Conv2d Pattern] Linear(64,2), Adam lr=0.01, 500 epochs (flattened 8x8 input)\n");

    Sequential* model = nn_sequential();
    sequential_add(model, (Module*)nn_linear(1 * 8 * 8, 2, DTYPE_FLOAT32, DEVICE_CPU, true));
    module_set_training((Module*)model, true);

    Parameter** params = NULL;
    int num_params     = 0;
    module_collect_parameters((Module*)model, &params, &num_params, true);
    Optimizer* optimizer = optim_adam(params, num_params, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);
    if (!optimizer) {
        printf("    ERROR: failed to create optimizer\n");
        cml_free(params);
        module_free((Module*)model);
        return 0;
    }

#define CONV_N 8
#define CONV_H 8
#define CONV_W 8
#define CONV_PIXELS (CONV_H * CONV_W)

    float img_data[CONV_N * 1 * CONV_H * CONV_W];
    float target_data[CONV_N * 2];

    memset(img_data, 0, sizeof(img_data));
    memset(target_data, 0, sizeof(target_data));

    for (int n = 0; n < CONV_N; n++) {
        int cls    = n % 2;
        float* img = &img_data[n * CONV_PIXELS];

        if (cls == 0) {
            for (int r = 0; r < CONV_H / 2; r++)
                for (int c = 0; c < CONV_W; c++)
                    img[r * CONV_W + c] = 1.0f;
        } else {
            for (int r = 0; r < CONV_H; r++)
                for (int c = 0; c < CONV_W / 2; c++)
                    img[r * CONV_W + c] = 1.0f;
        }

        target_data[n * 2 + cls] = 1.0f;
    }

    int img_shape[]  = {CONV_N, 1 * CONV_H * CONV_W};
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* X = cml_tensor(img_data, img_shape, 2, &cfg);
    Tensor* Y = cml_tensor_2d(target_data, CONV_N, 2);
    if (X)
        tensor_realize(X);
    if (Y)
        tensor_realize(Y);

    float final_loss = INFINITY;

    for (int epoch = 0; epoch < 500; epoch++) {
        optimizer_zero_grad(optimizer);

        Tensor* out  = module_forward((Module*)model, X);
        Tensor* loss = tensor_mse_loss(out, Y);
        final_loss   = get_scalar(loss);

        tensor_backward(loss, NULL, false, false);
        optimizer_step(optimizer);

        if ((epoch + 1) % 100 == 0) {
            printf("    epoch %4d  loss = %.6f\n", epoch + 1, (double)final_loss);
        }

        tensor_free(loss);
        tensor_free(out);
        cml_ir_reset_global_context();
    }

    Tensor* preds    = module_forward((Module*)model, X);
    float* pred_data = (float*)tensor_data_ptr(preds);
    float* tgt_data  = (float*)tensor_data_ptr(Y);

    int correct = 0;
    if (pred_data && tgt_data) {
        for (int n = 0; n < CONV_N; n++) {
            int pred_cls = (pred_data[n * 2 + 1] > pred_data[n * 2 + 0]) ? 1 : 0;
            int true_cls = (tgt_data[n * 2 + 1] > tgt_data[n * 2 + 0]) ? 1 : 0;
            if (pred_cls == true_cls)
                correct++;
        }
    }

    float accuracy = (float)correct / (float)CONV_N;
    int pass       = (accuracy >= 0.75f);

    printf("    accuracy = %.2f (%d/%d)  (threshold >= 0.75) => %s\n", (double)accuracy, correct,
           CONV_N, pass ? "PASS" : "FAIL");

    tensor_free(preds);
    tensor_free(X);
    tensor_free(Y);
    optimizer_free(optimizer);
    cml_free(params);
    module_free((Module*)model);

    return pass;
}

/* A real convolutional network trained end to end on the bundled 1797-sample
 * 8x8 digits set (10 classes), with a held-out test split and a hard accuracy
 * threshold. Unlike the toy cases above, this exercises the full conv training
 * path (im2col decompose, pooling backward, sparse cross-entropy) on real,
 * noisy, multi-class data, so passing means the stack can actually fit a model
 * that generalizes, not just memorize a handful of points. */
static int test_digits_cnn(void) {
    printf("  [Digits CNN] Conv(1,8)-ReLU-Pool-Conv(8,16)-ReLU-Pool-Linear(64,10), "
           "Adam lr=0.01, 12 epochs\n");

    cml_enable_grad();

    int n = 0, feats = 0;
    const float* src_x = cml_builtin_digits_data(&n, &feats);
    const float* src_y = cml_builtin_digits_labels(&n);
    if (!src_x || !src_y || n != 1797 || feats != 64) {
        printf("    ERROR: digits dataset unavailable\n");
        return 0;
    }

    /* The dataset is laid out class by class, so shuffle before splitting or the
     * test fold would hold entire unseen classes. Fixed LCG => deterministic. */
    int* order = cml_malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++)
        order[i] = i;
    uint32_t rng = 777u;
    for (int i = n - 1; i > 0; i--) {
        rng      = rng * 1103515245u + 12345u;
        int j    = (int)((rng >> 8) % (uint32_t)(i + 1));
        int tmp  = order[i];
        order[i] = order[j];
        order[j] = tmp;
    }

    float* X = cml_malloc(sizeof(float) * n * feats); /* pixels scaled to [0,1] */
    float* Y = cml_malloc(sizeof(float) * n);         /* class indices as floats */
    for (int i = 0; i < n; i++) {
        const float* row = &src_x[order[i] * feats];
        for (int p = 0; p < feats; p++)
            X[i * feats + p] = row[p] / 16.0f;
        Y[i] = src_y[order[i]];
    }

    const int n_test  = 360;
    const int n_train = n - n_test;
    const int batch   = 128;

    Sequential* model = nn_sequential();
    sequential_add(model, (Module*)nn_conv2d(1, 8, 3, 1, 1, 1, true, DTYPE_FLOAT32, DEVICE_CPU));
    sequential_add(model, (Module*)nn_relu(false));
    sequential_add(model, (Module*)nn_maxpool2d(2, 2, 0, 1, false));
    sequential_add(model, (Module*)nn_conv2d(8, 16, 3, 1, 1, 1, true, DTYPE_FLOAT32, DEVICE_CPU));
    sequential_add(model, (Module*)nn_relu(false));
    sequential_add(model, (Module*)nn_maxpool2d(2, 2, 0, 1, false));
    sequential_add(model, (Module*)nn_flatten(1, -1));
    sequential_add(model, (Module*)nn_linear(16 * 2 * 2, 10, DTYPE_FLOAT32, DEVICE_CPU, true));
    module_set_training((Module*)model, true);

    Parameter** params = NULL;
    int num_params     = 0;
    module_collect_parameters((Module*)model, &params, &num_params, true);
    Optimizer* optimizer = optim_adam(params, num_params, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);
    if (!optimizer) {
        printf("    ERROR: failed to create optimizer\n");
        cml_free(order);
        cml_free(X);
        cml_free(Y);
        cml_free(params);
        module_free((Module*)model);
        return 0;
    }

    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

    for (int epoch = 0; epoch < 12; epoch++) {
        float epoch_loss = 0.0f;
        int steps        = 0;
        for (int start = 0; start < n_train; start += batch) {
            int b      = (start + batch <= n_train) ? batch : (n_train - start);
            int xs[4]  = {b, 1, 8, 8};
            Tensor* Xb = cml_tensor(&X[start * feats], xs, 4, &cfg);
            Tensor* Yb = cml_tensor_1d(&Y[start], b);

            optimizer_zero_grad(optimizer);
            Tensor* out  = module_forward((Module*)model, Xb);
            Tensor* loss = tensor_sparse_cross_entropy_loss(out, Yb);
            epoch_loss += get_scalar(loss);
            steps++;

            tensor_backward(loss, NULL, false, false);
            optimizer_step(optimizer);

            tensor_free(loss);
            tensor_free(out);
            tensor_free(Xb);
            tensor_free(Yb);
            cml_ir_reset_global_context();
        }
        if ((epoch + 1) % 3 == 0)
            printf("    epoch %2d  mean train loss = %.4f\n", epoch + 1,
                   (double)(epoch_loss / (float)steps));
    }

    module_set_training((Module*)model, false);
    int xs[4]      = {n_test, 1, 8, 8};
    Tensor* Xt     = cml_tensor(&X[n_train * feats], xs, 4, &cfg);
    Tensor* logits = module_forward((Module*)model, Xt);
    float* lp      = (float*)tensor_data_ptr(logits);
    int correct    = 0;
    if (lp) {
        for (int i = 0; i < n_test; i++) {
            int best    = 0;
            float bestv = lp[i * 10];
            for (int c = 1; c < 10; c++)
                if (lp[i * 10 + c] > bestv) {
                    bestv = lp[i * 10 + c];
                    best  = c;
                }
            if (best == (int)(Y[n_train + i] + 0.5f))
                correct++;
        }
    }
    float accuracy = (float)correct / (float)n_test;
    int pass       = (accuracy >= 0.85f);
    printf("    test accuracy = %.3f (%d/%d)  (threshold >= 0.85) => %s\n", (double)accuracy,
           correct, n_test, pass ? "PASS" : "FAIL");

    tensor_free(logits);
    tensor_free(Xt);
    optimizer_free(optimizer);
    cml_free(params);
    module_free((Module*)model);
    cml_free(order);
    cml_free(X);
    cml_free(Y);
    cml_ir_reset_global_context();

    return pass;
}

int main(void) {
    cml_init();
    cml_seed(42);

    printf("test_convergence\n\n");

    CHECK("XOR MLP convergence", test_xor_mlp());
    printf("\n");
    CHECK("Linear Regression convergence", test_linear_regression());
    printf("\n");
    CHECK("Conv2d pattern classification convergence", test_conv2d_pattern());
    printf("\n");
    CHECK("Digits CNN classification convergence", test_digits_cnn());
    printf("\n");

    cml_cleanup();
    return TEST_SUMMARY();
}
