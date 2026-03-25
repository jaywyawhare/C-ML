#include "cml.h"
#include <stdio.h>
#include <stdlib.h>

#define N_EPOCHS 100

int main(void) {
    cml_init();
    printf("Example 12: Multi-Head Architecture (Wine)\n\n");

    Dataset* ds = cml_dataset_load("wine");
    if (!ds) { printf("Failed to load wine dataset\n"); return 1; }

    int n = ds->num_samples;
    int nf = ds->input_size;

    float* raw_y = malloc(sizeof(float) * n);
    for (int i = 0; i < n; i++)
        raw_y[i] = tensor_get_float(ds->y, i);

    float* reg_target = malloc(sizeof(float) * n);
    for (int i = 0; i < n; i++)
        reg_target[i] = tensor_get_float(ds->X, i * nf + 0);

    dataset_normalize(ds, "minmax");

    float* cls_target = malloc(sizeof(float) * n);
    for (int i = 0; i < n; i++)
        cls_target[i] = (raw_y[i] == 1.0f) ? 1.0f : 0.0f;

    float reg_min = reg_target[0], reg_max = reg_target[0];
    for (int i = 1; i < n; i++) {
        if (reg_target[i] < reg_min) reg_min = reg_target[i];
        if (reg_target[i] > reg_max) reg_max = reg_target[i];
    }
    for (int i = 0; i < n; i++)
        reg_target[i] = (reg_target[i] - reg_min) / (reg_max - reg_min + 1e-8f);

    int ycls_shape[] = {n, 1};
    int yreg_shape[] = {n, 1};
    Tensor* X     = ds->X;
    Tensor* Y_cls = cml_tensor(cls_target, ycls_shape, 2, NULL);
    Tensor* Y_reg = cml_tensor(reg_target, yreg_shape, 2, NULL);

    printf("Samples: %d, Features: %d\n", n, nf);
    printf("Task A: Wine type classification (binary)\n");
    printf("Task B: Alcohol content regression\n\n");

    Sequential* backbone = cml_nn_sequential();
    cml_nn_sequential_add(backbone, (Module*)cml_nn_linear(nf, 16, DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(backbone, (Module*)cml_nn_relu(false));
    cml_nn_sequential_add(backbone, (Module*)cml_nn_linear(16, 8, DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(backbone, (Module*)cml_nn_relu(false));

    Sequential* head_cls = cml_nn_sequential();
    cml_nn_sequential_add(head_cls, (Module*)cml_nn_linear(8, 1, DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(head_cls, (Module*)cml_nn_sigmoid());

    Sequential* head_reg = cml_nn_sequential();
    cml_nn_sequential_add(head_reg, (Module*)cml_nn_linear(8, 1, DTYPE_FLOAT32, DEVICE_CPU, true));

    cml_summary((Module*)backbone);

    Optimizer* opt_bb  = cml_optim_adam_for_model((Module*)backbone, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);
    Optimizer* opt_cls = cml_optim_adam_for_model((Module*)head_cls, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);
    Optimizer* opt_reg = cml_optim_adam_for_model((Module*)head_reg, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);

    for (int epoch = 1; epoch <= N_EPOCHS; epoch++) {
        Tensor* features = cml_nn_sequential_forward(backbone, X);

        Tensor* pred_cls = cml_nn_sequential_forward(head_cls, features);
        Tensor* loss_cls = cml_nn_bce_loss(pred_cls, Y_cls);

        Tensor* pred_reg = cml_nn_sequential_forward(head_reg, features);
        Tensor* loss_reg = cml_nn_mse_loss(pred_reg, Y_reg);

        Tensor* loss = cml_add(loss_cls, loss_reg);

        cml_optim_zero_grad(opt_bb);
        cml_optim_zero_grad(opt_cls);
        cml_optim_zero_grad(opt_reg);
        cml_backward(loss, NULL, false, false);
        cml_optim_step(opt_bb);
        cml_optim_step(opt_cls);
        cml_optim_step(opt_reg);
        cml_reset_ir_context();

        if (epoch % 20 == 0)
            printf("Epoch %3d  cls_loss: %.4f  reg_loss: %.4f\n",
                   epoch, tensor_get_float(loss_cls, 0), tensor_get_float(loss_reg, 0));
    }

    Tensor* features = cml_nn_sequential_forward(backbone, X);
    Tensor* pred_cls = cml_nn_sequential_forward(head_cls, features);
    Tensor* pred_reg = cml_nn_sequential_forward(head_reg, features);

    printf("\nSample predictions (first 8):\n");
    for (int i = 0; i < 8; i++) {
        float pc = tensor_get_float(pred_cls, i);
        float pr = tensor_get_float(pred_reg, i);
        int cls = pc > 0.5f ? 1 : 0;
        printf("  wine %d: cls pred=%d true=%.0f | reg pred=%.2f true=%.2f %s\n",
               i, cls, cls_target[i], pr, reg_target[i],
               cls == (int)cls_target[i] ? "OK" : "WRONG");
    }

    int total_correct = 0;
    for (int i = 0; i < n; i++) {
        float pc = tensor_get_float(pred_cls, i);
        total_correct += ((pc > 0.5f ? 1 : 0) == (int)cls_target[i]);
    }
    printf("Classification accuracy: %d/%d (%.0f%%)\n",
           total_correct, n, total_correct / (float)n * 100);

    free(raw_y);
    free(reg_target);
    free(cls_target);
    cml_cleanup();
    return 0;
}
