#include "cml.h"
#include "zoo/zoo.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void) {
    cml_init();
    cml_seed(42);
    printf("ResNet-18 Training Example\n\n");

    int num_classes = 10;
    int batch_size = 1;
    int epochs = 3;
    float lr = 0.01f;

    CMLZooConfig cfg = cml_zoo_default_config();
    cfg.num_classes = num_classes;

    Module* model = cml_zoo_resnet18(&cfg);
    if (!model) { printf("Failed to create ResNet-18\n"); return 1; }

    Optimizer* opt = cml_optim_sgd_for_model(model, lr, 0.9f, 1e-4f);
    if (!opt) { printf("Failed to create optimizer\n"); return 1; }

    TensorConfig tcfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                         .has_dtype = true, .has_device = true};

    printf("Model:      ResNet-18\n");
    printf("Classes:    %d\n", num_classes);
    printf("Batch size: %d\n", batch_size);
    printf("Optimizer:  SGD (lr=%.4f, momentum=0.9)\n", lr);
    printf("Epochs:     %d\n\n", epochs);
    printf("%-6s  %-12s  %-10s  %-8s\n", "Epoch", "Loss", "Accuracy", "LR");
    printf("------  ------------  ----------  --------\n");

    for (int epoch = 1; epoch <= epochs; epoch++) {
        int img_shape[] = {batch_size, 3, 32, 32};
        Tensor* images = tensor_randn(img_shape, 4, &tcfg);

        /* Random target labels */
        int lbl_shape[] = {batch_size};
        Tensor* labels = tensor_zeros(lbl_shape, 1, &tcfg);
        float* lbl_data = (float*)tensor_data_ptr(labels);
        for (int i = 0; i < batch_size; i++)
            lbl_data[i] = (float)(rand() % num_classes);

        /* Forward */
        Tensor* logits = cml_nn_module_forward(model, images);
        if (!logits) { printf("Forward failed at epoch %d\n", epoch); break; }

        /* Loss */
        Tensor* loss = cml_nn_cross_entropy_loss(logits, labels);

        /* Backward + step */
        cml_optim_zero_grad(opt);
        cml_backward(loss, NULL, false, false);
        cml_optim_step(opt);

        /* Accuracy */
        float* logit_data = (float*)tensor_data_ptr(logits);
        int correct = 0;
        for (int i = 0; i < batch_size; i++) {
            int pred = 0;
            float best = logit_data[i * num_classes];
            for (int c = 1; c < num_classes; c++) {
                float v = logit_data[i * num_classes + c];
                if (v > best) { best = v; pred = c; }
            }
            if (pred == (int)lbl_data[i]) correct++;
        }
        float acc = (float)correct / batch_size * 100.0f;
        float loss_val = tensor_get_float(loss, 0);
        float cur_lr = optimizer_get_group_lr(opt, 0);

        printf("%-6d  %-12.6f  %-9.1f%%  %.6f\n", epoch, loss_val, acc, cur_lr);

        tensor_free(images);
        tensor_free(labels);
    }

    printf("\nDone.\n");
    cml_cleanup();
    return 0;
}
