#include "cml.h"
#include "zoo/zoo.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    cml_init();
    cml_seed(42);
    printf("U-Net Segmentation Example\n\n");

    int num_classes = 2;
    int epochs = 5;

    CMLUNetConfig ucfg = cml_zoo_unet_config_default();
    printf("Config: in_ch=%d, classes=%d, depth=%d, base_filters=%d\n",
           ucfg.in_channels, ucfg.num_classes, ucfg.depth, ucfg.base_filters);

    Module* unet = cml_zoo_unet_create(&ucfg, DTYPE_FLOAT32, DEVICE_CPU);
    if (!unet) { printf("Failed to create U-Net\n"); return 1; }

    TensorConfig tcfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                         .has_dtype = true, .has_device = true};

    printf("Task:       Binary segmentation (32x32)\n\n");
    printf("%-6s  %-12s  %-12s  %-14s\n", "Epoch", "MSE Loss", "Pixel Acc", "Output Shape");
    printf("------  ------------  ------------  --------------\n");

    for (int epoch = 1; epoch <= epochs; epoch++) {
        int img_shape[] = {1, 1, 32, 32};
        Tensor* image = tensor_randn(img_shape, 4, &tcfg);

        /* Random binary mask [1, num_classes, 32, 32] */
        int msk_shape[] = {1, num_classes, 32, 32};
        Tensor* mask = tensor_zeros(msk_shape, 4, &tcfg);
        float* msk_data = (float*)tensor_data_ptr(mask);
        for (int i = 0; i < 32 * 32; i++) {
            int cls = rand() % num_classes;
            msk_data[cls * 32 * 32 + i] = 1.0f;
        }

        Tensor* pred = cml_nn_module_forward(unet, image);
        if (!pred) { printf("Forward failed\n"); break; }

        Tensor* loss = cml_nn_mse_loss(pred, mask);
        float loss_val = tensor_get_float(loss, 0);

        /* Pixel accuracy */
        float* pred_data = (float*)tensor_data_ptr(pred);
        int correct = 0;
        for (int i = 0; i < 32 * 32; i++) {
            int pred_cls = 0, true_cls = 0;
            float best_p = pred_data[i], best_t = msk_data[i];
            for (int c = 1; c < num_classes; c++) {
                float pv = pred_data[c * 32 * 32 + i];
                float tv = msk_data[c * 32 * 32 + i];
                if (pv > best_p) { best_p = pv; pred_cls = c; }
                if (tv > best_t) { best_t = tv; true_cls = c; }
            }
            if (pred_cls == true_cls) correct++;
        }
        float pixel_acc = (float)correct / (32 * 32) * 100.0f;

        char shape_str[64];
        snprintf(shape_str, sizeof(shape_str), "[%d,%d,%d,%d]",
                 pred->shape[0], pred->shape[1], pred->shape[2], pred->shape[3]);

        printf("%-6d  %-12.6f  %-11.1f%%  %-14s\n",
               epoch, loss_val, pixel_acc, shape_str);

        tensor_free(image);
        tensor_free(mask);
    }

    module_free(unet);
    printf("\nDone.\n");
    cml_cleanup();
    return 0;
}
