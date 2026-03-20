#include "cml.h"
#include "zoo/zoo.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    cml_init();
    cml_seed(42);
    printf("T5-Small Seq2Seq Example\n\n");

    int seq_len = 8;
    int epochs = 5;

    T5Config t5cfg = cml_zoo_t5_config_small();
    printf("Config: vocab=%d, layers=%d, heads=%d, d_model=%d, d_ff=%d\n",
           t5cfg.vocab_size, t5cfg.n_layer, t5cfg.n_head, t5cfg.d_model, t5cfg.d_ff);

    Module* t5 = cml_zoo_t5_create(&t5cfg, DTYPE_FLOAT32, DEVICE_CPU);
    if (!t5) { printf("Failed to create T5\n"); return 1; }

    TensorConfig tcfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                         .has_dtype = true, .has_device = true};

    printf("Task:       Encoder output + loss\n");
    printf("Seq length: %d\n\n", seq_len);
    printf("%-6s  %-12s  %-14s  %-14s\n", "Epoch", "MSE Loss", "Output Norm", "Output Shape");
    printf("------  ------------  --------------  --------------\n");

    for (int epoch = 1; epoch <= epochs; epoch++) {
        int shape[] = {1, seq_len};
        Tensor* src = tensor_ones(shape, 2, &tcfg);
        float* src_data = (float*)tensor_data_ptr(src);
        for (int i = 0; i < seq_len; i++)
            src_data[i] = (float)(rand() % 500 + 1);

        Tensor* encoded = t5_encode(t5, src);
        if (!encoded) { printf("Encode failed\n"); break; }

        /* Output norm */
        float* e_data = (float*)tensor_data_ptr(encoded);
        float norm = 0.0f;
        for (size_t i = 0; i < encoded->numel && i < 1000; i++)
            norm += e_data[i] * e_data[i];
        norm = sqrtf(norm);

        /* MSE loss */
        Tensor* target = tensor_zeros(encoded->shape, encoded->ndim, &tcfg);
        Tensor* loss = cml_nn_mse_loss(encoded, target);
        float loss_val = tensor_get_float(loss, 0);

        char shape_str[64];
        snprintf(shape_str, sizeof(shape_str), "[%d,%d,%d]",
                 encoded->shape[0], encoded->shape[1], encoded->shape[2]);

        printf("%-6d  %-12.6f  %-14.4f  %-14s\n", epoch, loss_val, norm, shape_str);

        tensor_free(src);
        tensor_free(target);
    }

    module_free(t5);
    printf("\nDone.\n");
    cml_cleanup();
    return 0;
}
