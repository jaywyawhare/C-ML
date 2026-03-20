#include "cml.h"
#include "zoo/zoo.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    cml_init();
    cml_seed(42);
    printf("GPT-2 Language Modeling Example\n\n");

    int seq_len = 8;
    int epochs = 3;

    /* Use a smaller config for CPU demo to avoid numerical overflow */
    GPT2Config gcfg = {
        .vocab_size = 1000,
        .n_layer    = 2,
        .n_head     = 4,
        .n_embd     = 64,
        .block_size = 128
    };
    printf("Config: vocab=%d, layers=%d, heads=%d, embed=%d\n",
           gcfg.vocab_size, gcfg.n_layer, gcfg.n_head, gcfg.n_embd);

    Module* gpt2 = cml_zoo_gpt2_create(&gcfg, DTYPE_FLOAT32, DEVICE_CPU);
    if (!gpt2) { printf("Failed to create GPT-2\n"); return 1; }

    TensorConfig tcfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                         .has_dtype = true, .has_device = true};

    printf("Task:       Next-token prediction inference\n");
    printf("Seq length: %d\n\n", seq_len);
    printf("%-6s  %-14s  %-14s\n", "Epoch", "Output Norm", "Shape");
    printf("------  --------------  --------------\n");

    for (int epoch = 1; epoch <= epochs; epoch++) {
        int shape[] = {1, seq_len};
        Tensor* tokens = tensor_ones(shape, 2, &tcfg);
        float* tok_data = (float*)tensor_data_ptr(tokens);
        for (int i = 0; i < seq_len; i++)
            tok_data[i] = (float)(rand() % (gcfg.vocab_size - 1) + 1);

        Tensor* logits = cml_nn_module_forward(gpt2, tokens);
        if (!logits) { printf("Forward failed\n"); break; }

        float* l_data = (float*)tensor_data_ptr(logits);
        float norm = 0.0f;
        if (l_data) {
            size_t n = logits->numel < 10000 ? logits->numel : 10000;
            for (size_t i = 0; i < n; i++)
                norm += l_data[i] * l_data[i];
            norm = sqrtf(norm);
        }

        printf("%-6d  %-14.4f  [%d,%d]\n", epoch, norm,
               logits->shape[0], logits->shape[1]);

        tensor_free(tokens);
    }

    module_free(gpt2);
    printf("\nDone.\n");
    cml_cleanup();
    return 0;
}
