#include "cml.h"
#include "zoo/zoo.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    cml_init();
    cml_seed(42);
    printf("BERT-Tiny Inference Example\n\n");

    BERTConfig bcfg = cml_zoo_bert_config_tiny();
    printf("Config: vocab=%d, layers=%d, heads=%d, hidden=%d\n",
           bcfg.vocab_size, bcfg.n_layer, bcfg.n_head, bcfg.hidden_size);
    fflush(stdout);

    CMLZooConfig cfg = cml_zoo_default_config();
    Module* bert = cml_zoo_bert_tiny(&cfg);
    if (!bert) { printf("Failed to create BERT\n"); return 1; }
    printf("BERT created.\n"); fflush(stdout);

    TensorConfig tcfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                         .has_dtype = true, .has_device = true};

    printf("\n%-6s  %-14s  %-10s\n", "Run", "Output Norm", "Shape");
    printf("------  --------------  ----------\n");

    for (int i = 1; i <= 3; i++) {
        int tok_shape[] = {1, 16};
        Tensor* tokens = tensor_ones(tok_shape, 2, &tcfg);

        Tensor* out = cml_nn_module_forward(bert, tokens);
        if (!out) { printf("Forward failed\n"); break; }

        float* data = (float*)tensor_data_ptr(out);
        float norm = 0.0f;
        if (data) {
            for (size_t j = 0; j < out->numel && j < 1000; j++)
                norm += data[j] * data[j];
            norm = sqrtf(norm);
        }

        printf("%-6d  %-14.4f  [%d,%d]\n", i, norm,
               out->shape[0], out->shape[1]);

        tensor_free(tokens);
    }

    module_free(bert);
    printf("\nDone.\n");
    cml_cleanup();
    return 0;
}
