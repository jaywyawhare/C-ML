#include "cml.h"
#include "zoo/zoo.h"
#include <stdio.h>

int main(void) {
    cml_init();
    cml_seed(42);
    printf("RNN-Transducer Example\n\n");

    CMLRNNTConfig rcfg = cml_zoo_rnnt_config_default();
    printf("RNN-T config:\n");
    printf("  input_features=%d, vocab_size=%d\n",
           rcfg.input_features, rcfg.vocab_size);
    printf("  encoder: layers=%d, dim=%d\n", rcfg.encoder_layers, rcfg.encoder_dim);
    printf("  predictor: layers=%d, dim=%d\n", rcfg.pred_layers, rcfg.pred_dim);
    printf("  joint_dim=%d\n", rcfg.joint_dim);

    Module* model = cml_zoo_rnnt_create(&rcfg, DTYPE_FLOAT32, DEVICE_CPU);
    if (!model) { printf("Failed to create RNN-T\n"); return 1; }

    printf("\nRNN-T created successfully.\n");
    printf("(Streaming speech recognition model.)\n");

    module_free(model);
    printf("\nDone.\n");
    cml_cleanup();
    return 0;
}
