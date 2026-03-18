#include "cml.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    cml_init();
    srand(42);
    printf("Example 13: Transformer Encoder\n\n");

    int batch = 2, seq_len = 4, d_model = 8, nhead = 2, dim_ff = 16;

    int input_shape[] = {batch, seq_len, d_model};
    float input_data[2 * 4 * 8];
    for (int i = 0; i < 2 * 4 * 8; i++)
        input_data[i] = (float)rand() / (float)RAND_MAX - 0.5f;

    Tensor* X = cml_tensor(input_data, input_shape, 3, NULL);

    MultiHeadAttention* mha = cml_nn_multihead_attention(d_model, nhead, 0.0f,
                                                          DTYPE_FLOAT32, DEVICE_CPU);
    printf("MultiHeadAttention: embed_dim=%d, num_heads=%d, head_dim=%d\n",
           mha->embed_dim, mha->num_heads, mha->head_dim);

    TransformerEncoderLayer* enc_layer = cml_nn_transformer_encoder_layer(
        d_model, nhead, dim_ff, 0.0f, DTYPE_FLOAT32, DEVICE_CPU);
    printf("TransformerEncoderLayer: d_model=%d, nhead=%d, dim_ff=%d\n",
           d_model, nhead, dim_ff);

    cml_summary((Module*)enc_layer);

    Tensor* out = module_forward((Module*)enc_layer, X);
    tensor_ensure_executed(out);

    printf("\nInput (first 8 values): ");
    for (int i = 0; i < 8; i++)
        printf("%.3f ", input_data[i]);
    printf("...\n");

    printf("Output (first 8 values): ");
    for (int i = 0; i < 8; i++)
        printf("%.3f ", tensor_get_float(out, i));
    printf("...\n");

    printf("\nStacking 2 encoder layers:\n");
    Sequential* encoder = cml_nn_sequential();
    cml_nn_sequential_add(encoder, (Module*)cml_nn_transformer_encoder_layer(
        d_model, nhead, dim_ff, 0.0f, DTYPE_FLOAT32, DEVICE_CPU));
    cml_nn_sequential_add(encoder, (Module*)cml_nn_transformer_encoder_layer(
        d_model, nhead, dim_ff, 0.0f, DTYPE_FLOAT32, DEVICE_CPU));
    cml_summary((Module*)encoder);

    Tensor* enc_out = cml_nn_sequential_forward(encoder, X);
    tensor_ensure_executed(enc_out);
    printf("Encoded output (first 8 values): ");
    for (int i = 0; i < 8; i++)
        printf("%.3f ", tensor_get_float(enc_out, i));
    printf("...\n");

    printf("\nTransformer example complete.\n");
    cml_cleanup();
    return 0;
}
