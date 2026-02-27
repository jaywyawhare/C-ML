/**
 * Example 10: Embedding Layer
 *
 * Demonstrates nn_embedding for token lookup tables.
 * Creates a small vocabulary, looks up embeddings, and shows
 * how embeddings can be used as input to downstream layers.
 */
#include "cml.h"
#include <stdio.h>

int main(void) {
    cml_init();
    printf("Example 10: Embedding Layer\n\n");

    int vocab_size = 8, embed_dim = 4;

    Embedding* emb = cml_nn_embedding(vocab_size, embed_dim, -1, DTYPE_FLOAT32, DEVICE_CPU);
    printf("Embedding: vocab=%d, dim=%d\n", vocab_size, embed_dim);
    printf("Weight shape: [%d, %d]\n\n", vocab_size, embed_dim);

    // Show the embedding table
    printf("Embedding table (randomly initialized):\n");
    Tensor* weight = emb->weight->tensor;
    for (int i = 0; i < vocab_size; i++) {
        printf("  token %d: [", i);
        for (int j = 0; j < embed_dim; j++) {
            printf("%.3f", tensor_get_float(weight, i * embed_dim + j));
            if (j < embed_dim - 1) printf(", ");
        }
        printf("]\n");
    }

    // Feed embeddings into a classifier: embed -> linear -> softmax
    Sequential* classifier = cml_nn_sequential();
    cml_nn_sequential_add(classifier, (Module*)cml_nn_linear(embed_dim, 3, DTYPE_FLOAT32, DEVICE_CPU, true));

    printf("\nClassifier: embed(%d) -> linear(%d, 3)\n", embed_dim, embed_dim);
    cml_summary((Module*)classifier);

    // Simulate: look up token 2's embedding and classify
    printf("\nLooking up token 2 and classifying:\n");
    int tok_shape[] = {1, embed_dim};
    float tok_data[4];
    for (int j = 0; j < embed_dim; j++)
        tok_data[j] = tensor_get_float(weight, 2 * embed_dim + j);

    Tensor* tok_emb = cml_tensor(tok_data, tok_shape, 2, NULL);
    Tensor* logits = cml_nn_sequential_forward(classifier, tok_emb);
    printf("  Logits: [%.3f, %.3f, %.3f]\n",
           tensor_get_float(logits, 0),
           tensor_get_float(logits, 1),
           tensor_get_float(logits, 2));

    printf("\nEmbedding example complete.\n");
    cml_cleanup();
    return 0;
}
