/**
 * @file test_nn_layers.c
 * @brief Unit tests for all neural network layer types
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "cml.h"

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    printf("  Testing: %s... ", #name); \
    tests_run++; \
    if (test_##name()) { \
        printf("PASS\n"); \
        tests_passed++; \
    } else { \
        printf("FAIL\n"); \
    } \
} while(0)

static int test_linear(void) {
    Linear* layer = cml_nn_linear(4, 8, DTYPE_FLOAT32, DEVICE_CPU, true);
    if (!layer) return 0;
    /* weight: 8*4=32, bias: 8, total: 40 */
    Module* m = (Module*)layer;
    int ok = (m->num_parameters == 2);
    printf("(params=%d) ", m->num_parameters);
    module_free(m);
    return ok;
}

static int test_relu(void) {
    ReLU* layer = cml_nn_relu(false);
    if (!layer) return 0;
    Module* m = (Module*)layer;
    int ok = (m->num_parameters == 0);
    module_free(m);
    return ok;
}

static int test_sigmoid(void) {
    Sigmoid* layer = cml_nn_sigmoid();
    if (!layer) return 0;
    module_free((Module*)layer);
    return 1;
}

static int test_tanh_layer(void) {
    Tanh* layer = cml_nn_tanh();
    if (!layer) return 0;
    module_free((Module*)layer);
    return 1;
}

static int test_leaky_relu(void) {
    LeakyReLU* layer = cml_nn_leaky_relu(0.01f, false);
    if (!layer) return 0;
    module_free((Module*)layer);
    return 1;
}

static int test_dropout(void) {
    Dropout* layer = cml_nn_dropout(0.5f, false);
    if (!layer) return 0;
    module_free((Module*)layer);
    return 1;
}

static int test_conv2d(void) {
    Conv2d* layer = cml_nn_conv2d(3, 16, 3, 1, 1, 1, true, DTYPE_FLOAT32, DEVICE_CPU);
    if (!layer) return 0;
    Module* m = (Module*)layer;
    printf("(params=%d) ", m->num_parameters);
    module_free(m);
    return 1;
}

static int test_conv1d(void) {
    Conv1d* layer = cml_nn_conv1d(3, 16, 3, 1, 1, 1, true, DTYPE_FLOAT32, DEVICE_CPU);
    if (!layer) return 0;
    Module* m = (Module*)layer;
    printf("(params=%d) ", m->num_parameters);
    module_free(m);
    return 1;
}

static int test_conv3d(void) {
    Conv3d* layer = cml_nn_conv3d(3, 16, 3, 1, 1, 1, true, DTYPE_FLOAT32, DEVICE_CPU);
    if (!layer) return 0;
    Module* m = (Module*)layer;
    printf("(params=%d) ", m->num_parameters);
    module_free(m);
    return 1;
}

static int test_batchnorm2d(void) {
    BatchNorm2d* layer = cml_nn_batchnorm2d(16, 1e-5f, 0.1f, true, true, DTYPE_FLOAT32, DEVICE_CPU);
    if (!layer) return 0;
    Module* m = (Module*)layer;
    printf("(params=%d) ", m->num_parameters);
    module_free(m);
    return 1;
}

static int test_layernorm(void) {
    LayerNorm* layer = cml_nn_layernorm(64, 1e-5f, true, DTYPE_FLOAT32, DEVICE_CPU);
    if (!layer) return 0;
    Module* m = (Module*)layer;
    printf("(params=%d) ", m->num_parameters);
    module_free(m);
    return 1;
}

static int test_groupnorm(void) {
    GroupNorm* layer = cml_nn_groupnorm(4, 16, 1e-5f, true, DTYPE_FLOAT32, DEVICE_CPU);
    if (!layer) return 0;
    Module* m = (Module*)layer;
    printf("(params=%d) ", m->num_parameters);
    module_free(m);
    return 1;
}

static int test_maxpool2d(void) {
    MaxPool2d* layer = cml_nn_maxpool2d(2, 2, 0, 1, false);
    if (!layer) return 0;
    module_free((Module*)layer);
    return 1;
}

static int test_avgpool2d(void) {
    AvgPool2d* layer = cml_nn_avgpool2d(2, 2, 0, false, true);
    if (!layer) return 0;
    module_free((Module*)layer);
    return 1;
}

static int test_embedding(void) {
    Embedding* layer = cml_nn_embedding(1000, 64, -1, DTYPE_FLOAT32, DEVICE_CPU);
    if (!layer) return 0;
    Module* m = (Module*)layer;
    printf("(params=%d) ", m->num_parameters);
    module_free(m);
    return 1;
}

static int test_rnn_cell(void) {
    RNNCell* cell = cml_nn_rnn_cell(32, 64, true, DTYPE_FLOAT32, DEVICE_CPU);
    if (!cell) return 0;
    Module* m = (Module*)cell;
    printf("(params=%d) ", m->num_parameters);
    module_free(m);
    return 1;
}

static int test_lstm_cell(void) {
    LSTMCell* cell = cml_nn_lstm_cell(32, 64, true, DTYPE_FLOAT32, DEVICE_CPU);
    if (!cell) return 0;
    Module* m = (Module*)cell;
    printf("(params=%d) ", m->num_parameters);
    module_free(m);
    return 1;
}

static int test_gru_cell(void) {
    GRUCell* cell = cml_nn_gru_cell(32, 64, true, DTYPE_FLOAT32, DEVICE_CPU);
    if (!cell) return 0;
    Module* m = (Module*)cell;
    printf("(params=%d) ", m->num_parameters);
    module_free(m);
    return 1;
}

static int test_multihead_attention(void) {
    MultiHeadAttention* mha = cml_nn_multihead_attention(64, 8, 0.1f, DTYPE_FLOAT32, DEVICE_CPU);
    if (!mha) return 0;
    Module* m = (Module*)mha;
    printf("(params=%d) ", m->num_parameters);
    module_free(m);
    return 1;
}

static int test_transformer_encoder_layer(void) {
    TransformerEncoderLayer* layer = cml_nn_transformer_encoder_layer(64, 8, 256, 0.1f,
                                                                       DTYPE_FLOAT32, DEVICE_CPU);
    if (!layer) return 0;
    Module* m = (Module*)layer;
    printf("(params=%d) ", m->num_parameters);
    module_free(m);
    return 1;
}

static int test_module_list(void) {
    ModuleList* list = cml_nn_module_list();
    if (!list) return 0;
    int ret = module_list_append(list, (Module*)cml_nn_relu(false));
    if (ret != 0) return 0;
    ret = module_list_append(list, (Module*)cml_nn_sigmoid());
    if (ret != 0) return 0;
    int len = module_list_length(list);
    printf("(len=%d) ", len);
    module_free((Module*)list);
    return (len == 2);
}

static int test_module_dict(void) {
    ModuleDict* dict = cml_nn_module_dict();
    if (!dict) return 0;
    int ret = module_dict_add(dict, "relu", (Module*)cml_nn_relu(false));
    if (ret != 0) return 0;
    Module* got = module_dict_get(dict, "relu");
    printf("(found=%s) ", got ? "yes" : "no");
    module_free((Module*)dict);
    return (got != NULL);
}

static int test_sequential_forward(void) {
    Sequential* seq = cml_nn_sequential();
    if (!seq) return 0;
    sequential_add(seq, (Module*)cml_nn_linear(4, 8, DTYPE_FLOAT32, DEVICE_CPU, true));
    sequential_add(seq, (Module*)cml_nn_relu(false));
    sequential_add(seq, (Module*)cml_nn_linear(8, 2, DTYPE_FLOAT32, DEVICE_CPU, true));

    /* Forward with batch=1, features=4 */
    Tensor* input = cml_ones_2d(1, 4);
    if (!input) { module_free((Module*)seq); return 0; }

    Tensor* output = module_forward((Module*)seq, input);
    if (!output) { tensor_free(input); module_free((Module*)seq); return 0; }

    int ok = (output->ndim == 2 && output->shape[0] == 1 && output->shape[1] == 2);
    printf("(out_shape=[%d,%d]) ", output->shape[0], output->shape[1]);

    tensor_free(input);
    tensor_free(output);
    module_free((Module*)seq);
    return ok;
}

int main(void) {
    cml_init();

    printf("\n=== Neural Network Layers Unit Tests ===\n\n");

    printf("Core Layers:\n");
    TEST(linear);
    TEST(sequential_forward);

    printf("\nActivations:\n");
    TEST(relu);
    TEST(sigmoid);
    TEST(tanh_layer);
    TEST(leaky_relu);

    printf("\nRegularization:\n");
    TEST(dropout);

    printf("\nConvolution:\n");
    TEST(conv2d);
    TEST(conv1d);
    TEST(conv3d);

    printf("\nNormalization:\n");
    TEST(batchnorm2d);
    TEST(layernorm);
    TEST(groupnorm);

    printf("\nPooling:\n");
    TEST(maxpool2d);
    TEST(avgpool2d);

    printf("\nEmbedding:\n");
    TEST(embedding);

    printf("\nRNN Cells:\n");
    TEST(rnn_cell);
    TEST(lstm_cell);
    TEST(gru_cell);

    printf("\nTransformer:\n");
    TEST(multihead_attention);
    TEST(transformer_encoder_layer);

    printf("\nContainers:\n");
    TEST(module_list);
    TEST(module_dict);

    printf("\n=========================================\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_run);
    printf("=========================================\n\n");

    return (tests_passed == tests_run) ? 0 : 1;
}
