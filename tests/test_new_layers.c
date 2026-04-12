#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "cml.h"
#include "nn/layers.h"
#include "autograd/loss_functions.h"
#include "autograd/amp.h"
#include "tensor/sparse_tensor.h"

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  %-40s ", #name); \
    fflush(stdout); \
    if (test_##name()) { \
        tests_passed++; \
        printf("[PASS]\n"); \
    } else { \
        printf("[FAIL]\n"); \
    } \
} while(0)

static int test_rnn_multi_layer(void) {
    RNN* rnn = nn_rnn(10, 20, 2, false, false, 0.0f, true, DTYPE_FLOAT32, DEVICE_CPU);
    if (!rnn) return 0;
    if (rnn->num_layers != 2) { module_free((Module*)rnn); return 0; }
    if (rnn->bidirectional) { module_free((Module*)rnn); return 0; }
    module_free((Module*)rnn);
    return 1;
}

static int test_rnn_bidirectional(void) {
    RNN* rnn = nn_rnn(10, 20, 1, true, false, 0.0f, true, DTYPE_FLOAT32, DEVICE_CPU);
    if (!rnn) return 0;
    if (!rnn->bidirectional) { module_free((Module*)rnn); return 0; }
    if (rnn->num_directions != 2) { module_free((Module*)rnn); return 0; }
    module_free((Module*)rnn);
    return 1;
}

static int test_lstm_multi_layer(void) {
    LSTM* lstm = nn_lstm(10, 20, 3, false, false, 0.0f, true, DTYPE_FLOAT32, DEVICE_CPU);
    if (!lstm) return 0;
    if (lstm->num_layers != 3) { module_free((Module*)lstm); return 0; }
    module_free((Module*)lstm);
    return 1;
}

static int test_gru_multi_layer(void) {
    GRU* gru = nn_gru(10, 20, 2, true, false, 0.0f, true, DTYPE_FLOAT32, DEVICE_CPU);
    if (!gru) return 0;
    if (gru->num_layers != 2) { module_free((Module*)gru); return 0; }
    if (!gru->bidirectional) { module_free((Module*)gru); return 0; }
    module_free((Module*)gru);
    return 1;
}

static int test_conv_transpose2d(void) {
    ConvTranspose2d* ct = nn_conv_transpose2d(16, 8, 3, 2, 1, 1, true, DTYPE_FLOAT32, DEVICE_CPU);
    if (!ct) return 0;
    if (ct->in_channels != 16 || ct->out_channels != 8) { module_free((Module*)ct); return 0; }
    if (ct->kernel_size[0] != 3 || ct->stride[0] != 2) { module_free((Module*)ct); return 0; }
    module_free((Module*)ct);
    return 1;
}

static int test_conv_transpose3d(void) {
    ConvTranspose3d* ct = nn_conv_transpose3d(16, 8, 3, 2, 1, 1, true, DTYPE_FLOAT32, DEVICE_CPU);
    if (!ct) return 0;
    if (ct->in_channels != 16 || ct->out_channels != 8) { module_free((Module*)ct); return 0; }
    if (ct->kernel_size[0] != 3) { module_free((Module*)ct); return 0; }
    module_free((Module*)ct);
    return 1;
}

static int test_upsample_nearest(void) {
    Upsample* up = nn_upsample(2.0f, NULL, 0, UPSAMPLE_NEAREST, false);
    if (!up) return 0;
    if (up->scale_factor != 2.0f) { module_free((Module*)up); return 0; }
    if (up->mode != UPSAMPLE_NEAREST) { module_free((Module*)up); return 0; }
    module_free((Module*)up);
    return 1;
}

static int test_upsample_bilinear(void) {
    int out_size[] = {8, 8};
    Upsample* up = nn_upsample(0.0f, out_size, 2, UPSAMPLE_BILINEAR, true);
    if (!up) return 0;
    if (up->output_size[0] != 8 || up->output_size[1] != 8) { module_free((Module*)up); return 0; }
    if (up->mode != UPSAMPLE_BILINEAR) { module_free((Module*)up); return 0; }
    if (!up->align_corners) { module_free((Module*)up); return 0; }
    module_free((Module*)up);
    return 1;
}

static int test_pixel_shuffle(void) {
    PixelShuffle* ps = nn_pixel_shuffle(2);
    if (!ps) return 0;
    if (ps->upscale_factor != 2) { module_free((Module*)ps); return 0; }
    module_free((Module*)ps);
    return 1;
}

static int test_pixel_unshuffle(void) {
    PixelUnshuffle* pu = nn_pixel_unshuffle(2);
    if (!pu) return 0;
    if (pu->downscale_factor != 2) { module_free((Module*)pu); return 0; }
    module_free((Module*)pu);
    return 1;
}

static int test_sparse_roundtrip(void) {
    float data[] = {1.0f, 0.0f, 0.0f, 2.0f};
    Tensor* dense = tensor_from_array_2d(data, 2, 2);
    if (!dense) return 0;

    SparseCOOData* sparse = sparse_from_dense(dense);
    if (!sparse) { tensor_free(dense); return 0; }
    if (sparse->nnz != 2) { sparse_free(sparse); tensor_free(dense); return 0; }

    Tensor* recovered = sparse_to_dense(sparse, NULL);
    if (!recovered) { sparse_free(sparse); tensor_free(dense); return 0; }

    sparse_free(sparse);
    tensor_free(recovered);
    tensor_free(dense);
    return 1;
}

static int test_sparse_matmul(void) {
    float data_a[] = {1.0f, 0.0f, 0.0f, 2.0f};
    Tensor* dense_a = tensor_from_array_2d(data_a, 2, 2);
    if (!dense_a) return 0;

    SparseCOOData* sparse_a = sparse_from_dense(dense_a);
    if (!sparse_a) { tensor_free(dense_a); return 0; }

    float data_b[] = {3.0f, 4.0f, 5.0f, 6.0f};
    Tensor* dense_b = tensor_from_array_2d(data_b, 2, 2);
    if (!dense_b) { sparse_free(sparse_a); tensor_free(dense_a); return 0; }

    Tensor* result = sparse_matmul(sparse_a, dense_b);
    if (!result) { tensor_free(dense_b); sparse_free(sparse_a); tensor_free(dense_a); return 0; }

    tensor_free(result);
    tensor_free(dense_b);
    sparse_free(sparse_a);
    tensor_free(dense_a);
    return 1;
}

static int test_autocast(void) {
    if (autocast_is_enabled()) return 0;

    autocast_enter(DTYPE_FLOAT16);
    if (!autocast_is_enabled()) return 0;

    AutocastContext* ctx = autocast_get_context();
    if (!ctx || !ctx->enabled) return 0;
    if (ctx->target_dtype != DTYPE_FLOAT16) return 0;

    autocast_exit();
    if (autocast_is_enabled()) return 0;

    return 1;
}

static int test_grad_scaler(void) {
    GradScaler* scaler = grad_scaler_create(65536.0f, 2.0f, 0.5f, 2000);
    if (!scaler) return 0;

    if (scaler->scale_factor != 65536.0f) { grad_scaler_free(scaler); return 0; }
    if (scaler->growth_factor != 2.0f) { grad_scaler_free(scaler); return 0; }
    if (scaler->backoff_factor != 0.5f) { grad_scaler_free(scaler); return 0; }

    scaler->found_inf = false;
    grad_scaler_update(scaler);

    grad_scaler_free(scaler);
    return 1;
}

static int test_flash_attention(void) {
    MultiHeadAttention* mha = nn_multihead_attention(32, 4, 0.0f, DTYPE_FLOAT32, DEVICE_CPU);
    if (!mha) return 0;

    int shape[] = {1, 4, 32};
    TensorConfig config = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                           .has_dtype = true, .has_device = true};
    Tensor* query = tensor_ones(shape, 3, &config);
    Tensor* key   = tensor_ones(shape, 3, &config);
    Tensor* value = tensor_ones(shape, 3, &config);

    if (!query || !key || !value) {
        if (query) tensor_free(query);
        if (key) tensor_free(key);
        if (value) tensor_free(value);
        module_free((Module*)mha);
        return 0;
    }

    FlashAttentionConfig fa_config = {
        .enabled = true, .block_size_q = 2, .block_size_kv = 2, .causal = false
    };

    Tensor* output = flash_attention_forward(mha, query, key, value, NULL, &fa_config);
    if (output) tensor_free(output);
    tensor_free(query);
    tensor_free(key);
    tensor_free(value);
    module_free((Module*)mha);
    return 1;
}

static int test_instancenorm2d(void) {
    InstanceNorm2d* inn = nn_instancenorm2d(16, 1e-5f, true, DTYPE_FLOAT32, DEVICE_CPU);
    if (!inn) return 0;
    if (inn->num_features != 16) { module_free((Module*)inn); return 0; }
    module_free((Module*)inn);
    return 1;
}

static int test_rmsnorm(void) {
    RMSNorm* rn = nn_rmsnorm(64, 1e-6f, DTYPE_FLOAT32, DEVICE_CPU);
    if (!rn) return 0;
    if (rn->normalized_shape != 64) { module_free((Module*)rn); return 0; }
    module_free((Module*)rn);
    return 1;
}

static int test_flatten(void) {
    Flatten* fl = nn_flatten(1, -1);
    if (!fl) return 0;
    if (fl->start_dim != 1 || fl->end_dim != -1) { module_free((Module*)fl); return 0; }
    module_free((Module*)fl);
    return 1;
}

static int test_identity(void) {
    Identity* id = nn_identity();
    if (!id) return 0;

    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor* input = tensor_from_array_2d(data, 2, 2);
    if (!input) { module_free((Module*)id); return 0; }

    Tensor* output = module_forward((Module*)id, input);
    if (output != input) { tensor_free(input); module_free((Module*)id); return 0; }

    tensor_free(input);
    module_free((Module*)id);
    return 1;
}

static int test_batchnorm1d(void) {
    BatchNorm1d* bn = nn_batchnorm1d(32, 1e-5f, 0.1f, true, true, DTYPE_FLOAT32, DEVICE_CPU);
    if (!bn) return 0;
    if (bn->num_features != 32) { module_free((Module*)bn); return 0; }
    if (!bn->affine) { module_free((Module*)bn); return 0; }
    if (!bn->track_running_stats) { module_free((Module*)bn); return 0; }
    module_free((Module*)bn);
    return 1;
}

static int test_prelu(void) {
    PReLU* pr = nn_prelu(1, 0.25f, DTYPE_FLOAT32, DEVICE_CPU);
    if (!pr) return 0;
    if (pr->num_parameters_ != 1) { module_free((Module*)pr); return 0; }
    if (!pr->alpha) { module_free((Module*)pr); return 0; }
    module_free((Module*)pr);
    return 1;
}

static int test_triplet_loss(void) {
    float anchor_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float pos_data[]    = {1.1f, 2.1f, 3.1f, 4.1f};
    float neg_data[]    = {5.0f, 6.0f, 7.0f, 8.0f};

    Tensor* anchor   = tensor_from_array_2d(anchor_data, 1, 4);
    Tensor* positive = tensor_from_array_2d(pos_data, 1, 4);
    Tensor* negative = tensor_from_array_2d(neg_data, 1, 4);
    if (!anchor || !positive || !negative) {
        if (anchor) tensor_free(anchor);
        if (positive) tensor_free(positive);
        if (negative) tensor_free(negative);
        return 0;
    }

    Tensor* loss = tensor_triplet_margin_loss(anchor, positive, negative, 1.0f);
    if (!loss) { tensor_free(anchor); tensor_free(positive); tensor_free(negative); return 0; }

    tensor_ensure_executed(loss);
    tensor_free(loss);
    tensor_free(anchor);
    tensor_free(positive);
    tensor_free(negative);
    cml_reset_ir_context();
    return 1;
}

static int test_cosine_embedding_loss(void) {
    float x1_data[]  = {1.0f, 0.0f, 0.0f, 0.0f};
    float x2_data[]  = {0.0f, 1.0f, 0.0f, 0.0f};
    float tgt_data[] = {1.0f};

    Tensor* x1     = tensor_from_array_2d(x1_data, 1, 4);
    Tensor* x2     = tensor_from_array_2d(x2_data, 1, 4);
    Tensor* target = tensor_from_array_2d(tgt_data, 1, 1);
    if (!x1 || !x2 || !target) {
        if (x1) tensor_free(x1);
        if (x2) tensor_free(x2);
        if (target) tensor_free(target);
        return 0;
    }

    Tensor* loss = tensor_cosine_embedding_loss(x1, x2, target, 0.0f);
    if (!loss) { tensor_free(x1); tensor_free(x2); tensor_free(target); return 0; }

    tensor_ensure_executed(loss);
    tensor_free(loss);
    tensor_free(x1);
    tensor_free(x2);
    tensor_free(target);
    cml_reset_ir_context();
    return 1;
}

static int test_nll_loss(void) {
    float lp_data[] = {-0.5f, -1.2f, -2.0f,
                       -1.5f, -0.3f, -1.8f};
    Tensor* log_probs = tensor_from_array_2d(lp_data, 2, 3);
    if (!log_probs) return 0;

    float tgt_data[] = {0.0f, 1.0f};
    Tensor* targets = tensor_from_array_2d(tgt_data, 1, 2);
    if (!targets) { tensor_free(log_probs); return 0; }

    int tgt_shape[] = {2};
    ReshapeParams rp = {.new_shape = tgt_shape, .new_ndim = 1};
    Tensor* targets_1d = uop_reshape(targets, &rp);
    if (!targets_1d) { tensor_free(log_probs); tensor_free(targets); return 0; }

    Tensor* loss = tensor_nll_loss(log_probs, targets_1d);
    if (!loss) { tensor_free(targets_1d); tensor_free(log_probs); tensor_free(targets); return 0; }

    tensor_ensure_executed(loss);
    tensor_free(loss);
    tensor_free(targets_1d);
    tensor_free(log_probs);
    tensor_free(targets);
    cml_reset_ir_context();
    return 1;
}

int main(void) {
    printf("test_new_layers\n\n");

    TEST(rnn_multi_layer);
    TEST(rnn_bidirectional);
    TEST(lstm_multi_layer);
    TEST(gru_multi_layer);
    TEST(conv_transpose2d);
    TEST(conv_transpose3d);
    TEST(upsample_nearest);
    TEST(upsample_bilinear);
    TEST(pixel_shuffle);
    TEST(pixel_unshuffle);
    TEST(sparse_roundtrip);
    TEST(sparse_matmul);
    TEST(autocast);
    TEST(grad_scaler);
    TEST(flash_attention);
    TEST(instancenorm2d);
    TEST(rmsnorm);
    TEST(batchnorm1d);
    TEST(flatten);
    TEST(identity);
    TEST(prelu);
    TEST(triplet_loss);
    TEST(cosine_embedding_loss);
    TEST(nll_loss);

    cml_reset_ir_context();

    printf("\n%d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
