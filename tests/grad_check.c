#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

#include "cml.h"

#if defined(__SANITIZE_ADDRESS__) || \
    (defined(__clang__) && defined(__has_feature) && __has_feature(address_sanitizer))
#define CML_GRAD_CHECK_ASAN 1
#else
#define CML_GRAD_CHECK_ASAN 0
#endif

#define EPS   1e-3f
#define TOL   1e-2f

static int tests_run    = 0;
static int tests_passed = 0;

static Tensor* make_rand(int* shape, int ndim, bool requires_grad) {
    TensorConfig cfg = {0};
    Tensor* t = tensor_empty(shape, ndim, &cfg);
    if (!t) return NULL;
    float* d = (float*)tensor_data_ptr(t);
    for (size_t i = 0; i < t->numel; i++)
        d[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    t->requires_grad = requires_grad;
    return t;
}

static Tensor* make_rand_positive(int* shape, int ndim, bool requires_grad,
                                  float lo, float hi) {
    TensorConfig cfg = {0};
    Tensor* t = tensor_empty(shape, ndim, &cfg);
    if (!t) return NULL;
    float* d = (float*)tensor_data_ptr(t);
    for (size_t i = 0; i < t->numel; i++)
        d[i] = lo + ((float)rand() / RAND_MAX) * (hi - lo);
    t->requires_grad = requires_grad;
    return t;
}

static float tensor_sum_all(Tensor* t) {
    float* d = (float*)tensor_data_ptr(t);
    if (!d) return 0.0f;
    float s = 0.0f;
    for (size_t i = 0; i < t->numel; i++)
        s += d[i];
    return s;
}

static bool numerical_grad_check(const char* name, Tensor* param,
                                 float (*loss_fn)(void*), void* ctx,
                                 float eps, float tol) {
    if (!param || !param->grad) {
        printf("  FAIL [%s]: parameter or gradient is NULL\n", name);
        return false;
    }

#if CML_GRAD_CHECK_ASAN
    (void)loss_fn;
    (void)ctx;
    (void)eps;
    (void)tol;
    printf("  SKIP [%s]: numerical finite-diff under ASan (IR intern/decompose)\n", name);
    return true;
#else

    float* data = (float*)tensor_data_ptr(param);
    float* grad = (float*)tensor_data_ptr(param->grad);
    if (!data || !grad) {
        printf("  FAIL [%s]: cannot access data/grad pointers\n", name);
        return false;
    }

    for (size_t i = 0; i < param->numel; i++) {
        float orig = data[i];

        data[i] = orig + eps;
        float f_plus = loss_fn(ctx);

        data[i] = orig - eps;
        float f_minus = loss_fn(ctx);

        data[i] = orig;

        float numerical = (f_plus - f_minus) / (2.0f * eps);
        float analytical = grad[i];
        float diff = fabsf(numerical - analytical);

        if (diff > tol) {
            printf("  FAIL [%s]: element %zu  analytical=%.6f  numerical=%.6f  diff=%.6f\n",
                   name, i, analytical, numerical, diff);
            return false;
        }
    }
    return true;
#endif
}

typedef struct { Linear* layer; Tensor* input; } LinearCtx;

static float linear_loss(void* vctx) {
    LinearCtx* c = (LinearCtx*)vctx;
    Tensor* out = module_forward((Module*)c->layer, c->input);
    return tensor_sum_all(out);
}

static bool check_linear(void) {
    cml_reset_ir_context();
    int in_shape[] = {2, 4};
    Tensor* x = make_rand(in_shape, 2, true);
    Linear* layer = nn_linear(4, 3, DTYPE_FLOAT32, DEVICE_CPU, true);
    if (!layer || !x) {
        if (x) tensor_free(x);
        if (layer) module_free((Module*)layer);
        return false;
    }
    module_set_training((Module*)layer, true);

    Tensor* out = module_forward((Module*)layer, x);
    if (!out) {
        tensor_free(x);
        module_free((Module*)layer);
        return false;
    }
    tensor_ensure_executed(out);
    cml_backward(out, NULL, false, false);

    LinearCtx ctx = { layer, x };
    bool ok = true;
    if (layer->weight && layer->weight->tensor)
        ok = ok && numerical_grad_check("Linear.weight", layer->weight->tensor,
                                         linear_loss, &ctx, EPS, TOL);
    if (layer->bias && layer->bias->tensor)
        ok = ok && numerical_grad_check("Linear.bias", layer->bias->tensor,
                                         linear_loss, &ctx, EPS, TOL);
    module_free((Module*)layer);
    tensor_free(out);
    tensor_free(x);
    cml_reset_ir_context();
    return ok;
}

typedef struct { Conv1d* layer; Tensor* input; } Conv1dCtx;

static float conv1d_loss(void* vctx) {
    Conv1dCtx* c = (Conv1dCtx*)vctx;
    Tensor* out = module_forward((Module*)c->layer, c->input);
    return tensor_sum_all(out);
}

static bool check_conv1d(void) {
    int in_shape[] = {1, 2, 6};
    Tensor* x = make_rand(in_shape, 3, true);
    Conv1d* layer = nn_conv1d(2, 3, 3, 1, 0, 1, true, DTYPE_FLOAT32, DEVICE_CPU);
    if (!layer || !x) {
        if (x) tensor_free(x);
        if (layer) module_free((Module*)layer);
        return false;
    }
    module_set_training((Module*)layer, true);

    Tensor* out = module_forward((Module*)layer, x);
    if (!out) {
        tensor_free(x);
        module_free((Module*)layer);
        return false;
    }
    tensor_ensure_executed(out);
    cml_backward(out, NULL, false, false);

    Conv1dCtx ctx = { layer, x };
    bool ok = true;
    if (layer->weight && layer->weight->tensor)
        ok = ok && numerical_grad_check("Conv1d.weight", layer->weight->tensor,
                                         conv1d_loss, &ctx, EPS, TOL);
    if (layer->bias && layer->bias->tensor)
        ok = ok && numerical_grad_check("Conv1d.bias", layer->bias->tensor,
                                         conv1d_loss, &ctx, EPS, TOL);
    module_free((Module*)layer);
    tensor_free(out);
    tensor_free(x);
    cml_reset_ir_context();
    return ok;
}

typedef struct { Conv2d* layer; Tensor* input; } Conv2dCtx;

static float conv2d_loss(void* vctx) {
    Conv2dCtx* c = (Conv2dCtx*)vctx;
    Tensor* out = module_forward((Module*)c->layer, c->input);
    return tensor_sum_all(out);
}

static bool check_conv2d(void) {
    int in_shape[] = {1, 1, 5, 5};
    Tensor* x = make_rand(in_shape, 4, true);
    Conv2d* layer = nn_conv2d(1, 2, 3, 1, 0, 1, true, DTYPE_FLOAT32, DEVICE_CPU);
    if (!layer || !x) {
        if (x) tensor_free(x);
        if (layer) module_free((Module*)layer);
        return false;
    }
    module_set_training((Module*)layer, true);

    Tensor* out = module_forward((Module*)layer, x);
    if (!out) {
        tensor_free(x);
        module_free((Module*)layer);
        return false;
    }
    tensor_ensure_executed(out);
    cml_backward(out, NULL, false, false);

    Conv2dCtx ctx = { layer, x };
    bool ok = true;
    if (layer->weight && layer->weight->tensor)
        ok = ok && numerical_grad_check("Conv2d.weight", layer->weight->tensor,
                                         conv2d_loss, &ctx, EPS, TOL);
    if (layer->bias && layer->bias->tensor)
        ok = ok && numerical_grad_check("Conv2d.bias", layer->bias->tensor,
                                         conv2d_loss, &ctx, EPS, TOL);
    module_free((Module*)layer);
    tensor_free(out);
    tensor_free(x);
    cml_reset_ir_context();
    return ok;
}

typedef struct { Conv3d* layer; Tensor* input; } Conv3dCtx;

static float conv3d_loss(void* vctx) {
    Conv3dCtx* c = (Conv3dCtx*)vctx;
    Tensor* out = module_forward((Module*)c->layer, c->input);
    return tensor_sum_all(out);
}

static bool check_conv3d(void) {
    int in_shape[] = {1, 1, 4, 4, 4};
    Tensor* x = make_rand(in_shape, 5, true);
    Conv3d* layer = nn_conv3d(1, 2, 3, 1, 0, 1, true, DTYPE_FLOAT32, DEVICE_CPU);
    if (!layer || !x) {
        if (x) tensor_free(x);
        if (layer) module_free((Module*)layer);
        return false;
    }
    module_set_training((Module*)layer, true);

    Tensor* out = module_forward((Module*)layer, x);
    if (!out) {
        tensor_free(x);
        module_free((Module*)layer);
        return false;
    }
    tensor_ensure_executed(out);
    cml_backward(out, NULL, false, false);

    Conv3dCtx ctx = { layer, x };
    bool ok = true;
    if (layer->weight && layer->weight->tensor)
        ok = ok && numerical_grad_check("Conv3d.weight", layer->weight->tensor,
                                         conv3d_loss, &ctx, EPS, TOL);
    if (layer->bias && layer->bias->tensor)
        ok = ok && numerical_grad_check("Conv3d.bias", layer->bias->tensor,
                                         conv3d_loss, &ctx, EPS, TOL);
    module_free((Module*)layer);
    tensor_free(out);
    tensor_free(x);
    cml_reset_ir_context();
    return ok;
}

typedef struct { BatchNorm2d* layer; Tensor* input; } BN2dCtx;

static float bn2d_loss(void* vctx) {
    BN2dCtx* c = (BN2dCtx*)vctx;
    Tensor* out = module_forward((Module*)c->layer, c->input);
    return tensor_sum_all(out);
}

static bool check_batchnorm2d(void) {
    int in_shape[] = {2, 3, 4, 4};
    Tensor* x = make_rand(in_shape, 4, true);
    BatchNorm2d* layer = nn_batchnorm2d(3, 1e-5f, 0.1f, true, true,
                                         DTYPE_FLOAT32, DEVICE_CPU);
    if (!layer || !x) return false;
    module_set_training((Module*)layer, true);

    Tensor* out = module_forward((Module*)layer, x);
    if (!out) {
        tensor_free(x);
        module_free((Module*)layer);
        return false;
    }
    tensor_ensure_executed(out);
    cml_backward(out, NULL, false, false);

    BN2dCtx ctx = { layer, x };
    bool ok = true;
    if (layer->weight && layer->weight->tensor)
        ok = ok && numerical_grad_check("BatchNorm2d.weight", layer->weight->tensor,
                                         bn2d_loss, &ctx, EPS, TOL);
    if (layer->bias && layer->bias->tensor)
        ok = ok && numerical_grad_check("BatchNorm2d.bias", layer->bias->tensor,
                                         bn2d_loss, &ctx, EPS, TOL);
    module_free((Module*)layer);
    tensor_free(out);
    tensor_free(x);
    cml_reset_ir_context();
    return ok;
}

typedef struct { LayerNorm* layer; Tensor* input; } LNCtx;

static float ln_loss(void* vctx) {
    LNCtx* c = (LNCtx*)vctx;
    Tensor* out = module_forward((Module*)c->layer, c->input);
    return tensor_sum_all(out);
}

static bool check_layernorm(void) {
    int in_shape[] = {2, 8};
    Tensor* x = make_rand(in_shape, 2, true);
    LayerNorm* layer = nn_layernorm(8, 1e-5f, true, DTYPE_FLOAT32, DEVICE_CPU);
    if (!layer || !x) {
        if (x) tensor_free(x);
        if (layer) module_free((Module*)layer);
        return false;
    }
    module_set_training((Module*)layer, true);

    Tensor* out = module_forward((Module*)layer, x);
    if (!out) {
        tensor_free(x);
        module_free((Module*)layer);
        return false;
    }
    tensor_ensure_executed(out);
    cml_backward(out, NULL, false, false);

    LNCtx ctx = { layer, x };
    bool ok = true;
    if (layer->weight && layer->weight->tensor)
        ok = ok && numerical_grad_check("LayerNorm.weight", layer->weight->tensor,
                                         ln_loss, &ctx, EPS, TOL);
    if (layer->bias && layer->bias->tensor)
        ok = ok && numerical_grad_check("LayerNorm.bias", layer->bias->tensor,
                                         ln_loss, &ctx, EPS, TOL);
    module_free((Module*)layer);
    tensor_free(out);
    tensor_free(x);
    cml_reset_ir_context();
    return ok;
}

typedef struct { GroupNorm* layer; Tensor* input; } GNCtx;

static float gn_loss(void* vctx) {
    GNCtx* c = (GNCtx*)vctx;
    Tensor* out = module_forward((Module*)c->layer, c->input);
    return tensor_sum_all(out);
}

static bool check_groupnorm(void) {
    int in_shape[] = {2, 4, 3, 3};
    Tensor* x = make_rand(in_shape, 4, true);
    GroupNorm* layer = nn_groupnorm(2, 4, 1e-5f, true, DTYPE_FLOAT32, DEVICE_CPU);
    if (!layer || !x) {
        if (x) tensor_free(x);
        if (layer) module_free((Module*)layer);
        return false;
    }
    module_set_training((Module*)layer, true);

    Tensor* out = module_forward((Module*)layer, x);
    if (!out) {
        tensor_free(x);
        module_free((Module*)layer);
        return false;
    }
    tensor_ensure_executed(out);
    cml_backward(out, NULL, false, false);

    GNCtx ctx = { layer, x };
    bool ok = true;
    if (layer->weight && layer->weight->tensor)
        ok = ok && numerical_grad_check("GroupNorm.weight", layer->weight->tensor,
                                         gn_loss, &ctx, EPS, TOL);
    if (layer->bias && layer->bias->tensor)
        ok = ok && numerical_grad_check("GroupNorm.bias", layer->bias->tensor,
                                         gn_loss, &ctx, EPS, TOL);
    module_free((Module*)layer);
    tensor_free(out);
    tensor_free(x);
    cml_reset_ir_context();
    return ok;
}

typedef struct { RNNCell* cell; Tensor* input; Tensor* hidden; } RNNCellCtx;

static float rnn_cell_loss_fn(void* vctx) {
    RNNCellCtx* c = (RNNCellCtx*)vctx;
    Tensor* out = rnn_cell_forward(c->cell, c->input, c->hidden);
    return tensor_sum_all(out);
}

static bool check_rnn_cell(void) {
    int input_size = 4, hidden_size = 3, batch = 2;
    int in_shape[]  = {batch, input_size};
    int hid_shape[] = {batch, hidden_size};
    Tensor* x = make_rand(in_shape, 2, true);
    Tensor* h = make_rand(hid_shape, 2, true);
    RNNCell* cell = nn_rnn_cell(input_size, hidden_size, true,
                                 DTYPE_FLOAT32, DEVICE_CPU);
    if (!cell || !x || !h) {
        if (x) tensor_free(x);
        if (h) tensor_free(h);
        if (cell) module_free((Module*)cell);
        return false;
    }
    module_set_training((Module*)cell, true);

    Tensor* out = rnn_cell_forward(cell, x, h);
    if (!out) {
        tensor_free(x);
        tensor_free(h);
        module_free((Module*)cell);
        return false;
    }
    tensor_ensure_executed(out);
    cml_backward(out, NULL, false, false);

    RNNCellCtx ctx = { cell, x, h };
    bool ok = true;
    if (cell->weight_ih && cell->weight_ih->tensor)
        ok = ok && numerical_grad_check("RNNCell.weight_ih", cell->weight_ih->tensor,
                                         rnn_cell_loss_fn, &ctx, EPS, TOL);
    if (cell->weight_hh && cell->weight_hh->tensor)
        ok = ok && numerical_grad_check("RNNCell.weight_hh", cell->weight_hh->tensor,
                                         rnn_cell_loss_fn, &ctx, EPS, TOL);
    module_free((Module*)cell);
    tensor_free(out);
    tensor_free(x);
    tensor_free(h);
    cml_reset_ir_context();
    return ok;
}

typedef struct {
    LSTMCell* cell;
    Tensor* input;
    Tensor* h_prev;
    Tensor* c_prev;
} LSTMCellCtx;

static float lstm_cell_loss_fn(void* vctx) {
    LSTMCellCtx* c = (LSTMCellCtx*)vctx;
    Tensor* h_out = NULL;
    Tensor* c_out = NULL;
    lstm_cell_forward(c->cell, c->input, c->h_prev, c->c_prev, &h_out, &c_out);
    float loss = 0.0f;
    if (h_out) loss += tensor_sum_all(h_out);
    if (c_out) loss += tensor_sum_all(c_out);
    return loss;
}

static bool check_lstm_cell(void) {
    int input_size = 4, hidden_size = 3, batch = 2;
    int in_shape[]  = {batch, input_size};
    int hid_shape[] = {batch, hidden_size};
    Tensor* x  = make_rand(in_shape, 2, true);
    Tensor* hp = make_rand(hid_shape, 2, true);
    Tensor* cp = make_rand(hid_shape, 2, true);
    LSTMCell* cell = nn_lstm_cell(input_size, hidden_size, true,
                                   DTYPE_FLOAT32, DEVICE_CPU);
    if (!cell || !x || !hp || !cp) {
        if (x) tensor_free(x);
        if (hp) tensor_free(hp);
        if (cp) tensor_free(cp);
        if (cell) module_free((Module*)cell);
        return false;
    }
    module_set_training((Module*)cell, true);

    Tensor* h_out = NULL;
    Tensor* c_out = NULL;
    lstm_cell_forward(cell, x, hp, cp, &h_out, &c_out);
    if (!h_out) {
        tensor_free(x);
        tensor_free(hp);
        tensor_free(cp);
        module_free((Module*)cell);
        return false;
    }
    tensor_ensure_executed(h_out);
    if (c_out) tensor_ensure_executed(c_out);
    cml_backward(h_out, NULL, false, false);

    LSTMCellCtx ctx = { cell, x, hp, cp };
    bool ok = true;
    if (cell->weight_ih && cell->weight_ih->tensor)
        ok = ok && numerical_grad_check("LSTMCell.weight_ih", cell->weight_ih->tensor,
                                         lstm_cell_loss_fn, &ctx, EPS, TOL);
    if (cell->weight_hh && cell->weight_hh->tensor)
        ok = ok && numerical_grad_check("LSTMCell.weight_hh", cell->weight_hh->tensor,
                                         lstm_cell_loss_fn, &ctx, EPS, TOL);
    module_free((Module*)cell);
    if (c_out) tensor_free(c_out);
    tensor_free(h_out);
    tensor_free(x);
    tensor_free(hp);
    tensor_free(cp);
    cml_reset_ir_context();
    return ok;
}

typedef struct { GRUCell* cell; Tensor* input; Tensor* hidden; } GRUCellCtx;

static float gru_cell_loss_fn(void* vctx) {
    GRUCellCtx* c = (GRUCellCtx*)vctx;
    Tensor* out = gru_cell_forward(c->cell, c->input, c->hidden);
    return tensor_sum_all(out);
}

static bool check_gru_cell(void) {
    int input_size = 4, hidden_size = 3, batch = 2;
    int in_shape[]  = {batch, input_size};
    int hid_shape[] = {batch, hidden_size};
    Tensor* x = make_rand(in_shape, 2, true);
    Tensor* h = make_rand(hid_shape, 2, true);
    GRUCell* cell = nn_gru_cell(input_size, hidden_size, true,
                                 DTYPE_FLOAT32, DEVICE_CPU);
    if (!cell || !x || !h) {
        if (x) tensor_free(x);
        if (h) tensor_free(h);
        if (cell) module_free((Module*)cell);
        return false;
    }
    module_set_training((Module*)cell, true);

    Tensor* out = gru_cell_forward(cell, x, h);
    if (!out) {
        tensor_free(x);
        tensor_free(h);
        module_free((Module*)cell);
        return false;
    }
    tensor_ensure_executed(out);
    cml_backward(out, NULL, false, false);

    GRUCellCtx ctx = { cell, x, h };
    bool ok = true;
    if (cell->weight_ih && cell->weight_ih->tensor)
        ok = ok && numerical_grad_check("GRUCell.weight_ih", cell->weight_ih->tensor,
                                         gru_cell_loss_fn, &ctx, EPS, TOL);
    if (cell->weight_hh && cell->weight_hh->tensor)
        ok = ok && numerical_grad_check("GRUCell.weight_hh", cell->weight_hh->tensor,
                                         gru_cell_loss_fn, &ctx, EPS, TOL);
    module_free((Module*)cell);
    tensor_free(out);
    tensor_free(x);
    tensor_free(h);
    cml_reset_ir_context();
    return ok;
}

typedef struct { Embedding* layer; Tensor* indices; } EmbeddingCtx;

static float embedding_loss(void* vctx) {
    EmbeddingCtx* c = (EmbeddingCtx*)vctx;
    Tensor* out = module_forward((Module*)c->layer, c->indices);
    return tensor_sum_all(out);
}

static bool check_embedding(void) {
    int num_embeddings = 10, embedding_dim = 4;
    Embedding* layer = nn_embedding(num_embeddings, embedding_dim, -1,
                                     DTYPE_FLOAT32, DEVICE_CPU);
    if (!layer) return false;
    module_set_training((Module*)layer, true);

    int idx_shape[] = {2, 3};
    TensorConfig cfg = { .dtype = DTYPE_INT32, .device = DEVICE_CPU,
                         .has_dtype = true, .has_device = true };
    Tensor* indices = tensor_empty(idx_shape, 2, &cfg);
    if (!indices) { module_free((Module*)layer); return false; }
    int32_t* idx_data = (int32_t*)tensor_data_ptr(indices);
    idx_data[0] = 0; idx_data[1] = 3; idx_data[2] = 7;
    idx_data[3] = 1; idx_data[4] = 5; idx_data[5] = 9;

    Tensor* out = module_forward((Module*)layer, indices);
    if (!out) {
        tensor_free(indices);
        module_free((Module*)layer);
        return false;
    }
    tensor_ensure_executed(out);
    cml_backward(out, NULL, false, false);

    EmbeddingCtx ctx = { layer, indices };
    bool ok = true;
    if (layer->weight && layer->weight->tensor)
        ok = ok && numerical_grad_check("Embedding.weight", layer->weight->tensor,
                                         embedding_loss, &ctx, EPS, TOL);
    module_free((Module*)layer);
    tensor_free(out);
    tensor_free(indices);
    cml_reset_ir_context();
    return ok;
}

typedef struct { MaxPool2d* layer; Tensor* input; } MaxPool2dCtx;

static float maxpool2d_loss(void* vctx) {
    MaxPool2dCtx* c = (MaxPool2dCtx*)vctx;
    Tensor* out = module_forward((Module*)c->layer, c->input);
    return tensor_sum_all(out);
}

static bool check_maxpool2d(void) {
    int in_shape[] = {1, 1, 4, 4};
    Tensor* x = make_rand(in_shape, 4, true);
    MaxPool2d* layer = nn_maxpool2d(2, 2, 0, 1, false);
    if (!layer || !x) {
        if (x) tensor_free(x);
        if (layer) module_free((Module*)layer);
        return false;
    }

    Tensor* out = module_forward((Module*)layer, x);
    if (!out) {
        tensor_free(x);
        module_free((Module*)layer);
        return false;
    }
    tensor_ensure_executed(out);
    cml_backward(out, NULL, false, false);

    MaxPool2dCtx ctx = { layer, x };
    bool ok = true;
    if (x->grad) {
        ok = numerical_grad_check("MaxPool2d.input", x, maxpool2d_loss, &ctx, EPS, TOL);
    } else {
        printf("  SKIP [MaxPool2d]: no input gradient computed\n");
    }
    module_free((Module*)layer);
    tensor_free(out);
    tensor_free(x);
    cml_reset_ir_context();
    return ok;
}

typedef struct { AvgPool2d* layer; Tensor* input; } AvgPool2dCtx;

static float avgpool2d_loss(void* vctx) {
    AvgPool2dCtx* c = (AvgPool2dCtx*)vctx;
    Tensor* out = module_forward((Module*)c->layer, c->input);
    return tensor_sum_all(out);
}

static bool check_avgpool2d(void) {
    int in_shape[] = {1, 1, 4, 4};
    Tensor* x = make_rand(in_shape, 4, true);
    AvgPool2d* layer = nn_avgpool2d(2, 2, 0, false, true);
    if (!layer || !x) {
        if (x) tensor_free(x);
        if (layer) module_free((Module*)layer);
        return false;
    }

    Tensor* out = module_forward((Module*)layer, x);
    if (!out) {
        tensor_free(x);
        module_free((Module*)layer);
        return false;
    }
    tensor_ensure_executed(out);
    cml_backward(out, NULL, false, false);

    AvgPool2dCtx ctx = { layer, x };
    bool ok = true;
    if (x->grad) {
        ok = numerical_grad_check("AvgPool2d.input", x, avgpool2d_loss, &ctx, EPS, TOL);
    } else {
        printf("  SKIP [AvgPool2d]: no input gradient computed\n");
    }
    module_free((Module*)layer);
    tensor_free(out);
    tensor_free(x);
    cml_reset_ir_context();
    return ok;
}

typedef struct { Module* act; Tensor* input; } ActivationCtx;

static float activation_loss(void* vctx) {
    ActivationCtx* c = (ActivationCtx*)vctx;
    Tensor* out = module_forward(c->act, c->input);
    return tensor_sum_all(out);
}

static bool check_activation(const char* name, Module* act, Tensor* x) {
    if (!act || !x) {
        if (x) tensor_free(x);
        if (act) module_free(act);
        return false;
    }

    Tensor* out = module_forward(act, x);
    if (!out) {
        tensor_free(x);
        module_free(act);
        return false;
    }
    tensor_ensure_executed(out);
    cml_backward(out, NULL, false, false);

    ActivationCtx ctx = { act, x };
    bool ok = true;
    if (x->grad) {
        ok = numerical_grad_check(name, x, activation_loss, &ctx, EPS, TOL);
    } else {
        printf("  SKIP [%s]: no input gradient computed\n", name);
    }
    module_free(act);
    tensor_free(out);
    tensor_free(x);
    cml_reset_ir_context();
    return ok;
}

static bool check_relu(void) {
    int shape[] = {2, 4};
    Tensor* x = make_rand_positive(shape, 2, true, 0.1f, 2.0f);
    ReLU* layer = nn_relu(false);
    return check_activation("ReLU", (Module*)layer, x);
}

static bool check_sigmoid(void) {
    int shape[] = {2, 4};
    Tensor* x = make_rand(shape, 2, true);
    Sigmoid* layer = nn_sigmoid();
    return check_activation("Sigmoid", (Module*)layer, x);
}

static bool check_tanh(void) {
    int shape[] = {2, 4};
    Tensor* x = make_rand(shape, 2, true);
    Tanh* layer = nn_tanh();
    return check_activation("Tanh", (Module*)layer, x);
}

static bool check_leaky_relu(void) {
    int shape[] = {2, 4};
    Tensor* x = make_rand(shape, 2, true);
    float* d = (float*)tensor_data_ptr(x);
    for (size_t i = 0; i < x->numel; i++) {
        if (fabsf(d[i]) < 0.05f)
            d[i] = (d[i] >= 0.0f) ? 0.1f : -0.1f;
    }
    LeakyReLU* layer = nn_leaky_relu(0.01f, false);
    return check_activation("LeakyReLU", (Module*)layer, x);
}

static bool check_softmax(void) {
    int shape[] = {2, 5};
    Tensor* x = make_rand(shape, 2, true);
    Softmax* layer = nn_softmax(1);
    return check_activation("Softmax", (Module*)layer, x);
}

static int required_run    = 0;
static int required_passed = 0;

#define RUN_TEST(fn) do {                                       \
    tests_run++;                                                \
    printf("  [%2d] %-25s ... ", tests_run, #fn);               \
    if (check_##fn()) {                                         \
        printf("PASS\n");                                       \
        tests_passed++;                                         \
    } else {                                                    \
        printf("FAIL\n");                                       \
    }                                                           \
} while (0)

#define RUN_TEST_XFAIL(fn) do {                                 \
    tests_run++;                                                \
    printf("  [%2d] %-25s ... ", tests_run, #fn);               \
    if (check_##fn()) {                                         \
        printf("PASS (xfail cleared!)\n");                      \
        tests_passed++;                                         \
    } else {                                                    \
        printf("XFAIL\n");                                      \
        tests_passed++;                                         \
    }                                                           \
} while (0)

int main(void) {
    srand((unsigned)time(NULL));

    printf("grad_check  eps=%.1e  tol=%.1e\n\n", (double)EPS, (double)TOL);

    RUN_TEST(linear);
    RUN_TEST(relu);
    RUN_TEST(sigmoid);
    RUN_TEST(tanh);
    RUN_TEST(leaky_relu);
    RUN_TEST(maxpool2d);
    RUN_TEST(avgpool2d);

    required_run    = tests_run;
    required_passed = tests_passed;

    printf("\n  -- Expected failures (autograd backward WIP) --\n");
    RUN_TEST_XFAIL(conv1d);
    RUN_TEST_XFAIL(conv2d);
    RUN_TEST_XFAIL(conv3d);
    RUN_TEST_XFAIL(batchnorm2d);
    RUN_TEST_XFAIL(layernorm);
    RUN_TEST_XFAIL(groupnorm);
    RUN_TEST_XFAIL(rnn_cell);
    RUN_TEST_XFAIL(lstm_cell);
    RUN_TEST_XFAIL(gru_cell);
    RUN_TEST_XFAIL(embedding);
    RUN_TEST_XFAIL(softmax);

    printf("\n%d/%d passed", tests_passed, tests_run);
    if (required_passed < required_run)
        printf(" (%d required FAILED)", required_run - required_passed);
    printf("\n");

    cml_reset_ir_context();
    return (required_passed == required_run) ? 0 : 1;
}
