/**
 * Forward-pass timings for the model families the conv/MLP harnesses miss:
 * recurrent cells (RNN/LSTM/GRU), attention, and LayerNorm. These paths are
 * dominated by per-op dispatch rather than BLAS, so the distribution (min /
 * median / p95 / stddev) matters more than a single mean -- a tail trial should
 * not be mistaken for the cost. Each case realizes its output and resets the IR
 * context per rep, the same lifecycle the training loops use.
 */
#include "profile_common.h"

#include "nn/layers/rnn.h"

enum { TRIALS = 30, REPS = 20 };

/* A fixed [rows, cols] float32 tensor with deterministic data. */
static Tensor* fixed_tensor(float* backing, int rows, int cols, float base) {
    for (int i = 0; i < rows * cols; i++)
        backing[i] = base + 0.01f * (float)((i % 17) - 8);
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    int shape[2] = {rows, cols};
    return cml_tensor(backing, shape, 2, &cfg);
}

int main(void) {
    cml_init();
    srand(42);

    const int batch = 32, in_sz = 64, hid = 128;
    printf("=== Recurrent cells (batch=%d, in=%d, hidden=%d) ===\n", batch, in_sz, hid);

    float xback[32 * 64], hback[32 * 128], cback[32 * 128];
    Tensor* x = fixed_tensor(xback, batch, in_sz, 0.3f);
    Tensor* h = fixed_tensor(hback, batch, hid, 0.1f);
    Tensor* c = fixed_tensor(cback, batch, hid, 0.2f);
    double samples[TRIALS];

    RNNCell* rnn = cml_nn_rnn_cell(in_sz, hid, true, DTYPE_FLOAT32, DEVICE_CPU);
    BENCH_COLLECT(samples, TRIALS, REPS, {
        Tensor* o = rnn_cell_forward(rnn, x, h);
        (void)tensor_data_ptr(o);
        cml_reset_ir_context();
    });
    bench_stats_print("RNN cell forward", bench_stats(samples, TRIALS));

    LSTMCell* lstm = cml_nn_lstm_cell(in_sz, hid, true, DTYPE_FLOAT32, DEVICE_CPU);
    BENCH_COLLECT(samples, TRIALS, REPS, {
        Tensor* ho = NULL;
        Tensor* co = NULL;
        lstm_cell_forward(lstm, x, h, c, &ho, &co);
        if (ho)
            (void)tensor_data_ptr(ho);
        if (co)
            (void)tensor_data_ptr(co);
        cml_reset_ir_context();
    });
    bench_stats_print("LSTM cell forward", bench_stats(samples, TRIALS));

    GRUCell* gru = cml_nn_gru_cell(in_sz, hid, true, DTYPE_FLOAT32, DEVICE_CPU);
    BENCH_COLLECT(samples, TRIALS, REPS, {
        Tensor* o = gru_cell_forward(gru, x, h);
        (void)tensor_data_ptr(o);
        cml_reset_ir_context();
    });
    bench_stats_print("GRU cell forward", bench_stats(samples, TRIALS));

    printf("\n=== Attention / LayerNorm ===\n");

    /* Scaled dot-product attention over [batch, seq, dim] folded as 2D. */
    const int seq = 64, dim = 64;
    float qb[64 * 64], kb[64 * 64], vb[64 * 64];
    Tensor* q = fixed_tensor(qb, seq, dim, 0.1f);
    Tensor* k = fixed_tensor(kb, seq, dim, 0.2f);
    Tensor* v = fixed_tensor(vb, seq, dim, 0.3f);
    BENCH_COLLECT(samples, TRIALS, REPS, {
        Tensor* o = uop_scaled_dot_product_attention(q, k, v, NULL);
        (void)tensor_data_ptr(o);
        cml_reset_ir_context();
    });
    bench_stats_print("SDPA forward", bench_stats(samples, TRIALS));

    const int ln_rows = 32, ln_dim = 256;
    float lnback[32 * 256];
    Tensor* ln_in = fixed_tensor(lnback, ln_rows, ln_dim, 0.5f);
    LayerNorm* ln = cml_nn_layernorm(ln_dim, 1e-5f, true, DTYPE_FLOAT32, DEVICE_CPU);
    BENCH_COLLECT(samples, TRIALS, REPS, {
        Tensor* o = module_forward((Module*)ln, ln_in);
        (void)tensor_data_ptr(o);
        cml_reset_ir_context();
    });
    bench_stats_print("LayerNorm forward", bench_stats(samples, TRIALS));

    /* --- Backward pass: isolate forward vs forward+backward on an MLP, since
     * backward dominates the training step but no harness measured it. --- */
    printf("\n=== Backward pass (MLP 128->256->10, batch=32) ===\n");
    {
        Sequential* net = cml_nn_sequential();
        sequential_add(net, (Module*)cml_nn_linear(128, 256, DTYPE_FLOAT32, DEVICE_CPU, true));
        sequential_add(net, (Module*)cml_nn_relu(false));
        sequential_add(net, (Module*)cml_nn_linear(256, 10, DTYPE_FLOAT32, DEVICE_CPU, true));
        float xb[32 * 128], yb[32 * 10];
        Tensor* X = fixed_tensor(xb, 32, 128, 0.2f);
        Tensor* Y = fixed_tensor(yb, 32, 10, 0.1f);

        BENCH_COLLECT(samples, TRIALS, REPS, {
            Tensor* o = cml_nn_sequential_forward(net, X);
            (void)tensor_data_ptr(o);
            cml_reset_ir_context();
        });
        bench_stats_print("forward only", bench_stats(samples, TRIALS));

        BENCH_COLLECT(samples, TRIALS, REPS, {
            Tensor* o = cml_nn_sequential_forward(net, X);
            Tensor* l = cml_nn_mse_loss(o, Y);
            cml_backward(l, NULL, false, false);
            cml_reset_ir_context();
        });
        bench_stats_print("forward + backward", bench_stats(samples, TRIALS));

        module_free((Module*)net);
    }

    /* --- Optimizer step cost across every optimizer, to confirm by measurement
     * (not just inspection) that each applies its update cheaply in place. --- */
    printf("\n=== Optimizer step (MLP params, grads materialized) ===\n");
    {
        Sequential* net = cml_nn_sequential();
        sequential_add(net, (Module*)cml_nn_linear(128, 256, DTYPE_FLOAT32, DEVICE_CPU, true));
        sequential_add(net, (Module*)cml_nn_relu(false));
        sequential_add(net, (Module*)cml_nn_linear(256, 10, DTYPE_FLOAT32, DEVICE_CPU, true));
        Parameter** params = NULL;
        int np             = 0;
        module_collect_parameters((Module*)net, &params, &np, true);

        float xb[32 * 128], yb[32 * 10];
        Tensor* X = fixed_tensor(xb, 32, 128, 0.2f);
        Tensor* Y = fixed_tensor(yb, 32, 10, 0.1f);
        Tensor* o = cml_nn_sequential_forward(net, X);
        Tensor* l = cml_nn_mse_loss(o, Y);
        cml_backward(l, NULL, false, false); /* materialise grads once */

        struct {
            const char* name;
            Optimizer* opt;
        } opts[] = {
            {"SGD", cml_optim_sgd(params, np, 0.01f, 0.9f, 0.0f)},
            {"Adam", cml_optim_adam(params, np, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f)},
            {"RMSprop", cml_optim_rmsprop(params, np, 0.01f, 0.0f, 0.99f, 1e-8f)},
            {"AdamW", cml_optim_adamw(params, np, 0.01f, 0.01f, 0.9f, 0.999f, 1e-8f)},
            {"Adagrad", cml_optim_adagrad(params, np, 0.01f, 0.0f, 1e-10f)},
            {"Adamax", cml_optim_adamax(params, np, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f)},
            {"NAdam", cml_optim_nadam(params, np, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f)},
            {"Adadelta", cml_optim_adadelta(params, np, 0.9f, 0.0f, 1e-6f)},
        };
        for (size_t oi = 0; oi < sizeof(opts) / sizeof(opts[0]); oi++) {
            if (!opts[oi].opt)
                continue;
            BENCH_COLLECT(samples, TRIALS, REPS, { optimizer_step(opts[oi].opt); });
            bench_stats_print(opts[oi].name, bench_stats(samples, TRIALS));
            optimizer_free(opts[oi].opt);
        }
        cml_free(params);
        module_free((Module*)net);
    }

    printf("\nExecution completed successfully\n");
    return 0;
}
