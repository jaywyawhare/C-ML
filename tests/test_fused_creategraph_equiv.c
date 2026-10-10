/**
 * The fused softmax/log-softmax/RNN-cell/LayerNorm VJPs use a fast fused
 * backward kernel for ordinary backward, but switch to a differentiable
 * primitive form under create_graph (so double-backward is correct). Both forms
 * must compute the same first-order gradient. This checks that the create_graph
 * (primitive) input gradient matches the non-create_graph (fused) one.
 */
#include <math.h>
#include <stdio.h>

#include "cml.h"
#include "autograd/autograd.h"
#include "nn/layers/rnn.h"
#include "ops/uops.h"

#define D 5

static TensorConfig CFG = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

/* Build the op's output from a fresh leaf x (filled from xd); returns x via *xo. */
typedef Tensor* (*Builder)(Tensor* x);

static Tensor* b_softmax(Tensor* x) { return uop_softmax(x, -1); }
static Tensor* b_logsm(Tensor* x) { return uop_log_softmax_lastdim(x); }

/* input grad of sum(w .* op(x)) for a fixed readout w, with the given create_graph flag. */
static void grad_of(Builder op, const float* xd, const float* w, int n, bool cg, float* out) {
    int s[]          = {1, n};
    Tensor* x        = cml_tensor((float*)xd, s, 2, &CFG);
    x->requires_grad = true;
    Tensor* y        = op(x);
    Tensor* wt       = cml_tensor((float*)w, s, 2, &CFG);
    Tensor* loss     = uop_sum(uop_mul(y, wt), &(ReduceParams){0});
    tensor_backward(loss, NULL, false, cg);
    const float* g = x->grad ? (const float*)tensor_data_ptr(x->grad) : NULL;
    for (int i = 0; i < n; i++)
        out[i] = g ? g[i] : 0.0f / 0.0f;
    cml_reset_ir_context();
}

static int check(const char* name, Builder op) {
    float xd[D] = {0.3f, -0.2f, 0.5f, 0.1f, -0.4f};
    float w[D]  = {1.0f, -0.5f, 0.25f, 0.75f, -1.0f};
    float gf[D], gt[D];
    grad_of(op, xd, w, D, false, gf); /* fused */
    grad_of(op, xd, w, D, true, gt);  /* primitive (create_graph) */
    float maxdiff = 0.0f;
    for (int i = 0; i < D; i++) {
        float d = fabsf(gf[i] - gt[i]);
        if (d > maxdiff)
            maxdiff = d;
    }
    int pass = maxdiff < 1e-5f;
    printf("  %-12s fused-vs-creategraph grad maxdiff=%.3e %s\n", name, (double)maxdiff,
           pass ? "PASS" : "FAIL");
    return pass;
}

/* RNN cell: grad of sum(w .* rnn_cell(ih,hh)) w.r.t ih, fused vs create_graph. */
static int check_rnn(void) {
    float ihd[D] = {0.3f, -0.2f, 0.5f, 0.1f, -0.4f};
    float hhd[D] = {0.1f, 0.2f, -0.1f, 0.0f, 0.3f};
    float w[D]   = {1.0f, -0.5f, 0.25f, 0.75f, -1.0f};
    int s[]      = {1, D};
    float gf[D], gt[D];
    for (int pass = 0; pass < 2; pass++) {
        Tensor* ih        = cml_tensor(ihd, s, 2, &CFG);
        ih->requires_grad = true;
        Tensor* hh        = cml_tensor(hhd, s, 2, &CFG);
        Tensor* y         = uop_rnn_cell(ih, hh);
        Tensor* wt        = cml_tensor(w, s, 2, &CFG);
        Tensor* loss      = uop_sum(uop_mul(y, wt), &(ReduceParams){0});
        tensor_backward(loss, NULL, false, pass == 1);
        const float* g = ih->grad ? (const float*)tensor_data_ptr(ih->grad) : NULL;
        for (int i = 0; i < D; i++)
            (pass ? gt : gf)[i] = g ? g[i] : 0.0f;
        cml_reset_ir_context();
    }
    float md = 0.0f;
    for (int i = 0; i < D; i++)
        md = fmaxf(md, fabsf(gf[i] - gt[i]));
    int ok = md < 1e-5f;
    printf("  %-12s fused-vs-creategraph grad maxdiff=%.3e %s\n", "rnn_cell", (double)md,
           ok ? "PASS" : "FAIL");
    return ok;
}

/* LayerNorm: grad of sum(w .* layernorm(x,gamma,beta)) w.r.t x, fused vs create_graph. */
static int check_layernorm(void) {
    float xd[D] = {0.3f, -0.2f, 0.5f, 0.1f, -0.4f};
    float gd[D] = {1.1f, 0.9f, 1.0f, 1.2f, 0.8f};
    float bd[D] = {0.0f, 0.1f, -0.1f, 0.0f, 0.05f};
    float w[D]  = {1.0f, -0.5f, 0.25f, 0.75f, -1.0f};
    int s[] = {1, D}, ds[] = {D};
    float gf[D], gt[D];
    for (int pass = 0; pass < 2; pass++) {
        Tensor* x        = cml_tensor(xd, s, 2, &CFG);
        x->requires_grad = true;
        Tensor* gamma    = cml_tensor(gd, ds, 1, &CFG);
        Tensor* beta     = cml_tensor(bd, ds, 1, &CFG);
        Tensor* y        = uop_layernorm(x, gamma, beta, 1e-5f);
        Tensor* wt       = cml_tensor(w, s, 2, &CFG);
        Tensor* loss     = uop_sum(uop_mul(y, wt), &(ReduceParams){0});
        tensor_backward(loss, NULL, false, pass == 1);
        const float* g = x->grad ? (const float*)tensor_data_ptr(x->grad) : NULL;
        for (int i = 0; i < D; i++)
            (pass ? gt : gf)[i] = g ? g[i] : 0.0f;
        cml_reset_ir_context();
    }
    float md = 0.0f;
    for (int i = 0; i < D; i++)
        md = fmaxf(md, fabsf(gf[i] - gt[i]));
    int ok = md < 1e-4f;
    printf("  %-12s fused-vs-creategraph grad maxdiff=%.3e %s\n", "layernorm", (double)md,
           ok ? "PASS" : "FAIL");
    return ok;
}

/* RMSNorm: grad of sum(w .* rmsnorm(x,weight)) w.r.t x, fused vs create_graph. */
static int check_rmsnorm(void) {
    float xd[D] = {0.3f, -0.2f, 0.5f, 0.1f, -0.4f};
    float wd[D] = {1.1f, 0.9f, 1.0f, 1.2f, 0.8f};
    float rw[D] = {1.0f, -0.5f, 0.25f, 0.75f, -1.0f};
    int s[] = {1, D}, ds[] = {D};
    float gf[D], gt[D];
    for (int pass = 0; pass < 2; pass++) {
        Tensor* x        = cml_tensor(xd, s, 2, &CFG);
        x->requires_grad = true;
        Tensor* weight   = cml_tensor(wd, ds, 1, &CFG);
        Tensor* y        = uop_rmsnorm(x, weight, 1e-5f);
        Tensor* wt       = cml_tensor(rw, s, 2, &CFG);
        Tensor* loss     = uop_sum(uop_mul(y, wt), &(ReduceParams){0});
        tensor_backward(loss, NULL, false, pass == 1);
        const float* g = x->grad ? (const float*)tensor_data_ptr(x->grad) : NULL;
        for (int i = 0; i < D; i++)
            (pass ? gt : gf)[i] = g ? g[i] : 0.0f;
        cml_reset_ir_context();
    }
    float md = 0.0f;
    for (int i = 0; i < D; i++)
        md = fmaxf(md, fabsf(gf[i] - gt[i]));
    int ok = md < 1e-4f;
    printf("  %-12s fused-vs-creategraph grad maxdiff=%.3e %s\n", "rmsnorm", (double)md,
           ok ? "PASS" : "FAIL");
    return ok;
}

/* GRU cell: grad of sum(w .* gru_cell(ih,hh,hidden)) w.r.t ih, fused vs cg. */
static int check_gru(void) {
    const int H  = 3;
    float ihd[9] = {0.3f, -0.2f, 0.5f, 0.1f, -0.4f, 0.2f, 0.0f, 0.3f, -0.1f};
    float hhd[9] = {0.1f, 0.2f, -0.1f, 0.0f, 0.3f, -0.2f, 0.4f, -0.3f, 0.2f};
    float hd[3]  = {0.2f, -0.1f, 0.3f};
    float w[3]   = {1.0f, -0.5f, 0.75f};
    int s3[] = {1, 3 * H}, s1[] = {1, H};
    float gf[9], gt[9];
    for (int pass = 0; pass < 2; pass++) {
        Tensor* ih        = cml_tensor(ihd, s3, 2, &CFG);
        ih->requires_grad = true;
        Tensor* hh        = cml_tensor(hhd, s3, 2, &CFG);
        Tensor* hid       = cml_tensor(hd, s1, 2, &CFG);
        Tensor* y         = uop_gru_cell(ih, hh, hid);
        Tensor* wt        = cml_tensor(w, s1, 2, &CFG);
        Tensor* loss      = uop_sum(uop_mul(y, wt), &(ReduceParams){0});
        tensor_backward(loss, NULL, false, pass == 1);
        const float* g = ih->grad ? (const float*)tensor_data_ptr(ih->grad) : NULL;
        for (int i = 0; i < 3 * H; i++)
            (pass ? gt : gf)[i] = g ? g[i] : 0.0f;
        cml_reset_ir_context();
    }
    float md = 0.0f;
    for (int i = 0; i < 9; i++)
        md = fmaxf(md, fabsf(gf[i] - gt[i]));
    int ok = md < 1e-5f;
    printf("  %-12s fused-vs-creategraph grad maxdiff=%.3e %s\n", "gru_cell", (double)md,
           ok ? "PASS" : "FAIL");
    return ok;
}

/* LSTM cell: grad of sum(w .* lstm_cell(gates,c_prev)) w.r.t gates, fused vs cg. */
static int check_lstm(void) {
    const int H  = 3;
    float gd[12] = {0.3f, -0.2f, 0.5f, 0.1f, -0.4f, 0.2f, 0.0f, 0.3f, -0.1f, 0.2f, 0.4f, -0.3f};
    float cd[3]  = {0.2f, -0.1f, 0.3f};
    float w[6]   = {1.0f, -0.5f, 0.75f, 0.25f, -1.0f, 0.5f}; /* over packed [h|c], 2H=6 */
    int s4[] = {1, 4 * H}, s1[] = {1, H}, s2[] = {1, 2 * H};
    float gf[12], gt[12];
    for (int pass = 0; pass < 2; pass++) {
        Tensor* gates        = cml_tensor(gd, s4, 2, &CFG);
        gates->requires_grad = true;
        Tensor* c_prev       = cml_tensor(cd, s1, 2, &CFG);
        Tensor* y            = uop_lstm_cell(gates, c_prev); /* packed [1,2H] */
        Tensor* wt           = cml_tensor(w, s2, 2, &CFG);
        Tensor* loss         = uop_sum(uop_mul(y, wt), &(ReduceParams){0});
        tensor_backward(loss, NULL, false, pass == 1);
        const float* g = gates->grad ? (const float*)tensor_data_ptr(gates->grad) : NULL;
        for (int i = 0; i < 4 * H; i++)
            (pass ? gt : gf)[i] = g ? g[i] : 0.0f;
        cml_reset_ir_context();
    }
    float md = 0.0f;
    for (int i = 0; i < 12; i++)
        md = fmaxf(md, fabsf(gf[i] - gt[i]));
    int ok = md < 1e-5f;
    printf("  %-12s fused-vs-creategraph grad maxdiff=%.3e %s\n", "lstm_cell", (double)md,
           ok ? "PASS" : "FAIL");
    return ok;
}

int main(void) {
    cml_init();
    printf("Fused vs create_graph first-order grad equivalence\n");
    int ok = 1;
    ok &= check("softmax", b_softmax);
    ok &= check("log_softmax", b_logsm);
    ok &= check_rnn();
    ok &= check_layernorm();
    ok &= check_rmsnorm();
    ok &= check_gru();
    ok &= check_lstm();
    printf(ok ? "Equivalence holds.\n" : "MISMATCH.\n");
    return ok ? 0 : 1;
}
