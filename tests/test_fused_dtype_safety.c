/**
 * The fused softmax/log-softmax/recurrent/layernorm kernels compute in f32 and
 * cast half inputs/outputs at the boundary, so they handle non-f32 dtypes
 * natively (not garbage from misreading half data). This checks that each fused
 * op's bf16 forward matches its f32 reference within bf16 precision.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cml.h"
#include "ops/uops.h"

static TensorConfig F32 = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

/* bf16 has ~3 significant bits of mantissa; tolerate its quantization. */
#define TOL 3e-2f

/* Build op(inputs...) for a given dtype; the builder reads leaves from `mk`. */
typedef Tensor* (*OpBuilder)(Tensor** leaves, int n);

/* Compare op's output in bf16 against its f32 reference. `vals`/`shapes`/`ndims`
 * describe n leaf tensors; each is materialized in f32 (reference) then bf16. */
static int check(const char* name, OpBuilder op, float** vals, int** shapes, int* ndims, int n) {
    Tensor* lf[8];
    for (int i = 0; i < n; i++)
        lf[i] = cml_tensor(vals[i], shapes[i], ndims[i], &F32);
    Tensor* yf      = op(lf, n);
    const float* rf = (const float*)tensor_data_ptr(yf);
    size_t m        = yf->numel;
    float* ref      = (float*)malloc(m * sizeof(float));
    for (size_t j = 0; j < m; j++)
        ref[j] = rf[j];
    cml_reset_ir_context();

    Tensor* lb[8];
    for (int i = 0; i < n; i++) {
        Tensor* f = cml_tensor(vals[i], shapes[i], ndims[i], &F32);
        lb[i]     = cml_cast(f, DTYPE_BFLOAT16);
    }
    Tensor* yb = op(lb, n);
    if (!yb || !tensor_data_ptr(yb)) {
        printf("  %-12s bf16 output missing FAIL\n", name);
        free(ref);
        return 0;
    }
    float maxdiff = 0.0f;
    for (size_t j = 0; j < m; j++) {
        float d = fabsf(tensor_get_float(yb, j) - ref[j]);
        if (d > maxdiff)
            maxdiff = d;
    }
    cml_reset_ir_context();
    free(ref);
    int pass = maxdiff < TOL;
    printf("  %-12s bf16 vs f32 maxdiff=%.3e %s\n", name, (double)maxdiff, pass ? "PASS" : "FAIL");
    return pass;
}

static Tensor* b_softmax(Tensor** l, int n) {
    (void)n;
    return uop_softmax(l[0], -1);
}
static Tensor* b_logsm(Tensor** l, int n) {
    (void)n;
    return uop_log_softmax_lastdim(l[0]);
}
static Tensor* b_rnn(Tensor** l, int n) {
    (void)n;
    return uop_rnn_cell(l[0], l[1]);
}
static Tensor* b_gru(Tensor** l, int n) {
    (void)n;
    return uop_gru_cell(l[0], l[1], l[2]);
}
static Tensor* b_lstm(Tensor** l, int n) {
    (void)n;
    return uop_lstm_cell(l[0], l[1]);
}
static Tensor* b_layernorm(Tensor** l, int n) {
    (void)n;
    return uop_layernorm(l[0], l[1], l[2], 1e-5f);
}
static Tensor* b_rmsnorm(Tensor** l, int n) {
    (void)n;
    return uop_rmsnorm(l[0], l[1], 1e-5f);
}

int main(void) {
    cml_init();
    printf("Fused-op dtype safety (native bf16)\n");
    int ok = 1;

    float x4[4] = {1.0f, 2.0f, 3.0f, 0.5f};
    int s14[]   = {1, 4};
    float* sm[] = {x4};
    int* sms[]  = {s14};
    int smn[]   = {2};
    ok &= check("softmax", b_softmax, sm, sms, smn, 1);
    ok &= check("log_softmax", b_logsm, sm, sms, smn, 1);

    /* RNN cell: ih, hh [1,H]. */
    float ih[3] = {0.3f, -0.2f, 0.5f}, hh[3] = {0.1f, 0.2f, -0.1f};
    int s13[]    = {1, 3};
    float* rnn[] = {ih, hh};
    int* rnns[]  = {s13, s13};
    int rnnn[]   = {2, 2};
    ok &= check("rnn_cell", b_rnn, rnn, rnns, rnnn, 2);

    /* GRU cell: ih, hh [1,3H], hidden [1,H], H=3. */
    float gih[9] = {0.3f, -0.2f, 0.5f, 0.1f, -0.4f, 0.2f, 0.0f, 0.3f, -0.1f};
    float ghh[9] = {0.1f, 0.2f, -0.1f, 0.0f, 0.3f, -0.2f, 0.4f, -0.3f, 0.2f};
    float ghd[3] = {0.2f, -0.1f, 0.3f};
    int s19[] = {1, 9}, s13b[] = {1, 3};
    float* gru[] = {gih, ghh, ghd};
    int* grus[]  = {s19, s19, s13b};
    int grun[]   = {2, 2, 2};
    ok &= check("gru_cell", b_gru, gru, grus, grun, 3);

    /* LSTM cell: gates [1,4H], c_prev [1,H], H=3. */
    float lg[12] = {0.3f, -0.2f, 0.5f, 0.1f, -0.4f, 0.2f, 0.0f, 0.3f, -0.1f, 0.2f, 0.4f, -0.3f};
    float lc[3]  = {0.2f, -0.1f, 0.3f};
    int s112[] = {1, 12}, s13c[] = {1, 3};
    float* lstm[] = {lg, lc};
    int* lstms[]  = {s112, s13c};
    int lstmn[]   = {2, 2};
    ok &= check("lstm_cell", b_lstm, lstm, lstms, lstmn, 2);

    /* LayerNorm: x [1,D], gamma/beta [D], D=5. */
    float lx[5]  = {0.3f, -0.2f, 0.5f, 0.1f, -0.4f};
    float lgm[5] = {1.1f, 0.9f, 1.0f, 1.2f, 0.8f};
    float lbt[5] = {0.0f, 0.1f, -0.1f, 0.0f, 0.05f};
    int s15[] = {1, 5}, s5[] = {5};
    float* ln[] = {lx, lgm, lbt};
    int* lns[]  = {s15, s5, s5};
    int lnn[]   = {2, 1, 1};
    ok &= check("layernorm", b_layernorm, ln, lns, lnn, 3);

    /* RMSNorm: x [1,D], weight [D], D=5. */
    float rx[5] = {0.3f, -0.2f, 0.5f, 0.1f, -0.4f};
    float rw[5] = {1.1f, 0.9f, 1.0f, 1.2f, 0.8f};
    int rs[] = {1, 5}, rws[] = {5};
    float* rn[] = {rx, rw};
    int* rns[]  = {rs, rws};
    int rnn2[]  = {2, 1};
    ok &= check("rmsnorm", b_rmsnorm, rn, rns, rnn2, 2);

    printf(ok ? "Fused-op dtype safety passed.\n" : "FAILED.\n");
    return ok ? 0 : 1;
}
