/**
 * The fused attention op (uop_sdpa / UOP_SDPA) must match the explicit
 * matmul/scale/mask/softmax/matmul compose it replaces, for both the forward
 * output and the q/k/v gradients, with and without a mask and additive bias.
 * Also checks that the fused forward runs natively in bf16.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cml.h"
#include "autograd/autograd.h"
#include "ops/uops.h"

#define B 1
#define H 2
#define SQ 3
#define SK 3
#define DH 4
#define NEL (B * H * SQ * DH)

static TensorConfig CFG = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};

static float QD[NEL], KD[NEL], VD[NEL];
static float MASKD[B * H * SQ * SK], BIASD[B * H * SQ * SK];

static void init_data(void) {
    for (int i = 0; i < NEL; i++) {
        QD[i] = 0.1f * (float)((i * 7) % 11 - 5);
        KD[i] = 0.1f * (float)((i * 5) % 9 - 4);
        VD[i] = 0.1f * (float)((i * 3) % 13 - 6);
    }
    for (int i = 0; i < B * H * SQ * SK; i++) {
        MASKD[i] = (i % 4 == 0) ? 1.0f : 0.0f; /* masked_fill where != 0 */
        BIASD[i] = 0.05f * (float)((i % 5) - 2);
    }
}

static int qs[] = {B, H, SQ, DH}, ks[] = {B, H, SK, DH}, ms[] = {B, H, SQ, SK};

/* Independent naive attention reference straight from the arrays. */
static void naive(int with_mask, int with_bias, float scale, float* out) {
    for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++)
            for (int sq = 0; sq < SQ; sq++) {
                float s[SK], mx = -1e30f, sum = 0.0f;
                for (int sk = 0; sk < SK; sk++) {
                    float dot = 0.0f;
                    for (int d = 0; d < DH; d++)
                        dot += QD[((b * H + h) * SQ + sq) * DH + d] *
                               KD[((b * H + h) * SK + sk) * DH + d];
                    s[sk] = dot * scale;
                    if (with_bias)
                        s[sk] += BIASD[((b * H + h) * SQ + sq) * SK + sk];
                    if (with_mask && MASKD[((b * H + h) * SQ + sq) * SK + sk] != 0.0f)
                        s[sk] = -1e9f;
                    if (s[sk] > mx)
                        mx = s[sk];
                }
                for (int sk = 0; sk < SK; sk++) {
                    s[sk] = expf(s[sk] - mx);
                    sum += s[sk];
                }
                for (int d = 0; d < DH; d++) {
                    float acc = 0.0f;
                    for (int sk = 0; sk < SK; sk++)
                        acc += (s[sk] / sum) * VD[((b * H + h) * SK + sk) * DH + d];
                    out[((b * H + h) * SQ + sq) * DH + d] = acc;
                }
            }
}

/* Forward: fused vs an independent naive attention (ground truth). */
static int check_forward(const char* name, int with_mask, int with_bias) {
    float scale = 1.0f / sqrtf((float)DH);
    float nref[NEL], got[NEL];
    naive(with_mask, with_bias, scale, nref);
    Tensor* q       = cml_tensor(QD, qs, 4, &CFG);
    Tensor* k       = cml_tensor(KD, ks, 4, &CFG);
    Tensor* v       = cml_tensor(VD, ks, 4, &CFG);
    Tensor* mask    = with_mask ? cml_tensor(MASKD, ms, 4, &CFG) : NULL;
    Tensor* bias    = with_bias ? cml_tensor(BIASD, ms, 4, &CFG) : NULL;
    Tensor* o       = uop_sdpa(q, k, v, mask, bias, scale);
    const float* od = (const float*)tensor_data_ptr(o);
    for (int i = 0; i < NEL; i++)
        got[i] = od ? od[i] : 0.0f / 0.0f;
    cml_reset_ir_context();
    float md = 0.0f;
    for (int i = 0; i < NEL; i++)
        md = fmaxf(md, fabsf(nref[i] - got[i]));
    int ok = md < 1e-5f;
    printf("  %-16s fwd fused-vs-naive maxdiff=%.3e %s\n", name, (double)md, ok ? "PASS" : "FAIL");
    return ok;
}

/* Scalar loss sum(fused SDPA) for a finite-difference grad-check over `buf`. */
static float fwd_loss(int with_mask, int with_bias, float scale) {
    Tensor* q    = cml_tensor(QD, qs, 4, &CFG);
    Tensor* k    = cml_tensor(KD, ks, 4, &CFG);
    Tensor* v    = cml_tensor(VD, ks, 4, &CFG);
    Tensor* mask = with_mask ? cml_tensor(MASKD, ms, 4, &CFG) : NULL;
    Tensor* bias = with_bias ? cml_tensor(BIASD, ms, 4, &CFG) : NULL;
    Tensor* o    = uop_sdpa(q, k, v, mask, bias, scale);
    Tensor* loss = uop_sum(o, &(ReduceParams){0});
    float val    = tensor_get_float(loss, 0);
    cml_reset_ir_context();
    return val;
}

static float max_fd_diff(int wm, int wb, float scale, float* buf, const float* analytic) {
    const float eps = 1e-3f;
    float md        = 0.0f;
    for (int i = 0; i < NEL; i++) {
        float o   = buf[i];
        buf[i]    = o + eps;
        float lp  = fwd_loss(wm, wb, scale);
        buf[i]    = o - eps;
        float lm  = fwd_loss(wm, wb, scale);
        buf[i]    = o;
        float num = (lp - lm) / (2.0f * eps);
        md        = fmaxf(md, fabsf(num - analytic[i]));
    }
    return md;
}

/* Gradient check: analytic dq/dk/dv vs finite differences of the fused forward. */
static int check_grad(const char* name, int with_mask, int with_bias) {
    float scale = 1.0f / sqrtf((float)DH);
    float aq[NEL], ak[NEL], av[NEL];
    {
        Tensor* q        = cml_tensor(QD, qs, 4, &CFG);
        q->requires_grad = true;
        Tensor* k        = cml_tensor(KD, ks, 4, &CFG);
        k->requires_grad = true;
        Tensor* v        = cml_tensor(VD, ks, 4, &CFG);
        v->requires_grad = true;
        Tensor* mask     = with_mask ? cml_tensor(MASKD, ms, 4, &CFG) : NULL;
        Tensor* bias     = with_bias ? cml_tensor(BIASD, ms, 4, &CFG) : NULL;
        Tensor* o        = uop_sdpa(q, k, v, mask, bias, scale);
        Tensor* loss     = uop_sum(o, &(ReduceParams){0});
        tensor_backward(loss, NULL, false, false);
        const float* dq = (const float*)tensor_data_ptr(q->grad);
        const float* dk = (const float*)tensor_data_ptr(k->grad);
        const float* dv = (const float*)tensor_data_ptr(v->grad);
        for (int i = 0; i < NEL; i++) {
            aq[i] = dq ? dq[i] : 0.0f;
            ak[i] = dk ? dk[i] : 0.0f;
            av[i] = dv ? dv[i] : 0.0f;
        }
        cml_reset_ir_context();
    }
    float dq = max_fd_diff(with_mask, with_bias, scale, QD, aq);
    float dk = max_fd_diff(with_mask, with_bias, scale, KD, ak);
    float dv = max_fd_diff(with_mask, with_bias, scale, VD, av);
    float md = fmaxf(dq, fmaxf(dk, dv));
    int ok   = md < 5e-3f;
    printf("  %-16s grad vs finite-diff maxdiff=%.3e %s\n", name, (double)md, ok ? "PASS" : "FAIL");
    return ok;
}

/* Native bf16 forward matches the f32 reference within bf16 precision. */
static int check_bf16(void) {
    float scale = 1.0f / sqrtf((float)DH);
    float ref[NEL];
    {
        Tensor* q       = cml_tensor(QD, qs, 4, &CFG);
        Tensor* k       = cml_tensor(KD, ks, 4, &CFG);
        Tensor* v       = cml_tensor(VD, ks, 4, &CFG);
        Tensor* o       = uop_sdpa(q, k, v, NULL, NULL, scale);
        const float* od = (const float*)tensor_data_ptr(o);
        for (int i = 0; i < NEL; i++)
            ref[i] = od[i];
        cml_reset_ir_context();
    }
    Tensor* q = cml_cast(cml_tensor(QD, qs, 4, &CFG), DTYPE_BFLOAT16);
    Tensor* k = cml_cast(cml_tensor(KD, ks, 4, &CFG), DTYPE_BFLOAT16);
    Tensor* v = cml_cast(cml_tensor(VD, ks, 4, &CFG), DTYPE_BFLOAT16);
    Tensor* o = uop_sdpa(q, k, v, NULL, NULL, scale);
    float md  = 0.0f;
    for (int i = 0; i < NEL; i++)
        md = fmaxf(md, fabsf(tensor_get_float(o, i) - ref[i]));
    cml_reset_ir_context();
    int ok = md < 3e-2f;
    printf("  %-16s bf16-vs-f32 maxdiff=%.3e %s\n", "native-bf16", (double)md,
           ok ? "PASS" : "FAIL");
    return ok;
}

int main(void) {
    cml_init();
    init_data();
    printf("Fused SDPA (UOP_SDPA) vs compose\n");
    int ok = 1;
    ok &= check_forward("plain", 0, 0);
    ok &= check_forward("mask", 1, 0);
    ok &= check_forward("bias", 0, 1);
    ok &= check_forward("mask+bias", 1, 1);
    ok &= check_grad("plain", 0, 0);
    ok &= check_grad("mask", 1, 0);
    ok &= check_grad("bias", 0, 1);
    ok &= check_bf16();
    printf(ok ? "Fused SDPA equivalence holds.\n" : "FAILED.\n");
    return ok ? 0 : 1;
}
