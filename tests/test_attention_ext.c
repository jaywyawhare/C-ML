#include "nn/llm_ops.h"
#include "tensor/tensor.h"
#include "alloc/cml_allocator.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "test_assert.h"

static Tensor* rand3(int b, int s, int d) {
    size_t n   = (size_t)b * s * d;
    float* buf = (float*)cml_malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++)
        buf[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.5f;
    int shape[]      = {b, s, d};
    TensorConfig cfg = {
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* t = tensor_from_data(buf, shape, 3, &cfg);
    cml_free(buf);
    return t;
}

/* Independent O(n^2) reference matching the fold-kernel masking/bias semantics exactly. */
static void ref_attention(const float* Q, const float* K, const float* V, int seq_q, int seq_kv,
                          int H, int KVH, int D, bool causal, int window, int num_sink,
                          bool use_alibi, float* out) {
    int groups  = H / KVH;
    float scale = 1.0f / sqrtf((float)D);
    float alibi[64];
    if (use_alibi)
        cml_alibi_slopes(H, alibi);
    float* sc = (float*)cml_malloc((size_t)seq_kv * sizeof(float));
    for (int h = 0; h < H; h++) {
        int kvh     = h / groups;
        float slope = use_alibi ? alibi[h] : 0.0f;
        for (int qi = 0; qi < seq_q; qi++) {
            float mx = -1e30f;
            for (int kj = 0; kj < seq_kv; kj++) {
                float dot = 0.0f;
                for (int d = 0; d < D; d++)
                    dot += Q[(qi * H + h) * D + d] * K[(kj * KVH + kvh) * D + d];
                float s = dot * scale - slope * (float)(qi - kj);
                if (causal && kj > qi)
                    s = -1e30f;
                if (window > 0 && kj >= num_sink) {
                    int dist = qi - kj;
                    if (dist < 0)
                        dist = -dist;
                    if (dist > window)
                        s = -1e30f;
                }
                sc[kj] = s;
                if (s > mx)
                    mx = s;
            }
            float sum = 0.0f;
            for (int kj = 0; kj < seq_kv; kj++) {
                sc[kj] = expf(sc[kj] - mx);
                sum += sc[kj];
            }
            float inv = (sum > 0.0f) ? 1.0f / sum : 0.0f;
            for (int d = 0; d < D; d++) {
                float a = 0.0f;
                for (int kj = 0; kj < seq_kv; kj++)
                    a += sc[kj] * V[(kj * KVH + kvh) * D + d];
                out[(qi * H + h) * D + d] = a * inv;
            }
        }
    }
    cml_free(sc);
}

static void check_flash(const char* name, int H, int KVH, int D, int seq, bool causal, int window,
                        int num_sink, bool use_alibi) {
    printf("%s...\n", name);
    Tensor* Q = rand3(1, seq, H * D);
    Tensor* K = rand3(1, seq, KVH * D);
    Tensor* V = rand3(1, seq, KVH * D);

    CMLGQAConfig cfg           = {.num_heads       = H,
                                  .num_kv_heads    = KVH,
                                  .head_dim        = D,
                                  .scale           = 0.0f,
                                  .causal          = causal,
                                  .window_size     = window,
                                  .num_sink_tokens = num_sink,
                                  .use_alibi       = use_alibi};
    CMLFlashAttentionConfig fc = cml_flash_attention_default_config();
    Tensor* out                = cml_gqa_flash_forward(Q, K, V, &cfg, &fc);
    ASSERT(out != NULL, "flash forward");

    float* ref = (float*)cml_malloc((size_t)seq * H * D * sizeof(float));
    ref_attention((float*)tensor_data_ptr(Q), (float*)tensor_data_ptr(K),
                  (float*)tensor_data_ptr(V), seq, seq, H, KVH, D, causal, window, num_sink,
                  use_alibi, ref);
    const float* op = (const float*)tensor_data_ptr(out);
    int ok          = 1;
    for (int i = 0; i < seq * H * D; i++)
        if (fabsf(op[i] - ref[i]) > 2e-3f)
            ok = 0;
    ASSERT(ok, "flash matches reference");
    cml_free(ref);
    tensor_free(Q);
    tensor_free(K);
    tensor_free(V);
    tensor_free(out);
    if (ok)
        tests_passed++;
}

static void test_alibi_slopes(void) {
    printf("test_alibi_slopes...\n");
    float s8[8], s4[4];
    cml_alibi_slopes(8, s8);
    cml_alibi_slopes(4, s4);
    for (int i = 0; i < 8; i++)
        ASSERT(fabsf(s8[i] - powf(2.0f, -(float)(i + 1))) < 1e-6f, "slopes(8) == 2^-(i+1)");
    float w4[4] = {0.25f, 0.0625f, 0.015625f, 0.00390625f};
    for (int i = 0; i < 4; i++)
        ASSERT(fabsf(s4[i] - w4[i]) < 1e-6f, "slopes(4)");
    tests_passed++;
}

static void test_rope_scaling(void) {
    printf("test_rope_scaling...\n");
    int seq = 4, dim = 8;
    for (int mode = 0; mode < 3; mode++) {
        float factor = 4.0f;
        /* Build x with every rotation pair set to (1,0). */
        float* buf = (float*)cml_malloc((size_t)seq * dim * sizeof(float));
        for (int s = 0; s < seq; s++)
            for (int i = 0; i < dim / 2; i++) {
                buf[s * dim + 2 * i]     = 1.0f;
                buf[s * dim + 2 * i + 1] = 0.0f;
            }
        int shape[]      = {seq, dim};
        TensorConfig cfg = {
            .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
        Tensor* x = tensor_from_data(buf, shape, 2, &cfg);
        cml_free(buf);

        CMLRoPEConfig rc = {.dim            = dim,
                            .max_seq_len    = seq,
                            .base           = 10000.0f,
                            .scaling        = (CMLRoPEScalingType)mode,
                            .scaling_factor = factor};
        cml_rope_forward(x, 0, &rc);

        /* Recompute expected angles from the scaling definition, independently. */
        float base = 10000.0f, pos_scale = 1.0f;
        if (mode == CML_ROPE_SCALING_LINEAR)
            pos_scale = 1.0f / factor;
        else if (mode == CML_ROPE_SCALING_NTK)
            base *= powf(factor, (float)dim / (float)(dim - 2));

        float* xd = (float*)tensor_data_ptr(x);
        int ok    = 1;
        for (int s = 0; s < seq; s++)
            for (int i = 0; i < dim / 2; i++) {
                float freq  = 1.0f / powf(base, (float)(2 * i) / (float)dim);
                float theta = (float)s * pos_scale * freq;
                if (fabsf(xd[s * dim + 2 * i] - cosf(theta)) > 1e-4f ||
                    fabsf(xd[s * dim + 2 * i + 1] - sinf(theta)) > 1e-4f)
                    ok = 0;
            }
        ASSERT(ok, "rope scaling angles match closed form");
        tensor_free(x);
    }
    tests_passed++;
}

int main(void) {
    printf("Attention extensions (ALiBi / RoPE scaling / streaming sink)\n\n");
    srand(123);
    test_alibi_slopes();
    test_rope_scaling();
    check_flash("flash_causal", 4, 2, 16, 12, true, 0, 0, false);
    check_flash("flash_causal_alibi", 4, 2, 16, 12, true, 0, 0, true);
    check_flash("flash_window_sink", 4, 2, 16, 16, true, 4, 2, false);
    check_flash("flash_window_sink_alibi", 4, 2, 16, 16, true, 4, 2, true);
    check_flash("flash_noncausal", 4, 4, 16, 8, false, 0, 0, false);
    printf("\nResults: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
