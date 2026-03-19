#include "nn/llm_ops.h"
#include "tensor/tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, tol, msg) do { \
    float _a = (a), _b = (b); \
    if (fabsf(_a - _b) > (tol)) { \
        printf("  FAIL: %s: %.6f != %.6f (tol=%.6f) (line %d)\n", \
               msg, _a, _b, (float)(tol), __LINE__); \
        tests_failed++; \
        return; \
    } \
} while(0)

static Tensor* make_rand_tensor(int b, int s, int d) {
    size_t n = (size_t)b * s * d;
    float* buf = (float*)malloc(n * sizeof(float));
    if (!buf) return NULL;
    for (size_t i = 0; i < n; i++) {
        buf[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.1f;
    }
    int shape[] = {b, s, d};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    Tensor* t = tensor_from_data(buf, shape, 3, &cfg);
    free(buf);
    return t;
}

static void test_flash_vs_standard(void) {
    printf("test_flash_vs_standard...\n");

    int batch = 1, seq = 8, num_heads = 4, num_kv_heads = 2, head_dim = 16;
    int q_dim = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;

    Tensor* Q = make_rand_tensor(batch, seq, q_dim);
    Tensor* K = make_rand_tensor(batch, seq, kv_dim);
    Tensor* V = make_rand_tensor(batch, seq, kv_dim);
    ASSERT(Q && K && V, "tensor creation");

    CMLGQAConfig gqa_cfg = {
        .num_heads = num_heads, .num_kv_heads = num_kv_heads,
        .head_dim = head_dim, .scale = 0.0f, .causal = true,
        .window_size = 0
    };

    /* Standard GQA */
    Tensor* std_out = cml_gqa_forward(Q, K, V, &gqa_cfg, NULL);
    ASSERT(std_out != NULL, "standard GQA");

    /* Flash GQA */
    CMLFlashAttentionConfig flash_cfg = cml_flash_attention_default_config();
    flash_cfg.tile_size_q = 4;
    flash_cfg.tile_size_kv = 4;
    Tensor* flash_out = cml_gqa_flash_forward(Q, K, V, &gqa_cfg, &flash_cfg);
    ASSERT(flash_out != NULL, "flash GQA");

    /* Compare outputs */
    tensor_ensure_executed(std_out);
    tensor_ensure_executed(flash_out);
    float* std_data = (float*)tensor_data_ptr(std_out);
    float* flash_data = (float*)tensor_data_ptr(flash_out);
    ASSERT(std_data && flash_data, "data pointers");

    size_t n = (size_t)batch * seq * q_dim;
    float max_err = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float err = fabsf(std_data[i] - flash_data[i]);
        if (err > max_err) max_err = err;
    }
    printf("  max error between standard and flash: %.8f\n", max_err);
    ASSERT(max_err < 1e-4f, "flash matches standard");

    tensor_free(Q); tensor_free(K); tensor_free(V);
    tensor_free(std_out); tensor_free(flash_out);
    tests_passed++;
    printf("  PASS\n");
}

static void test_flash_cached(void) {
    printf("test_flash_cached...\n");

    int num_heads = 4, num_kv_heads = 2, head_dim = 16;
    int q_dim = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;

    CMLKVCache* cache = cml_kv_cache_create(64, num_kv_heads, head_dim);
    ASSERT(cache != NULL, "cache create");

    CMLGQAConfig gqa_cfg = {
        .num_heads = num_heads, .num_kv_heads = num_kv_heads,
        .head_dim = head_dim, .scale = 0.0f, .causal = true,
        .window_size = 0
    };
    CMLFlashAttentionConfig flash_cfg = cml_flash_attention_default_config();
    flash_cfg.tile_size_q = 4;
    flash_cfg.tile_size_kv = 4;

    /* Append a few tokens via flash cached */
    for (int step = 0; step < 4; step++) {
        Tensor* Q = make_rand_tensor(1, 1, q_dim);
        Tensor* K = make_rand_tensor(1, 1, kv_dim);
        Tensor* V = make_rand_tensor(1, 1, kv_dim);
        ASSERT(Q && K && V, "token tensors");

        Tensor* out = cml_gqa_flash_forward_cached(Q, K, V, cache, &gqa_cfg, &flash_cfg);
        ASSERT(out != NULL, "flash cached forward");
        ASSERT(out->shape[1] == 1, "output seq=1");

        tensor_free(Q); tensor_free(K); tensor_free(V);
        tensor_free(out);
    }

    ASSERT(cache->current_len == 4, "cache length");
    cml_kv_cache_free(cache);
    tests_passed++;
    printf("  PASS\n");
}

static void test_sliding_window(void) {
    printf("test_sliding_window...\n");

    int batch = 1, seq = 8, num_heads = 2, num_kv_heads = 2, head_dim = 8;
    int dim = num_heads * head_dim;

    Tensor* Q = make_rand_tensor(batch, seq, dim);
    Tensor* K = make_rand_tensor(batch, seq, dim);
    Tensor* V = make_rand_tensor(batch, seq, dim);
    ASSERT(Q && K && V, "tensor creation");

    /* Full causal attention */
    CMLGQAConfig cfg_full = {
        .num_heads = num_heads, .num_kv_heads = num_kv_heads,
        .head_dim = head_dim, .scale = 0.0f, .causal = true,
        .window_size = 0
    };
    Tensor* out_full = cml_gqa_forward(Q, K, V, &cfg_full, NULL);
    ASSERT(out_full != NULL, "full attention");

    /* Sliding window = 3 */
    CMLGQAConfig cfg_win = cfg_full;
    cfg_win.window_size = 3;
    Tensor* out_win = cml_gqa_forward(Q, K, V, &cfg_win, NULL);
    ASSERT(out_win != NULL, "windowed attention");

    /* Outputs should differ for positions > window_size */
    tensor_ensure_executed(out_full);
    tensor_ensure_executed(out_win);
    float* full_data = (float*)tensor_data_ptr(out_full);
    float* win_data = (float*)tensor_data_ptr(out_win);

    /* Position 0 should be the same (only attends to itself) */
    float err_pos0 = 0.0f;
    for (int d = 0; d < dim; d++) {
        err_pos0 += fabsf(full_data[d] - win_data[d]);
    }
    printf("  position 0 diff: %.6f (should be ~0)\n", err_pos0);
    ASSERT(err_pos0 < 1e-5f, "pos 0 matches");

    /* Later positions should differ if kv_len > window */
    float diff_later = 0.0f;
    int pos = seq - 1;
    for (int d = 0; d < dim; d++) {
        diff_later += fabsf(full_data[pos * dim + d] - win_data[pos * dim + d]);
    }
    printf("  position %d diff: %.6f (should be > 0)\n", pos, diff_later);
    ASSERT(diff_later > 1e-6f, "window makes difference");

    /* Flash attention with window */
    CMLFlashAttentionConfig flash_cfg = cml_flash_attention_default_config();
    flash_cfg.tile_size_q = 4;
    flash_cfg.tile_size_kv = 4;
    Tensor* out_flash_win = cml_gqa_flash_forward(Q, K, V, &cfg_win, &flash_cfg);
    ASSERT(out_flash_win != NULL, "flash with window");

    tensor_ensure_executed(out_flash_win);
    float* flash_win_data = (float*)tensor_data_ptr(out_flash_win);
    float max_err = 0.0f;
    size_t n = (size_t)batch * seq * dim;
    for (size_t i = 0; i < n; i++) {
        float e = fabsf(win_data[i] - flash_win_data[i]);
        if (e > max_err) max_err = e;
    }
    printf("  flash vs standard window max err: %.8f\n", max_err);
    ASSERT(max_err < 1e-4f, "flash window matches standard window");

    tensor_free(Q); tensor_free(K); tensor_free(V);
    tensor_free(out_full); tensor_free(out_win); tensor_free(out_flash_win);
    tests_passed++;
    printf("  PASS\n");
}

static void test_flash_large_tiles(void) {
    printf("test_flash_large_tiles...\n");

    int batch = 1, seq = 4, num_heads = 2, num_kv_heads = 1, head_dim = 8;
    int q_dim = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;

    Tensor* Q = make_rand_tensor(batch, seq, q_dim);
    Tensor* K = make_rand_tensor(batch, seq, kv_dim);
    Tensor* V = make_rand_tensor(batch, seq, kv_dim);
    ASSERT(Q && K && V, "tensor creation");

    CMLGQAConfig cfg = {
        .num_heads = num_heads, .num_kv_heads = num_kv_heads,
        .head_dim = head_dim, .scale = 0.0f, .causal = true,
        .window_size = 0
    };

    Tensor* std_out = cml_gqa_forward(Q, K, V, &cfg, NULL);

    CMLFlashAttentionConfig flash_cfg = {
        .tile_size_q = 128,  /* larger than seq_len */
        .tile_size_kv = 128,
        .enabled = true
    };
    Tensor* flash_out = cml_gqa_flash_forward(Q, K, V, &cfg, &flash_cfg);
    ASSERT(std_out && flash_out, "forward passes");

    tensor_ensure_executed(std_out);
    tensor_ensure_executed(flash_out);
    float* sd = (float*)tensor_data_ptr(std_out);
    float* fd = (float*)tensor_data_ptr(flash_out);

    float max_err = 0.0f;
    for (size_t i = 0; i < (size_t)batch * seq * q_dim; i++) {
        float e = fabsf(sd[i] - fd[i]);
        if (e > max_err) max_err = e;
    }
    printf("  large tile max error: %.8f\n", max_err);
    ASSERT(max_err < 1e-4f, "large tiles match");

    tensor_free(Q); tensor_free(K); tensor_free(V);
    tensor_free(std_out); tensor_free(flash_out);
    tests_passed++;
    printf("  PASS\n");
}

int main(void) {
    printf("Flash Attention GQA Tests\n\n");
    srand(42);

    test_flash_vs_standard();
    test_flash_cached();
    test_sliding_window();
    test_flash_large_tiles();

    printf("\nResults: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
