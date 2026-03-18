#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "cml.h"
#include "nn/llama.h"

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  %-50s ", #name); \
    fflush(stdout); \
    if (test_##name()) { \
        tests_passed++; \
        printf("[PASS]\n"); \
    } else { \
        printf("[FAIL]\n"); \
    } \
} while(0)


static CMLLLaMAConfig tiny_test_config(void) {
    CMLLLaMAConfig config = {
        .vocab_size        = 64,
        .hidden_size       = 32,
        .intermediate_size = 64,
        .num_layers        = 2,
        .num_heads         = 4,
        .num_kv_heads      = 4,
        .max_seq_len       = 32,
        .rope_theta        = 10000.0f,
        .rms_norm_eps      = 1e-5f
    };
    return config;
}

static int init_layer_random(CMLLLaMALayer* layer, const CMLLLaMAConfig* config) {
    int hidden = config->hidden_size;
    int inter = config->intermediate_size;
    int head_dim = hidden / config->num_heads;
    int q_dim = config->num_heads * head_dim;
    int kv_dim = config->num_kv_heads * head_dim;

    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};

    int q_shape[] = {hidden, q_dim};
    int kv_shape[] = {hidden, kv_dim};
    int o_shape[] = {q_dim, hidden};
    int gate_shape[] = {hidden, inter};
    int down_shape[] = {inter, hidden};
    int norm_shape[] = {hidden};

    layer->q_proj = tensor_rand(q_shape, 2, &cfg);
    layer->k_proj = tensor_rand(kv_shape, 2, &cfg);
    layer->v_proj = tensor_rand(kv_shape, 2, &cfg);
    layer->o_proj = tensor_rand(o_shape, 2, &cfg);
    layer->gate_proj = tensor_rand(gate_shape, 2, &cfg);
    layer->up_proj = tensor_rand(gate_shape, 2, &cfg);
    layer->down_proj = tensor_rand(down_shape, 2, &cfg);
    layer->input_layernorm = tensor_ones(norm_shape, 1, &cfg);
    layer->post_attn_layernorm = tensor_ones(norm_shape, 1, &cfg);

    if (!layer->q_proj || !layer->k_proj || !layer->v_proj || !layer->o_proj ||
        !layer->gate_proj || !layer->up_proj || !layer->down_proj ||
        !layer->input_layernorm || !layer->post_attn_layernorm) {
        return -1;
    }
    return 0;
}

static int init_model_random(CMLLLaMAModel* model) {
    const CMLLLaMAConfig* config = &model->config;
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};

    int embed_shape[] = {config->vocab_size, config->hidden_size};
    int norm_shape[] = {config->hidden_size};
    int lm_head_shape[] = {config->hidden_size, config->vocab_size};

    model->embed_tokens = tensor_rand(embed_shape, 2, &cfg);
    model->norm = tensor_ones(norm_shape, 1, &cfg);
    model->lm_head = tensor_rand(lm_head_shape, 2, &cfg);

    if (!model->embed_tokens || !model->norm || !model->lm_head) return -1;

    for (int i = 0; i < model->num_layers; i++) {
        if (init_layer_random(model->layers[i], config) != 0) return -1;
    }

    model->weights_loaded = true;
    return 0;
}


static int test_config_7b(void) {
    CMLLLaMAConfig cfg = cml_llama_config_7b();
    if (cfg.hidden_size != 4096) return 0;
    if (cfg.num_layers != 32) return 0;
    if (cfg.num_heads != 32) return 0;
    if (cfg.num_kv_heads != 32) return 0;
    if (cfg.vocab_size != 32000) return 0;
    if (cfg.intermediate_size != 11008) return 0;
    if (cfg.max_seq_len != 2048) return 0;
    if (fabsf(cfg.rope_theta - 10000.0f) > 1e-3f) return 0;
    if (fabsf(cfg.rms_norm_eps - 1e-5f) > 1e-8f) return 0;
    return 1;
}

static int test_config_13b(void) {
    CMLLLaMAConfig cfg = cml_llama_config_13b();
    if (cfg.hidden_size != 5120) return 0;
    if (cfg.num_layers != 40) return 0;
    if (cfg.num_heads != 40) return 0;
    if (cfg.num_kv_heads != 40) return 0;
    if (cfg.intermediate_size != 13824) return 0;
    return 1;
}

static int test_config_70b(void) {
    CMLLLaMAConfig cfg = cml_llama_config_70b();
    if (cfg.hidden_size != 8192) return 0;
    if (cfg.num_layers != 80) return 0;
    if (cfg.num_heads != 64) return 0;
    if (cfg.num_kv_heads != 8) return 0;    /* GQA: 8 KV heads */
    if (cfg.intermediate_size != 28672) return 0;
    if (cfg.max_seq_len != 4096) return 0;
    return 1;
}


static int test_model_create_free(void) {
    CMLLLaMAConfig cfg = tiny_test_config();
    CMLLLaMAModel* model = cml_llama_create(&cfg);
    if (!model) return 0;
    if (model->num_layers != cfg.num_layers) { cml_llama_free(model); return 0; }
    if (model->weights_loaded) { cml_llama_free(model); return 0; }
    if (model->current_seq_len != 0) { cml_llama_free(model); return 0; }
    if (!model->layers) { cml_llama_free(model); return 0; }
    if (!model->layers[0]) { cml_llama_free(model); return 0; }
    if (!model->layers[1]) { cml_llama_free(model); return 0; }
    if (!model->layers[0]->kv_cache) { cml_llama_free(model); return 0; }
    cml_llama_free(model);
    return 1;
}

static int test_model_create_null_config(void) {
    CMLLLaMAModel* model = cml_llama_create(NULL);
    if (model != NULL) { cml_llama_free(model); return 0; }
    return 1;
}

static int test_model_free_null(void) {
    /* Should not crash */
    cml_llama_free(NULL);
    return 1;
}


static int test_forward_basic(void) {
    CMLLLaMAConfig cfg = tiny_test_config();
    CMLLLaMAModel* model = cml_llama_create(&cfg);
    if (!model) return 0;

    if (init_model_random(model) != 0) {
        cml_llama_free(model);
        return 0;
    }

    /* Run forward with a small sequence */
    int tokens[] = {1, 5, 10, 3};
    int seq_len = 4;
    Tensor* logits = cml_llama_forward(model, tokens, seq_len);

    if (!logits) {
        cml_llama_free(model);
        return 0;
    }

    /* Logits shape should be [seq_len, vocab_size] */
    if (logits->ndim != 2) { tensor_free(logits); cml_llama_free(model); return 0; }
    if (logits->shape[0] != seq_len) { tensor_free(logits); cml_llama_free(model); return 0; }
    if (logits->shape[1] != cfg.vocab_size) { tensor_free(logits); cml_llama_free(model); return 0; }

    /* Verify current_seq_len updated */
    if (model->current_seq_len != seq_len) { tensor_free(logits); cml_llama_free(model); return 0; }

    tensor_free(logits);
    cml_llama_free(model);
    return 1;
}

static int test_forward_single_token(void) {
    CMLLLaMAConfig cfg = tiny_test_config();
    CMLLLaMAModel* model = cml_llama_create(&cfg);
    if (!model) return 0;

    if (init_model_random(model) != 0) {
        cml_llama_free(model);
        return 0;
    }

    int tokens[] = {7};
    Tensor* logits = cml_llama_forward(model, tokens, 1);
    if (!logits) { cml_llama_free(model); return 0; }
    if (logits->ndim != 2) { tensor_free(logits); cml_llama_free(model); return 0; }
    if (logits->shape[0] != 1) { tensor_free(logits); cml_llama_free(model); return 0; }
    if (logits->shape[1] != cfg.vocab_size) { tensor_free(logits); cml_llama_free(model); return 0; }

    tensor_free(logits);
    cml_llama_free(model);
    return 1;
}

static int test_forward_no_weights(void) {
    CMLLLaMAConfig cfg = tiny_test_config();
    CMLLLaMAModel* model = cml_llama_create(&cfg);
    if (!model) return 0;

    /* Forward without loading weights should fail gracefully */
    int tokens[] = {1, 2};
    Tensor* logits = cml_llama_forward(model, tokens, 2);
    /* Should return NULL since embed_tokens is not loaded */
    if (logits != NULL) { tensor_free(logits); cml_llama_free(model); return 0; }

    cml_llama_free(model);
    return 1;
}

static int test_forward_invalid_args(void) {
    CMLLLaMAConfig cfg = tiny_test_config();
    CMLLLaMAModel* model = cml_llama_create(&cfg);
    if (!model) return 0;

    /* NULL token_ids */
    Tensor* logits = cml_llama_forward(model, NULL, 4);
    if (logits != NULL) { tensor_free(logits); cml_llama_free(model); return 0; }

    /* Zero seq_len */
    int tokens[] = {1};
    logits = cml_llama_forward(model, tokens, 0);
    if (logits != NULL) { tensor_free(logits); cml_llama_free(model); return 0; }

    /* NULL model */
    logits = cml_llama_forward(NULL, tokens, 1);
    if (logits != NULL) { tensor_free(logits); cml_llama_free(model); return 0; }

    cml_llama_free(model);
    return 1;
}


static int test_sample_greedy(void) {
    /* Create 1D logits with a clear maximum */
    float data[] = {0.1f, 0.5f, 2.0f, 0.3f, -1.0f};
    int shape[] = {5};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    Tensor* logits = tensor_from_data(data, shape, 1, &cfg);
    if (!logits) return 0;

    CMLGenerationConfig gen_cfg = cml_generation_default_config();
    gen_cfg.do_sample = false; /* greedy */

    int token = cml_llama_sample_token(logits, &gen_cfg);
    tensor_free(logits);

    /* The maximum logit is at index 2 (value 2.0) */
    if (token != 2) return 0;
    return 1;
}

static int test_sample_greedy_2d(void) {
    /* Create 2D logits [2, 5] - should sample from last row */
    float data[] = {
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f,   /* row 0 - max at 0 */
        0.0f, 0.0f, 0.0f, 5.0f, 0.0f     /* row 1 - max at 3 */
    };
    int shape[] = {2, 5};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    Tensor* logits = tensor_from_data(data, shape, 2, &cfg);
    if (!logits) return 0;

    CMLGenerationConfig gen_cfg = cml_generation_default_config();
    gen_cfg.do_sample = false;

    int token = cml_llama_sample_token(logits, &gen_cfg);
    tensor_free(logits);

    /* Should return 3 (max of last row) */
    if (token != 3) return 0;
    return 1;
}

static int test_sample_with_temperature(void) {
    /* Sampling with temperature should still produce valid token IDs */
    float data[] = {1.0f, 2.0f, 3.0f, 0.5f};
    int shape[] = {4};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    Tensor* logits = tensor_from_data(data, shape, 1, &cfg);
    if (!logits) return 0;

    CMLGenerationConfig gen_cfg = cml_generation_default_config();
    gen_cfg.do_sample = true;
    gen_cfg.temperature = 0.5f;
    gen_cfg.top_k = 4;
    gen_cfg.top_p = 1.0f;

    int token = cml_llama_sample_token(logits, &gen_cfg);
    tensor_free(logits);

    /* Token should be a valid index */
    if (token < 0 || token >= 4) return 0;
    return 1;
}

static int test_sample_null_args(void) {
    CMLGenerationConfig gen_cfg = cml_generation_default_config();
    int token = cml_llama_sample_token(NULL, &gen_cfg);
    if (token != -1) return 0;

    float data[] = {1.0f};
    int shape[] = {1};
    TensorConfig cfg = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                        .has_dtype = true, .has_device = true};
    Tensor* logits = tensor_from_data(data, shape, 1, &cfg);
    if (!logits) return 0;

    token = cml_llama_sample_token(logits, NULL);
    tensor_free(logits);
    if (token != -1) return 0;
    return 1;
}


static int test_generation_default_config(void) {
    CMLGenerationConfig cfg = cml_generation_default_config();
    if (fabsf(cfg.temperature - 0.8f) > 1e-5f) return 0;
    if (fabsf(cfg.top_p - 0.9f) > 1e-5f) return 0;
    if (cfg.top_k != 40) return 0;
    if (cfg.max_new_tokens != 256) return 0;
    if (cfg.eos_token_id != 2) return 0;
    if (!cfg.do_sample) return 0;
    return 1;
}


static int test_model_reset(void) {
    CMLLLaMAConfig cfg = tiny_test_config();
    CMLLLaMAModel* model = cml_llama_create(&cfg);
    if (!model) return 0;

    if (init_model_random(model) != 0) {
        cml_llama_free(model);
        return 0;
    }

    /* Run a forward to advance current_seq_len */
    int tokens[] = {1, 2, 3};
    Tensor* logits = cml_llama_forward(model, tokens, 3);
    if (!logits) { cml_llama_free(model); return 0; }
    tensor_free(logits);

    if (model->current_seq_len != 3) { cml_llama_free(model); return 0; }

    /* Reset */
    cml_llama_reset(model);
    if (model->current_seq_len != 0) { cml_llama_free(model); return 0; }

    /* KV caches should be reset */
    for (int i = 0; i < model->num_layers; i++) {
        if (model->layers[i]->kv_cache->current_len != 0) {
            cml_llama_free(model);
            return 0;
        }
    }

    cml_llama_free(model);
    return 1;
}


static int test_print_config(void) {
    CMLLLaMAConfig cfg = cml_llama_config_7b();
    /* Should not crash */
    cml_llama_print_config(&cfg);
    cml_llama_print_config(NULL); /* Should handle NULL gracefully */
    return 1;
}


static int test_generation_result_free(void) {
    /* Should handle NULL gracefully */
    cml_generation_result_free(NULL);

    /* Create and free a result */
    CMLGenerationResult* result = (CMLGenerationResult*)calloc(1, sizeof(CMLGenerationResult));
    if (!result) return 0;
    result->token_ids = (int*)malloc(4 * sizeof(int));
    result->num_tokens = 4;
    result->text = (char*)malloc(16);
    if (result->text) strcpy(result->text, "hello");
    cml_generation_result_free(result);
    return 1;
}


static int test_model_gqa_config(void) {
    /* Verify a model with GQA (num_kv_heads < num_heads) creates properly */
    CMLLLaMAConfig cfg = tiny_test_config();
    cfg.num_heads = 8;
    cfg.num_kv_heads = 2;
    cfg.hidden_size = 64; /* head_dim = 64/8 = 8 */

    CMLLLaMAModel* model = cml_llama_create(&cfg);
    if (!model) return 0;

    /* Verify KV cache was created with correct kv_heads */
    if (model->layers[0]->kv_cache->num_kv_heads != 2) {
        cml_llama_free(model);
        return 0;
    }

    /* head_dim should be 8 */
    if (model->layers[0]->kv_cache->head_dim != 8) {
        cml_llama_free(model);
        return 0;
    }

    cml_llama_free(model);
    return 1;
}


int main(void) {
    printf("test_llama\n\n");

    /* Config tests */
    TEST(config_7b);
    TEST(config_13b);
    TEST(config_70b);

    /* Model create/free tests */
    TEST(model_create_free);
    TEST(model_create_null_config);
    TEST(model_free_null);
    TEST(model_gqa_config);

    /* Forward pass tests */
    TEST(forward_basic);
    TEST(forward_single_token);
    TEST(forward_no_weights);
    TEST(forward_invalid_args);

    /* Sampling tests */
    TEST(sample_greedy);
    TEST(sample_greedy_2d);
    TEST(sample_with_temperature);
    TEST(sample_null_args);

    /* Generation config tests */
    TEST(generation_default_config);
    TEST(generation_result_free);

    /* Utility tests */
    TEST(model_reset);
    TEST(print_config);

    printf("\n%d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
