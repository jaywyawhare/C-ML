#include "cml.h"
#include "nn/llama.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void print_usage(const char* prog) {
    printf("Usage: %s <model.gguf> [options]\n\n", prog);
    printf("LLaMA inference using C-ML\n\n");
    printf("Options:\n");
    printf("  -p, --prompt <text>     Input prompt (default: \"Hello, world!\")\n");
    printf("  -t, --temperature <f>   Sampling temperature (default: 0.8)\n");
    printf("  -n, --max-tokens <n>    Maximum tokens to generate (default: 256)\n");
    printf("  -k, --top-k <n>         Top-k sampling (default: 40)\n");
    printf("  --top-p <f>             Nucleus sampling threshold (default: 0.9)\n");
    printf("  --greedy                Use greedy decoding (no sampling)\n");
    printf("  -s, --seed <n>          Random seed\n");
    printf("  --7b                    Use 7B model config (default)\n");
    printf("  --13b                   Use 13B model config\n");
    printf("  --70b                   Use 70B model config\n");
    printf("  -h, --help              Show this help message\n");
    printf("\nExample:\n");
    printf("  %s model.gguf -p \"Once upon a time\" -t 0.7 -n 128\n", prog);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    const char* model_path = argv[1];
    const char* prompt = "Hello, world!";
    float temperature = 0.8f;
    int max_tokens = 256;
    int top_k = 40;
    float top_p = 0.9f;
    bool greedy = false;
    unsigned int seed = (unsigned int)time(NULL);
    int model_size = 7;

    for (int i = 2; i < argc; i++) {
        if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) && i + 1 < argc) {
            prompt = argv[++i];
        } else if ((strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--temperature") == 0) && i + 1 < argc) {
            temperature = (float)atof(argv[++i]);
        } else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--max-tokens") == 0) && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-k") == 0 || strcmp(argv[i], "--top-k") == 0) && i + 1 < argc) {
            top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            top_p = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--greedy") == 0) {
            greedy = true;
        } else if ((strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seed") == 0) && i + 1 < argc) {
            seed = (unsigned int)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--7b") == 0) {
            model_size = 7;
        } else if (strcmp(argv[i], "--13b") == 0) {
            model_size = 13;
        } else if (strcmp(argv[i], "--70b") == 0) {
            model_size = 70;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    srand(seed);
    tensor_manual_seed((uint64_t)seed);

    if (cml_init() != 0) {
        fprintf(stderr, "Error: Failed to initialize C-ML\n");
        return 1;
    }

    CMLLLaMAConfig config;
    switch (model_size) {
        case 13:  config = cml_llama_config_13b(); break;
        case 70:  config = cml_llama_config_70b(); break;
        default:  config = cml_llama_config_7b();  break;
    }

    printf("=== LLaMA Inference (C-ML) ===\n\n");
    cml_llama_print_config(&config);
    printf("\n");

    printf("Creating model...\n");
    CMLLLaMAModel* model = cml_llama_create(&config);
    if (!model) {
        fprintf(stderr, "Error: Failed to create model\n");
        cml_cleanup();
        return 1;
    }

    printf("Loading weights from: %s\n", model_path);
    if (cml_llama_load_gguf(model, model_path) != 0) {
        fprintf(stderr, "Error: Failed to load GGUF weights from '%s'\n", model_path);
        cml_llama_free(model);
        cml_cleanup();
        return 1;
    }
    printf("Weights loaded successfully.\n\n");

    CMLGenerationConfig gen_config = cml_generation_default_config();
    gen_config.temperature = temperature;
    gen_config.top_k = top_k;
    gen_config.top_p = top_p;
    gen_config.max_new_tokens = max_tokens;
    gen_config.do_sample = !greedy;

    printf("Generation settings:\n");
    printf("  Prompt:      \"%s\"\n", prompt);
    printf("  Temperature: %.2f\n", gen_config.temperature);
    printf("  Top-k:       %d\n", gen_config.top_k);
    printf("  Top-p:       %.2f\n", gen_config.top_p);
    printf("  Max tokens:  %d\n", gen_config.max_new_tokens);
    printf("  Sampling:    %s\n", gen_config.do_sample ? "yes" : "greedy");
    printf("  Seed:        %u\n", seed);
    printf("\n");

    printf("--- Output ---\n");
    CMLGenerationResult* result = cml_llama_generate(model, prompt, &gen_config);

    if (!result) {
        fprintf(stderr, "Error: Generation failed\n");
        cml_llama_free(model);
        cml_cleanup();
        return 1;
    }

    if (result->text) {
        printf("%s\n", result->text);
    } else {
        printf("[No text output - tokenizer may not be loaded]\n");
        printf("Token IDs:");
        for (int i = 0; i < result->num_tokens && i < 50; i++) {
            printf(" %d", result->token_ids[i]);
        }
        if (result->num_tokens > 50) {
            printf(" ... (%d more)", result->num_tokens - 50);
        }
        printf("\n");
    }

    printf("\n--- Statistics ---\n");
    printf("  Total tokens:     %d\n", result->num_tokens);
    printf("  Total time:       %.1f ms\n", result->total_time_ms);
    printf("  Tokens/second:    %.1f\n", result->tokens_per_second);

    cml_generation_result_free(result);
    cml_llama_free(model);
    cml_cleanup();

    return 0;
}
