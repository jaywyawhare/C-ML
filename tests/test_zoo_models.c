
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tensor/tensor.h"
#include "nn.h"
#include "zoo/zoo.h"

static int tests_passed = 0;
static int tests_total  = 0;

static const TensorConfig cpu_f32 = {
    .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
    .has_dtype = true, .has_device = true
};

#define TEST(name)                                     \
    do {                                               \
        tests_total++;                                 \
        printf("  TEST: %s ... ", #name);              \
        fflush(stdout);                                \
        if (test_##name()) {                           \
            tests_passed++;                            \
            printf("PASSED\n");                        \
        } else {                                       \
            printf("FAILED\n");                        \
        }                                              \
    } while (0)

static int has_nonzero(Tensor* t) {
    if (!t) return 0;
    if (tensor_ensure_executed(t) != 0) return 0;
    float* d = t->data;
    if (!d) return 0;
    for (size_t i = 0; i < t->numel; i++)
        if (d[i] != 0.0f) return 1;
    return 0;
}

static int test_resnet18(void) {
    Module* m = cml_zoo_resnet18_create(1000, DTYPE_FLOAT32, DEVICE_CPU);
    if (!m) return 0;
    if (module_get_total_parameters(m) <= 0) { module_free(m); return 0; }
    
    int shape[] = {1, 3, 64, 64};
    Tensor* x   = tensor_rand(shape, 4, &cpu_f32);
    Tensor* out = module_forward(m, x);
    int ok = (out != NULL);
    tensor_free(x);
    if (out) tensor_free(out);
    module_free(m);
    return ok;
}

static int test_resnet34(void) {
    Module* m = cml_zoo_resnet34_create(10, DTYPE_FLOAT32, DEVICE_CPU);
    if (!m) return 0;
    module_free(m);
    return 1;
}

static int test_resnet50(void) {
    Module* m = cml_zoo_resnet50_create(10, DTYPE_FLOAT32, DEVICE_CPU);
    if (!m) return 0;
    if (module_get_total_parameters(m) <= 0) { module_free(m); return 0; }
    module_free(m);
    return 1;
}

static int test_gpt2_small(void) {
    GPT2Config cfg = cml_zoo_gpt2_config_small();
    Module* m = cml_zoo_gpt2_create(&cfg, DTYPE_FLOAT32, DEVICE_CPU);
    if (!m) return 0;
    module_free(m);
    return 1;
}

static int test_gpt2_configs(void) {
    
    GPT2Config sm = cml_zoo_gpt2_config_small();
    GPT2Config md = cml_zoo_gpt2_config_medium();
    GPT2Config lg = cml_zoo_gpt2_config_large();
    GPT2Config xl = cml_zoo_gpt2_config_xl();
    
    if (sm.n_layer >= md.n_layer) return 0;
    if (md.n_layer >= lg.n_layer) return 0;
    if (lg.n_layer >= xl.n_layer) return 0;
    
    if (sm.n_embd <= 0 || md.n_embd <= 0) return 0;
    return 1;
}

static int test_bert_tiny(void) {
    BERTConfig cfg = cml_zoo_bert_config_tiny();
    Module* m = cml_zoo_bert_create(&cfg, DTYPE_FLOAT32, DEVICE_CPU);
    if (!m) return 0;
    module_free(m);
    return 1;
}

static int test_bert_configs(void) {
    BERTConfig tiny  = cml_zoo_bert_config_tiny();
    BERTConfig mini  = cml_zoo_bert_config_mini();
    BERTConfig small = cml_zoo_bert_config_small();
    BERTConfig base  = cml_zoo_bert_config_base();
    
    if (tiny.hidden_size > mini.hidden_size) return 0;
    if (mini.hidden_size > small.hidden_size) return 0;
    if (small.hidden_size > base.hidden_size) return 0;
    if (base.vocab_size <= 0) return 0;
    return 1;
}

static int test_vit_tiny(void) {
    ViTConfig cfg = cml_zoo_vit_config_tiny();
    Module* m = cml_zoo_vit_create(&cfg, DTYPE_FLOAT32, DEVICE_CPU);
    if (!m) return 0;
    if (module_get_total_parameters(m) <= 0) { module_free(m); return 0; }
    module_free(m);
    return 1;
}

static int test_vit_configs(void) {
    ViTConfig tiny  = cml_zoo_vit_config_tiny();
    ViTConfig small = cml_zoo_vit_config_small();
    ViTConfig base  = cml_zoo_vit_config_base();
    
    if (tiny.hidden_size > small.hidden_size) return 0;
    if (small.hidden_size > base.hidden_size) return 0;
    if (base.n_head <= 0) return 0;
    return 1;
}

static int test_convnext_tiny(void) {
    ConvNeXtConfig cfg = cml_zoo_convnext_config_tiny();
    Module* m = cml_zoo_convnext_create(&cfg, 1000, DTYPE_FLOAT32, DEVICE_CPU);
    if (!m) return 0;
    if (module_get_total_parameters(m) <= 0) { module_free(m); return 0; }
    module_free(m);
    return 1;
}

static int test_convnext_configs_increasing(void) {
    ConvNeXtConfig tiny  = cml_zoo_convnext_config_tiny();
    ConvNeXtConfig small = cml_zoo_convnext_config_small();
    ConvNeXtConfig base  = cml_zoo_convnext_config_base();
    
    if (tiny.dims[0] > small.dims[0]) return 0;
    if (small.dims[0] > base.dims[0]) return 0;
    return 1;
}

static int test_efficientnet_b0(void) {
    EfficientNetConfig cfg = efficientnet_b0_config(1000);
    Module* m = cml_zoo_efficientnet_b0(&cfg);
    if (!m) return 0;
    if (module_get_total_parameters(m) <= 0) { module_free(m); return 0; }
    module_free(m);
    return 1;
}

static int test_efficientnet_scaling(void) {
    
    EfficientNetConfig b0_cfg = efficientnet_b0_config(1000);
    EfficientNetConfig b7_cfg = efficientnet_b7_config(1000);
    
    if (b7_cfg.width_mult <= b0_cfg.width_mult) return 0;
    if (b7_cfg.depth_mult <= b0_cfg.depth_mult) return 0;
    return 1;
}

static int test_whisper_tiny(void) {
    WhisperConfig cfg = whisper_tiny_config();
    Module* m = cml_zoo_whisper(&cfg);
    if (!m) return 0;
    if (module_get_total_parameters(m) <= 0) { module_free(m); return 0; }
    module_free(m);
    return 1;
}

static int test_whisper_configs(void) {
    WhisperConfig tiny   = whisper_tiny_config();
    WhisperConfig medium = whisper_medium_config();
    WhisperConfig large  = whisper_large_config();
    
    if (tiny.n_audio_state >= medium.n_audio_state) return 0;
    if (medium.n_audio_state >= large.n_audio_state) return 0;
    return 1;
}

static int test_yolov8n(void) {
    YOLOv8Config cfg = yolov8n_config(80);
    Module* m = cml_zoo_yolov8n(&cfg);
    if (!m) return 0;
    if (module_get_total_parameters(m) <= 0) { module_free(m); return 0; }
    module_free(m);
    return 1;
}

static int test_yolov8_configs(void) {
    YOLOv8Config n = yolov8n_config(80);
    YOLOv8Config s = yolov8s_config(80);
    YOLOv8Config x = yolov8x_config(80);
    
    if (n.input_size <= 0) return 0;
    if (s.input_size != n.input_size) return 0;
    
    if (n.num_classes != 80 || s.num_classes != 80 || x.num_classes != 80) return 0;
    return 1;
}

static int test_clip_vit_b32(void) {
    CMLCLIPConfig cfg = cml_zoo_clip_config_vit_b32();
    Module* m = cml_zoo_clip_create(&cfg, DTYPE_FLOAT32, DEVICE_CPU);
    if (!m) return 0;
    if (module_get_total_parameters(m) <= 0) { module_free(m); return 0; }
    module_free(m);
    return 1;
}

static int test_clip_configs(void) {
    CMLCLIPConfig b32 = cml_zoo_clip_config_vit_b32();
    CMLCLIPConfig b16 = cml_zoo_clip_config_vit_b16();
    CMLCLIPConfig l14 = cml_zoo_clip_config_vit_l14();
    
    if (b32.embed_dim <= 0) return 0;
    
    if (l14.vision_dim <= b32.vision_dim) return 0;
    (void)b16;
    return 1;
}

static int test_t5_small(void) {
    T5Config cfg = cml_zoo_t5_config_small();
    Module* m = cml_zoo_t5_create(&cfg, DTYPE_FLOAT32, DEVICE_CPU);
    if (!m) return 0;
    if (module_get_total_parameters(m) <= 0) { module_free(m); return 0; }
    module_free(m);
    return 1;
}

static int test_t5_configs(void) {
    T5Config sm = cml_zoo_t5_config_small();
    T5Config ba = cml_zoo_t5_config_base();
    T5Config lg = cml_zoo_t5_config_large();
    if (sm.d_model >= ba.d_model) return 0;
    if (ba.d_model >= lg.d_model) return 0;
    return 1;
}

static int test_resnet_param_count_reasonable(void) {
    
    Module* r18 = cml_zoo_resnet18_create(1000, DTYPE_FLOAT32, DEVICE_CPU);
    Module* r50 = cml_zoo_resnet50_create(1000, DTYPE_FLOAT32, DEVICE_CPU);
    if (!r18 || !r50) { module_free(r18); module_free(r50); return 0; }
    int p18 = module_get_total_parameters(r18);
    int p50 = module_get_total_parameters(r50);
    module_free(r18);
    module_free(r50);
    
    if (p50 <= p18) return 0;
    
    if (p18 < 1000000) return 0;
    return 1;
}

static int test_gpt2_param_count_reasonable(void) {
    GPT2Config sm = cml_zoo_gpt2_config_small();
    GPT2Config xl = cml_zoo_gpt2_config_xl();
    Module* msm = cml_zoo_gpt2_create(&sm, DTYPE_FLOAT32, DEVICE_CPU);
    Module* mxl = cml_zoo_gpt2_create(&xl, DTYPE_FLOAT32, DEVICE_CPU);
    if (!msm || !mxl) { module_free(msm); module_free(mxl); return 0; }
    int psm = module_get_total_parameters(msm);
    int pxl = module_get_total_parameters(mxl);
    module_free(msm); module_free(mxl);
    if (psm <= 0 || pxl <= 0) return 0;
    if (pxl <= psm) return 0; 
    return 1;
}

static int test_module_double_free_safe(void) {
    
    module_free(NULL);
    return 1;
}

static int test_module_training_flag(void) {
    Module* m = cml_zoo_resnet18_create(10, DTYPE_FLOAT32, DEVICE_CPU);
    if (!m) return 0;
    module_set_training(m, true);
    if (!module_is_training(m)) { module_free(m); return 0; }
    module_set_training(m, false);
    if (module_is_training(m)) { module_free(m); return 0; }
    module_free(m);
    return 1;
}

int main(void) {
    printf("Zoo Model Instantiation Tests\n");

    
    TEST(resnet18);
    TEST(resnet34);
    TEST(resnet50);
    

    
    TEST(gpt2_small);
    TEST(gpt2_configs);
    

    
    TEST(bert_tiny);
    TEST(bert_configs);

    
    TEST(vit_tiny);
    TEST(vit_configs);

    
    TEST(convnext_tiny);
    TEST(convnext_configs_increasing);

    
    TEST(efficientnet_b0);
    TEST(efficientnet_scaling);

    
    TEST(whisper_tiny);
    TEST(whisper_configs);

    
    TEST(yolov8n);
    TEST(yolov8_configs);

    
    TEST(clip_vit_b32);
    TEST(clip_configs);

    
    
    TEST(t5_configs);

    
    TEST(module_double_free_safe);
    TEST(module_training_flag);

    printf("\nResults: %d/%d passed\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
