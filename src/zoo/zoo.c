#include "zoo/zoo.h"
#include "zoo/resnet.h"
#include "zoo/gpt2.h"
#include "zoo/bert.h"
#include "zoo/vit.h"
#include "zoo/clip.h"
#include "zoo/t5.h"
#include "zoo/unet.h"
#include "zoo/unet3d.h"
#include "zoo/rnnt.h"
#include "nn.h"
#include "nn/layers.h"
#include "nn/model_io.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>
#include <dlfcn.h>

static const char* g_weights_base_url = "https://weights.cml-lib.org/v1";
static char g_weights_dir[512] = "";

static const char* zoo_model_names[] = {
    [CML_ZOO_MLP_MNIST]   = "mlp_mnist",
    [CML_ZOO_MLP_CIFAR10] = "mlp_cifar10",
    [CML_ZOO_RESNET18]    = "resnet18",
    [CML_ZOO_RESNET34]    = "resnet34",
    [CML_ZOO_RESNET50]    = "resnet50",
    [CML_ZOO_VGG11]       = "vgg11",
    [CML_ZOO_VGG16]       = "vgg16",
    [CML_ZOO_GPT2_SMALL]  = "gpt2_small",
    [CML_ZOO_GPT2_MEDIUM] = "gpt2_medium",
    [CML_ZOO_GPT2_LARGE]  = "gpt2_large",
    [CML_ZOO_GPT2_XL]     = "gpt2_xl",
    [CML_ZOO_BERT_TINY]   = "bert_tiny",
    [CML_ZOO_BERT_MINI]   = "bert_mini",
    [CML_ZOO_BERT_SMALL]  = "bert_small",
    [CML_ZOO_BERT_BASE]   = "bert_base",
    [CML_ZOO_BERT_LARGE]  = "bert_large",
    [CML_ZOO_VIT_TINY]    = "vit_tiny",
    [CML_ZOO_VIT_SMALL]   = "vit_small",
    [CML_ZOO_VIT_BASE]    = "vit_base",
    [CML_ZOO_VIT_LARGE]   = "vit_large",
    [CML_ZOO_CLIP_VIT_B32]= "clip_vit_b32",
    [CML_ZOO_CLIP_VIT_B16]= "clip_vit_b16",
    [CML_ZOO_CLIP_VIT_L14]= "clip_vit_l14",
    [CML_ZOO_T5_SMALL]    = "t5_small",
    [CML_ZOO_T5_BASE]     = "t5_base",
    [CML_ZOO_T5_LARGE]    = "t5_large",
    [CML_ZOO_UNET_DEFAULT]    = "unet_default",
    [CML_ZOO_CONVNEXT_TINY]   = "convnext_tiny",
    [CML_ZOO_CONVNEXT_SMALL]  = "convnext_small",
    [CML_ZOO_CONVNEXT_BASE]   = "convnext_base",
    [CML_ZOO_CONVNEXT_LARGE]  = "convnext_large",
    [CML_ZOO_INCEPTION_V3]    = "inception_v3",
    [CML_ZOO_RETINANET]       = "retinanet",
    [CML_ZOO_MASK_RCNN]       = "mask_rcnn",
    [CML_ZOO_UNET3D]         = "unet3d",
    [CML_ZOO_RNNT]           = "rnnt",
};

CMLZooConfig cml_zoo_default_config(void) {
    CMLZooConfig cfg = {
        .pretrained = false,
        .num_classes = 0,
        .dtype = DTYPE_FLOAT32,
        .device = DEVICE_CPU,
        .weights_dir = NULL
    };
    return cfg;
}

const char* cml_zoo_model_name(CMLZooModel model) {
    if (model >= CML_ZOO_MODEL_COUNT)
        return "unknown";
    return zoo_model_names[model];
}

const char* cml_zoo_get_weights_dir(void) {
    if (g_weights_dir[0] == '\0') {
        const char* home = getenv("HOME");
        if (!home) home = "/tmp";
        snprintf(g_weights_dir, sizeof(g_weights_dir), "%s/.cml/weights", home);
    }
    return g_weights_dir;
}

void cml_zoo_set_weights_url(const char* base_url) {
    if (base_url)
        g_weights_base_url = base_url;
}

static int ensure_dir_exists(const char* path) {
    struct stat st;
    if (stat(path, &st) == 0)
        return 0;

    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "mkdir -p '%s'", path);
    return system(cmd);
}

const char* cml_zoo_download_weights(CMLZooModel model, const char* weights_dir) {
    if (model >= CML_ZOO_MODEL_COUNT)
        return NULL;

    const char* dir = weights_dir ? weights_dir : cml_zoo_get_weights_dir();
    ensure_dir_exists(dir);

    static char path[1024];
    const char* name = cml_zoo_model_name(model);
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);

    /* Check if already downloaded */
    struct stat st;
    if (stat(path, &st) == 0 && st.st_size > 0) {
        LOG_INFO("Weights already cached: %s", path);
        return path;
    }

    /* Try curl command */
    char url[1024];
    snprintf(url, sizeof(url), "%s/%s.bin", g_weights_base_url, name);

    char cmd[8192];
    snprintf(cmd, sizeof(cmd), "curl -fsSL -o '%s' '%s' 2>/dev/null || wget -q -O '%s' '%s' 2>/dev/null",
             path, url, path, url);

    LOG_INFO("Downloading weights: %s -> %s", url, path);
    int result = system(cmd);

    if (result != 0) {
        LOG_WARNING("Failed to download weights for %s (this is expected for custom models)", name);
        return NULL;
    }

    LOG_INFO("Weights downloaded: %s", path);
    return path;
}

int cml_zoo_load_weights(Module* module, const char* weights_path) {
    if (!module || !weights_path) {
        LOG_ERROR("Invalid arguments to cml_zoo_load_weights");
        return -1;
    }

    return model_load(module, weights_path);
}

Module* cml_zoo_mlp_mnist(const CMLZooConfig* config) {
    CMLZooConfig cfg = config ? *config : cml_zoo_default_config();
    int num_classes = cfg.num_classes > 0 ? cfg.num_classes : 10;

    Sequential* model = nn_sequential();
    sequential_add(model, (Module*)nn_linear(784, 256, cfg.dtype, cfg.device, true));
    sequential_add(model, (Module*)nn_relu(false));
    sequential_add(model, (Module*)nn_linear(256, 128, cfg.dtype, cfg.device, true));
    sequential_add(model, (Module*)nn_relu(false));
    sequential_add(model, (Module*)nn_linear(128, num_classes, cfg.dtype, cfg.device, true));

    if (cfg.pretrained) {
        const char* path = cml_zoo_download_weights(CML_ZOO_MLP_MNIST, cfg.weights_dir);
        if (path)
            model_load((Module*)model, path);
    }

    LOG_INFO("Created MLP-MNIST: 784->256->128->%d", num_classes);
    return (Module*)model;
}

Module* cml_zoo_mlp_cifar10(const CMLZooConfig* config) {
    CMLZooConfig cfg = config ? *config : cml_zoo_default_config();
    int num_classes = cfg.num_classes > 0 ? cfg.num_classes : 10;

    Sequential* model = nn_sequential();
    sequential_add(model, (Module*)nn_linear(3072, 512, cfg.dtype, cfg.device, true));
    sequential_add(model, (Module*)nn_relu(false));
    sequential_add(model, (Module*)nn_linear(512, 256, cfg.dtype, cfg.device, true));
    sequential_add(model, (Module*)nn_relu(false));
    sequential_add(model, (Module*)nn_linear(256, num_classes, cfg.dtype, cfg.device, true));

    if (cfg.pretrained) {
        const char* path = cml_zoo_download_weights(CML_ZOO_MLP_CIFAR10, cfg.weights_dir);
        if (path)
            model_load((Module*)model, path);
    }

    LOG_INFO("Created MLP-CIFAR10: 3072->512->256->%d", num_classes);
    return (Module*)model;
}

static Module* create_resnet_block(int in_channels, int out_channels, int stride,
                                    DType dtype, DeviceType device) {
    /* ResNet basic block uses 2 conv layers with skip connection */
    /* Since we don't have a custom ResidualBlock module, we use sequential + manual skip */
    Sequential* block = nn_sequential();

    /* Conv 3x3 -> BN -> ReLU -> Conv 3x3 -> BN */
    sequential_add(block, (Module*)nn_conv2d(in_channels, out_channels, 3, stride, 1, 1, false, dtype, device));
    sequential_add(block, (Module*)nn_batchnorm2d(out_channels, 1e-5f, 0.1f, true, true, dtype, device));
    sequential_add(block, (Module*)nn_relu(false));
    sequential_add(block, (Module*)nn_conv2d(out_channels, out_channels, 3, 1, 1, 1, false, dtype, device));
    sequential_add(block, (Module*)nn_batchnorm2d(out_channels, 1e-5f, 0.1f, true, true, dtype, device));

    return (Module*)block;
}

static Module* create_resnet(const int* layers, int num_layers, const CMLZooConfig* cfg_ptr) __attribute__((unused));
static Module* create_resnet(const int* layers, int num_layers, const CMLZooConfig* cfg_ptr) {
    CMLZooConfig cfg = cfg_ptr ? *cfg_ptr : cml_zoo_default_config();
    int num_classes = cfg.num_classes > 0 ? cfg.num_classes : 1000;

    Sequential* model = nn_sequential();

    /* Initial conv: 7x7, stride 2 */
    sequential_add(model, (Module*)nn_conv2d(3, 64, 7, 2, 3, 1, false, cfg.dtype, cfg.device));
    sequential_add(model, (Module*)nn_batchnorm2d(64, 1e-5f, 0.1f, true, true, cfg.dtype, cfg.device));
    sequential_add(model, (Module*)nn_relu(false));
    sequential_add(model, (Module*)nn_maxpool2d(3, 2, 1, 1, false));

    /* Residual layers */
    int channels[] = {64, 128, 256, 512};
    for (int layer = 0; layer < num_layers && layer < 4; layer++) {
        int in_ch = (layer == 0) ? 64 : channels[layer - 1];
        int out_ch = channels[layer];

        for (int block = 0; block < layers[layer]; block++) {
            int stride = (block == 0 && layer > 0) ? 2 : 1;
            int block_in = (block == 0) ? in_ch : out_ch;
            sequential_add(model, create_resnet_block(block_in, out_ch, stride, cfg.dtype, cfg.device));
            sequential_add(model, (Module*)nn_relu(false));
        }
    }

    /* Average pool + FC */
    sequential_add(model, (Module*)nn_avgpool2d(7, 1, 0, false, true));
    sequential_add(model, (Module*)nn_linear(512, num_classes, cfg.dtype, cfg.device, true));

    return (Module*)model;
}

Module* cml_zoo_resnet18(const CMLZooConfig* config) {
    CMLZooConfig cfg = config ? *config : cml_zoo_default_config();
    int num_classes = cfg.num_classes > 0 ? cfg.num_classes : 1000;

    Module* model = cml_zoo_resnet18_create(num_classes, cfg.dtype, cfg.device);
    if (!model)
        return NULL;

    if (cfg.pretrained) {
        const char* path = cml_zoo_download_weights(CML_ZOO_RESNET18, cfg.weights_dir);
        if (path) model_load(model, path);
    }

    return model;
}

Module* cml_zoo_resnet34(const CMLZooConfig* config) {
    CMLZooConfig cfg = config ? *config : cml_zoo_default_config();
    int num_classes = cfg.num_classes > 0 ? cfg.num_classes : 1000;

    Module* model = cml_zoo_resnet34_create(num_classes, cfg.dtype, cfg.device);
    if (!model)
        return NULL;

    if (cfg.pretrained) {
        const char* path = cml_zoo_download_weights(CML_ZOO_RESNET34, cfg.weights_dir);
        if (path) model_load(model, path);
    }

    return model;
}

Module* cml_zoo_resnet50(const CMLZooConfig* config) {
    CMLZooConfig cfg = config ? *config : cml_zoo_default_config();
    int num_classes = cfg.num_classes > 0 ? cfg.num_classes : 1000;

    Module* model = cml_zoo_resnet50_create(num_classes, cfg.dtype, cfg.device);

    if (cfg.pretrained) {
        const char* path = cml_zoo_download_weights(CML_ZOO_RESNET50, cfg.weights_dir);
        if (path) model_load(model, path);
    }

    return model;
}

static Module* create_vgg(const int* layer_cfg, int num_blocks, const CMLZooConfig* cfg_ptr) {
    CMLZooConfig cfg = cfg_ptr ? *cfg_ptr : cml_zoo_default_config();
    int num_classes = cfg.num_classes > 0 ? cfg.num_classes : 1000;

    Sequential* model = nn_sequential();

    int in_channels = 3;
    for (int i = 0; i < num_blocks; i++) {
        if (layer_cfg[i] == 0) {
            /* MaxPool marker */
            sequential_add(model, (Module*)nn_maxpool2d(2, 2, 0, 1, false));
        } else {
            sequential_add(model, (Module*)nn_conv2d(in_channels, layer_cfg[i], 3, 1, 1, 1, true, cfg.dtype, cfg.device));
            sequential_add(model, (Module*)nn_relu(false));
            in_channels = layer_cfg[i];
        }
    }

    /* Classifier */
    sequential_add(model, (Module*)nn_linear(512 * 7 * 7, 4096, cfg.dtype, cfg.device, true));
    sequential_add(model, (Module*)nn_relu(false));
    sequential_add(model, (Module*)nn_linear(4096, 4096, cfg.dtype, cfg.device, true));
    sequential_add(model, (Module*)nn_relu(false));
    sequential_add(model, (Module*)nn_linear(4096, num_classes, cfg.dtype, cfg.device, true));

    return (Module*)model;
}

Module* cml_zoo_vgg11(const CMLZooConfig* config) {
    /* VGG-11 config: 64, M, 128, M, 256, 256, M, 512, 512, M, 512, 512, M */
    int layer_cfg[] = {64, 0, 128, 0, 256, 256, 0, 512, 512, 0, 512, 512, 0};
    Module* model = create_vgg(layer_cfg, 13, config);

    if (config && config->pretrained) {
        const char* path = cml_zoo_download_weights(CML_ZOO_VGG11, config->weights_dir);
        if (path) model_load(model, path);
    }

    LOG_INFO("Created VGG-11");
    return model;
}

Module* cml_zoo_vgg16(const CMLZooConfig* config) {
    /* VGG-16 config */
    int layer_cfg[] = {64, 64, 0, 128, 128, 0, 256, 256, 256, 0, 512, 512, 512, 0, 512, 512, 512, 0};
    Module* model = create_vgg(layer_cfg, 18, config);

    if (config && config->pretrained) {
        const char* path = cml_zoo_download_weights(CML_ZOO_VGG16, config->weights_dir);
        if (path) model_load(model, path);
    }

    LOG_INFO("Created VGG-16");
    return model;
}

static Module* create_gpt2_from_config(GPT2Config gpt2_cfg, const CMLZooConfig* config,
                                        CMLZooModel zoo_model) {
    CMLZooConfig cfg = config ? *config : cml_zoo_default_config();

    Module* model = cml_zoo_gpt2_create(&gpt2_cfg, cfg.dtype, cfg.device);
    if (!model)
        return NULL;

    if (cfg.pretrained) {
        const char* path = cml_zoo_download_weights(zoo_model, cfg.weights_dir);
        if (path) model_load(model, path);
    }

    return model;
}

Module* cml_zoo_gpt2_small(const CMLZooConfig* config) {
    return create_gpt2_from_config(cml_zoo_gpt2_config_small(), config, CML_ZOO_GPT2_SMALL);
}

Module* cml_zoo_gpt2_medium(const CMLZooConfig* config) {
    return create_gpt2_from_config(cml_zoo_gpt2_config_medium(), config, CML_ZOO_GPT2_MEDIUM);
}

Module* cml_zoo_gpt2_large(const CMLZooConfig* config) {
    return create_gpt2_from_config(cml_zoo_gpt2_config_large(), config, CML_ZOO_GPT2_LARGE);
}

Module* cml_zoo_gpt2_xl(const CMLZooConfig* config) {
    return create_gpt2_from_config(cml_zoo_gpt2_config_xl(), config, CML_ZOO_GPT2_XL);
}

static Module* create_bert_from_config(BERTConfig bert_cfg, const CMLZooConfig* config,
                                        CMLZooModel zoo_model) {
    CMLZooConfig cfg = config ? *config : cml_zoo_default_config();

    Module* model = cml_zoo_bert_create(&bert_cfg, cfg.dtype, cfg.device);
    if (!model)
        return NULL;

    if (cfg.pretrained) {
        const char* path = cml_zoo_download_weights(zoo_model, cfg.weights_dir);
        if (path) model_load(model, path);
    }

    return model;
}

Module* cml_zoo_bert_tiny(const CMLZooConfig* config) {
    return create_bert_from_config(cml_zoo_bert_config_tiny(), config, CML_ZOO_BERT_TINY);
}

Module* cml_zoo_bert_mini(const CMLZooConfig* config) {
    return create_bert_from_config(cml_zoo_bert_config_mini(), config, CML_ZOO_BERT_MINI);
}

Module* cml_zoo_bert_small(const CMLZooConfig* config) {
    return create_bert_from_config(cml_zoo_bert_config_small(), config, CML_ZOO_BERT_SMALL);
}

Module* cml_zoo_bert_base(const CMLZooConfig* config) {
    return create_bert_from_config(cml_zoo_bert_config_base(), config, CML_ZOO_BERT_BASE);
}

Module* cml_zoo_bert_large(const CMLZooConfig* config) {
    return create_bert_from_config(cml_zoo_bert_config_large(), config, CML_ZOO_BERT_LARGE);
}

static Module* create_vit_from_config(ViTConfig vit_cfg, const CMLZooConfig* config,
                                       CMLZooModel zoo_model) {
    CMLZooConfig cfg = config ? *config : cml_zoo_default_config();
    if (cfg.num_classes > 0)
        vit_cfg.num_classes = cfg.num_classes;

    Module* model = cml_zoo_vit_create(&vit_cfg, cfg.dtype, cfg.device);
    if (!model)
        return NULL;

    if (cfg.pretrained) {
        const char* path = cml_zoo_download_weights(zoo_model, cfg.weights_dir);
        if (path) model_load(model, path);
    }

    return model;
}

Module* cml_zoo_vit_tiny(const CMLZooConfig* config) {
    return create_vit_from_config(cml_zoo_vit_config_tiny(), config, CML_ZOO_VIT_TINY);
}

Module* cml_zoo_vit_small(const CMLZooConfig* config) {
    return create_vit_from_config(cml_zoo_vit_config_small(), config, CML_ZOO_VIT_SMALL);
}

Module* cml_zoo_vit_base(const CMLZooConfig* config) {
    return create_vit_from_config(cml_zoo_vit_config_base(), config, CML_ZOO_VIT_BASE);
}

Module* cml_zoo_vit_large(const CMLZooConfig* config) {
    return create_vit_from_config(cml_zoo_vit_config_large(), config, CML_ZOO_VIT_LARGE);
}

static Module* create_clip_from_config(CMLCLIPConfig clip_cfg, const CMLZooConfig* config,
                                        CMLZooModel zoo_model) {
    CMLZooConfig cfg = config ? *config : cml_zoo_default_config();

    Module* model = cml_zoo_clip_create(&clip_cfg, cfg.dtype, cfg.device);
    if (!model)
        return NULL;

    if (cfg.pretrained) {
        const char* path = cml_zoo_download_weights(zoo_model, cfg.weights_dir);
        if (path) model_load(model, path);
    }

    return model;
}

Module* cml_zoo_clip_vit_b32(const CMLZooConfig* config) {
    return create_clip_from_config(cml_zoo_clip_config_vit_b32(), config, CML_ZOO_CLIP_VIT_B32);
}

Module* cml_zoo_clip_vit_b16(const CMLZooConfig* config) {
    return create_clip_from_config(cml_zoo_clip_config_vit_b16(), config, CML_ZOO_CLIP_VIT_B16);
}

Module* cml_zoo_clip_vit_l14(const CMLZooConfig* config) {
    return create_clip_from_config(cml_zoo_clip_config_vit_l14(), config, CML_ZOO_CLIP_VIT_L14);
}

static Module* create_t5_from_config(T5Config t5_cfg, const CMLZooConfig* config,
                                      CMLZooModel zoo_model) {
    CMLZooConfig cfg = config ? *config : cml_zoo_default_config();

    Module* model = cml_zoo_t5_create(&t5_cfg, cfg.dtype, cfg.device);
    if (!model)
        return NULL;

    if (cfg.pretrained) {
        const char* path = cml_zoo_download_weights(zoo_model, cfg.weights_dir);
        if (path) model_load(model, path);
    }

    return model;
}

Module* cml_zoo_t5_small(const CMLZooConfig* config) {
    return create_t5_from_config(cml_zoo_t5_config_small(), config, CML_ZOO_T5_SMALL);
}

Module* cml_zoo_t5_base(const CMLZooConfig* config) {
    return create_t5_from_config(cml_zoo_t5_config_base(), config, CML_ZOO_T5_BASE);
}

Module* cml_zoo_t5_large(const CMLZooConfig* config) {
    return create_t5_from_config(cml_zoo_t5_config_large(), config, CML_ZOO_T5_LARGE);
}

Module* cml_zoo_unet_default(const CMLZooConfig* config) {
    CMLZooConfig cfg = config ? *config : cml_zoo_default_config();
    CMLUNetConfig unet_cfg = cml_zoo_unet_config_default();
    if (cfg.num_classes > 0)
        unet_cfg.num_classes = cfg.num_classes;

    Module* model = cml_zoo_unet_create(&unet_cfg, cfg.dtype, cfg.device);
    if (!model)
        return NULL;

    if (cfg.pretrained) {
        const char* path = cml_zoo_download_weights(CML_ZOO_UNET_DEFAULT, cfg.weights_dir);
        if (path) model_load(model, path);
    }

    return model;
}

static Module* create_convnext(ConvNeXtConfig cnx_cfg, const CMLZooConfig* config,
                                CMLZooModel zoo_model) {
    CMLZooConfig cfg = config ? *config : cml_zoo_default_config();
    int num_classes = cfg.num_classes > 0 ? cfg.num_classes : 1000;

    Module* model = cml_zoo_convnext_create(&cnx_cfg, num_classes, cfg.dtype, cfg.device);
    if (!model)
        return NULL;

    if (cfg.pretrained) {
        const char* path = cml_zoo_download_weights(zoo_model, cfg.weights_dir);
        if (path) model_load(model, path);
    }

    return model;
}

Module* cml_zoo_convnext_tiny(const CMLZooConfig* config) {
    return create_convnext(cml_zoo_convnext_config_tiny(), config, CML_ZOO_CONVNEXT_TINY);
}

Module* cml_zoo_convnext_small(const CMLZooConfig* config) {
    return create_convnext(cml_zoo_convnext_config_small(), config, CML_ZOO_CONVNEXT_SMALL);
}

Module* cml_zoo_convnext_base(const CMLZooConfig* config) {
    return create_convnext(cml_zoo_convnext_config_base(), config, CML_ZOO_CONVNEXT_BASE);
}

Module* cml_zoo_convnext_large(const CMLZooConfig* config) {
    return create_convnext(cml_zoo_convnext_config_large(), config, CML_ZOO_CONVNEXT_LARGE);
}

Module* cml_zoo_inception_v3(const CMLZooConfig* config) {
    CMLZooConfig cfg = config ? *config : cml_zoo_default_config();
    int num_classes = cfg.num_classes > 0 ? cfg.num_classes : 1000;

    Module* model = cml_zoo_inception_v3_create(num_classes, cfg.dtype, cfg.device);
    if (!model)
        return NULL;

    if (cfg.pretrained) {
        const char* path = cml_zoo_download_weights(CML_ZOO_INCEPTION_V3, cfg.weights_dir);
        if (path) model_load(model, path);
    }

    return model;
}

Module* cml_zoo_retinanet(const CMLZooConfig* config) {
    CMLZooConfig cfg = config ? *config : cml_zoo_default_config();
    RetinaNetConfig rt_cfg = cml_zoo_retinanet_default_config();
    if (cfg.num_classes > 0)
        rt_cfg.num_classes = cfg.num_classes;

    Module* model = cml_zoo_retinanet_create(&rt_cfg, cfg.dtype, cfg.device);
    if (!model)
        return NULL;

    if (cfg.pretrained) {
        const char* path = cml_zoo_download_weights(CML_ZOO_RETINANET, cfg.weights_dir);
        if (path) model_load(model, path);
    }

    return model;
}

Module* cml_zoo_mask_rcnn(const CMLZooConfig* config) {
    CMLZooConfig cfg = config ? *config : cml_zoo_default_config();
    MaskRCNNConfig mr_cfg = cml_zoo_mask_rcnn_default_config();
    if (cfg.num_classes > 0)
        mr_cfg.num_classes = cfg.num_classes;

    Module* model = cml_zoo_mask_rcnn_create(&mr_cfg, cfg.dtype, cfg.device);
    if (!model)
        return NULL;

    if (cfg.pretrained) {
        const char* path = cml_zoo_download_weights(CML_ZOO_MASK_RCNN, cfg.weights_dir);
        if (path) model_load(model, path);
    }

    return model;
}

Module* cml_zoo_unet3d(const CMLZooConfig* config) {
    CMLZooConfig cfg = config ? *config : cml_zoo_default_config();
    CMLUNet3DConfig u3_cfg = cml_zoo_unet3d_config_default();
    if (cfg.num_classes > 0)
        u3_cfg.num_classes = cfg.num_classes;

    Module* model = cml_zoo_unet3d_create(&u3_cfg, cfg.dtype, cfg.device);
    if (!model) return NULL;

    if (cfg.pretrained) {
        const char* path = cml_zoo_download_weights(CML_ZOO_UNET3D, cfg.weights_dir);
        if (path) model_load(model, path);
    }

    return model;
}

Module* cml_zoo_rnnt(const CMLZooConfig* config) {
    CMLZooConfig cfg = config ? *config : cml_zoo_default_config();
    CMLRNNTConfig rt_cfg = cml_zoo_rnnt_config_default();

    Module* model = cml_zoo_rnnt_create(&rt_cfg, cfg.dtype, cfg.device);
    if (!model) return NULL;

    if (cfg.pretrained) {
        const char* path = cml_zoo_download_weights(CML_ZOO_RNNT, cfg.weights_dir);
        if (path) model_load(model, path);
    }

    return model;
}

Module* cml_zoo_create(CMLZooModel model, const CMLZooConfig* config) {
    switch (model) {
    case CML_ZOO_MLP_MNIST:   return cml_zoo_mlp_mnist(config);
    case CML_ZOO_MLP_CIFAR10: return cml_zoo_mlp_cifar10(config);
    case CML_ZOO_RESNET18:    return cml_zoo_resnet18(config);
    case CML_ZOO_RESNET34:    return cml_zoo_resnet34(config);
    case CML_ZOO_RESNET50:    return cml_zoo_resnet50(config);
    case CML_ZOO_VGG11:       return cml_zoo_vgg11(config);
    case CML_ZOO_VGG16:       return cml_zoo_vgg16(config);
    case CML_ZOO_GPT2_SMALL:  return cml_zoo_gpt2_small(config);
    case CML_ZOO_GPT2_MEDIUM: return cml_zoo_gpt2_medium(config);
    case CML_ZOO_GPT2_LARGE:  return cml_zoo_gpt2_large(config);
    case CML_ZOO_GPT2_XL:     return cml_zoo_gpt2_xl(config);
    case CML_ZOO_BERT_TINY:   return cml_zoo_bert_tiny(config);
    case CML_ZOO_BERT_MINI:   return cml_zoo_bert_mini(config);
    case CML_ZOO_BERT_SMALL:  return cml_zoo_bert_small(config);
    case CML_ZOO_BERT_BASE:   return cml_zoo_bert_base(config);
    case CML_ZOO_BERT_LARGE:  return cml_zoo_bert_large(config);
    case CML_ZOO_VIT_TINY:    return cml_zoo_vit_tiny(config);
    case CML_ZOO_VIT_SMALL:   return cml_zoo_vit_small(config);
    case CML_ZOO_VIT_BASE:    return cml_zoo_vit_base(config);
    case CML_ZOO_VIT_LARGE:   return cml_zoo_vit_large(config);
    case CML_ZOO_CLIP_VIT_B32:return cml_zoo_clip_vit_b32(config);
    case CML_ZOO_CLIP_VIT_B16:return cml_zoo_clip_vit_b16(config);
    case CML_ZOO_CLIP_VIT_L14:return cml_zoo_clip_vit_l14(config);
    case CML_ZOO_T5_SMALL:    return cml_zoo_t5_small(config);
    case CML_ZOO_T5_BASE:     return cml_zoo_t5_base(config);
    case CML_ZOO_T5_LARGE:    return cml_zoo_t5_large(config);
    case CML_ZOO_UNET_DEFAULT:   return cml_zoo_unet_default(config);
    case CML_ZOO_CONVNEXT_TINY:  return cml_zoo_convnext_tiny(config);
    case CML_ZOO_CONVNEXT_SMALL: return cml_zoo_convnext_small(config);
    case CML_ZOO_CONVNEXT_BASE:  return cml_zoo_convnext_base(config);
    case CML_ZOO_CONVNEXT_LARGE: return cml_zoo_convnext_large(config);
    case CML_ZOO_INCEPTION_V3:   return cml_zoo_inception_v3(config);
    case CML_ZOO_RETINANET:      return cml_zoo_retinanet(config);
    case CML_ZOO_MASK_RCNN:      return cml_zoo_mask_rcnn(config);
    case CML_ZOO_UNET3D:         return cml_zoo_unet3d(config);
    case CML_ZOO_RNNT:           return cml_zoo_rnnt(config);
    default:
        LOG_ERROR("Unknown zoo model: %d", model);
        return NULL;
    }
}
