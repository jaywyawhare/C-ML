#include "zoo/stable_diffusion.h"
#include "zoo/zoo.h"
#include "nn.h"
#include "nn/layers.h"
#include "nn/model_io.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>

StableDiffusionConfig stable_diffusion_v1_config(void) {
    StableDiffusionConfig cfg = {
        .vae = {
            .latent_channels = 4,
            .image_channels = 3,
            .block_channels = {128, 256, 512, 512},
            .num_res_blocks = 2,
            .attention_resolutions = {32, 16},
            .num_attention_resolutions = 2,
            .dtype = DTYPE_FLOAT32,
            .device = DEVICE_CPU
        },
        .unet = {
            .in_channels = 4,
            .model_channels = 320,
            .out_channels = 4,
            .num_res_blocks = 2,
            .attention_resolutions = {4, 2, 1},
            .num_attention_resolutions = 3,
            .channel_mult = {1, 2, 4, 4},
            .num_channel_mult = 4,
            .num_heads = 8,
            .context_dim = 768,
            .time_embed_dim = 1280,
            .dtype = DTYPE_FLOAT32,
            .device = DEVICE_CPU
        },
        .clip = {
            .vocab_size = 49408,
            .embed_dim = 768,
            .num_heads = 12,
            .num_layers = 12,
            .max_seq_len = 77,
            .projection_dim = 768,
            .dtype = DTYPE_FLOAT32,
            .device = DEVICE_CPU
        },
        .num_timesteps = 1000,
        .beta_start = 0.00085f,
        .beta_end = 0.012f
    };
    return cfg;
}

static void add_resblock(Sequential* seq, int in_ch, int out_ch, int groups,
                          DType dtype, DeviceType device) {
    sequential_add(seq, (Module*)nn_groupnorm(groups, in_ch, 1e-6f, true, dtype, device));
    sequential_add(seq, (Module*)nn_silu());
    sequential_add(seq, (Module*)nn_conv2d(in_ch, out_ch, 3, 1, 1, 1, true, dtype, device));
    sequential_add(seq, (Module*)nn_groupnorm(groups, out_ch, 1e-6f, true, dtype, device));
    sequential_add(seq, (Module*)nn_silu());
    sequential_add(seq, (Module*)nn_conv2d(out_ch, out_ch, 3, 1, 1, 1, true, dtype, device));

    if (in_ch != out_ch)
        sequential_add(seq, (Module*)nn_conv2d(in_ch, out_ch, 1, 1, 0, 1, true, dtype, device));
}

static void add_spatial_attention(Sequential* seq, int channels, int num_heads,
                                   DType dtype, DeviceType device) {
    sequential_add(seq, (Module*)nn_groupnorm(32, channels, 1e-6f, true, dtype, device));
    sequential_add(seq, (Module*)nn_multihead_attention(channels, num_heads, 0.0f, dtype, device));
}

static void add_cross_attention(Sequential* seq, int channels, int context_dim,
                                 int num_heads, DType dtype, DeviceType device) {
    sequential_add(seq, (Module*)nn_layernorm(channels, 1e-5f, true, dtype, device));
    sequential_add(seq, (Module*)nn_multihead_attention(channels, num_heads, 0.0f, dtype, device));

    sequential_add(seq, (Module*)nn_layernorm(channels, 1e-5f, true, dtype, device));
    sequential_add(seq, (Module*)nn_linear(channels, channels, dtype, device, true));
    sequential_add(seq, (Module*)nn_linear(context_dim, channels, dtype, device, false));
    sequential_add(seq, (Module*)nn_multihead_attention(channels, num_heads, 0.0f, dtype, device));

    sequential_add(seq, (Module*)nn_layernorm(channels, 1e-5f, true, dtype, device));
    sequential_add(seq, (Module*)nn_linear(channels, channels * 4, dtype, device, true));
    sequential_add(seq, (Module*)nn_gelu(false));
    sequential_add(seq, (Module*)nn_linear(channels * 4, channels, dtype, device, true));
}

Module* cml_zoo_stable_diffusion_vae(const VAEConfig* config) {
    VAEConfig cfg = config ? *config : stable_diffusion_v1_config().vae;
    DType dt = cfg.dtype;
    DeviceType dev = cfg.device;
    int ng = 32;

    Sequential* model = nn_sequential();

    /* Encoder */
    sequential_add(model, (Module*)nn_conv2d(cfg.image_channels, cfg.block_channels[0], 3, 1, 1, 1, true, dt, dev));

    int ch = cfg.block_channels[0];
    for (int level = 0; level < 4; level++) {
        int out_ch = cfg.block_channels[level];
        for (int b = 0; b < cfg.num_res_blocks; b++) {
            add_resblock(model, ch, out_ch, ng, dt, dev);
            ch = out_ch;
        }
        if (level < 3) {
            sequential_add(model, (Module*)nn_conv2d(ch, ch, 3, 2, 1, 1, true, dt, dev));
        }
    }

    add_resblock(model, ch, ch, ng, dt, dev);
    add_spatial_attention(model, ch, 1, dt, dev);
    add_resblock(model, ch, ch, ng, dt, dev);

    sequential_add(model, (Module*)nn_groupnorm(ng, ch, 1e-6f, true, dt, dev));
    sequential_add(model, (Module*)nn_silu());
    sequential_add(model, (Module*)nn_conv2d(ch, cfg.latent_channels * 2, 3, 1, 1, 1, true, dt, dev));
    sequential_add(model, (Module*)nn_conv2d(cfg.latent_channels * 2, cfg.latent_channels * 2, 1, 1, 0, 1, true, dt, dev));

    /* Decoder */
    sequential_add(model, (Module*)nn_conv2d(cfg.latent_channels, cfg.latent_channels, 1, 1, 0, 1, true, dt, dev));
    sequential_add(model, (Module*)nn_conv2d(cfg.latent_channels, ch, 3, 1, 1, 1, true, dt, dev));

    add_resblock(model, ch, ch, ng, dt, dev);
    add_spatial_attention(model, ch, 1, dt, dev);
    add_resblock(model, ch, ch, ng, dt, dev);

    for (int level = 3; level >= 0; level--) {
        int out_ch = cfg.block_channels[level];
        for (int b = 0; b < cfg.num_res_blocks + 1; b++) {
            add_resblock(model, ch, out_ch, ng, dt, dev);
            ch = out_ch;
        }
        if (level > 0) {
            sequential_add(model, (Module*)nn_upsample(2.0f, NULL, 0, UPSAMPLE_NEAREST, false));
            sequential_add(model, (Module*)nn_conv2d(ch, ch, 3, 1, 1, 1, true, dt, dev));
        }
    }

    sequential_add(model, (Module*)nn_groupnorm(ng, ch, 1e-6f, true, dt, dev));
    sequential_add(model, (Module*)nn_silu());
    sequential_add(model, (Module*)nn_conv2d(ch, cfg.image_channels, 3, 1, 1, 1, true, dt, dev));

    LOG_INFO("Created Stable Diffusion VAE: %d latent channels", cfg.latent_channels);
    return (Module*)model;
}

Module* cml_zoo_stable_diffusion_unet(const UNetConfig* config) {
    UNetConfig cfg = config ? *config : stable_diffusion_v1_config().unet;
    DType dt = cfg.dtype;
    DeviceType dev = cfg.device;
    int ng = 32;

    Sequential* model = nn_sequential();

    sequential_add(model, (Module*)nn_conv2d(cfg.in_channels, cfg.model_channels, 3, 1, 1, 1, true, dt, dev));

    /* Time embedding: sinusoidal -> Linear -> SiLU -> Linear */
    sequential_add(model, (Module*)nn_linear(cfg.model_channels, cfg.time_embed_dim, dt, dev, true));
    sequential_add(model, (Module*)nn_silu());
    sequential_add(model, (Module*)nn_linear(cfg.time_embed_dim, cfg.time_embed_dim, dt, dev, true));

    /* Down blocks */
    int ch = cfg.model_channels;
    for (int level = 0; level < cfg.num_channel_mult; level++) {
        int mult = cfg.channel_mult[level];
        int out_ch = cfg.model_channels * mult;

        for (int b = 0; b < cfg.num_res_blocks; b++) {
            add_resblock(model, ch, out_ch, ng, dt, dev);
            ch = out_ch;

            bool has_attn = false;
            for (int a = 0; a < cfg.num_attention_resolutions; a++) {
                if (cfg.attention_resolutions[a] == level + 1)
                    has_attn = true;
            }
            if (has_attn)
                add_cross_attention(model, ch, cfg.context_dim, cfg.num_heads, dt, dev);
        }

        if (level < cfg.num_channel_mult - 1) {
            sequential_add(model, (Module*)nn_conv2d(ch, ch, 3, 2, 1, 1, true, dt, dev));
        }
    }

    /* Mid block */
    add_resblock(model, ch, ch, ng, dt, dev);
    add_cross_attention(model, ch, cfg.context_dim, cfg.num_heads, dt, dev);
    add_resblock(model, ch, ch, ng, dt, dev);

    /* Up blocks */
    for (int level = cfg.num_channel_mult - 1; level >= 0; level--) {
        int mult = cfg.channel_mult[level];
        int out_ch = cfg.model_channels * mult;

        for (int b = 0; b < cfg.num_res_blocks + 1; b++) {
            add_resblock(model, ch, out_ch, ng, dt, dev);
            ch = out_ch;

            bool has_attn = false;
            for (int a = 0; a < cfg.num_attention_resolutions; a++) {
                if (cfg.attention_resolutions[a] == level + 1)
                    has_attn = true;
            }
            if (has_attn)
                add_cross_attention(model, ch, cfg.context_dim, cfg.num_heads, dt, dev);
        }

        if (level > 0) {
            sequential_add(model, (Module*)nn_upsample(2.0f, NULL, 0, UPSAMPLE_NEAREST, false));
            sequential_add(model, (Module*)nn_conv2d(ch, ch, 3, 1, 1, 1, true, dt, dev));
        }
    }

    /* Output */
    sequential_add(model, (Module*)nn_groupnorm(ng, ch, 1e-6f, true, dt, dev));
    sequential_add(model, (Module*)nn_silu());
    sequential_add(model, (Module*)nn_conv2d(ch, cfg.out_channels, 3, 1, 1, 1, true, dt, dev));

    LOG_INFO("Created Stable Diffusion UNet: %d model channels, context_dim=%d",
             cfg.model_channels, cfg.context_dim);
    return (Module*)model;
}

Module* cml_zoo_stable_diffusion_clip(const CLIPConfig* config) {
    CLIPConfig cfg = config ? *config : stable_diffusion_v1_config().clip;
    DType dt = cfg.dtype;
    DeviceType dev = cfg.device;
    int d = cfg.embed_dim;
    int ff = d * 4;

    Sequential* model = nn_sequential();

    sequential_add(model, (Module*)nn_embedding(cfg.vocab_size, d, -1, dt, dev));
    sequential_add(model, (Module*)nn_embedding(cfg.max_seq_len, d, -1, dt, dev));

    for (int i = 0; i < cfg.num_layers; i++) {
        sequential_add(model, (Module*)nn_layernorm(d, 1e-5f, true, dt, dev));
        sequential_add(model, (Module*)nn_multihead_attention(d, cfg.num_heads, 0.0f, dt, dev));
        sequential_add(model, (Module*)nn_layernorm(d, 1e-5f, true, dt, dev));
        sequential_add(model, (Module*)nn_linear(d, ff, dt, dev, true));
        sequential_add(model, (Module*)nn_gelu(false));
        sequential_add(model, (Module*)nn_linear(ff, d, dt, dev, true));
    }

    sequential_add(model, (Module*)nn_layernorm(d, 1e-5f, true, dt, dev));

    if (cfg.projection_dim != d)
        sequential_add(model, (Module*)nn_linear(d, cfg.projection_dim, dt, dev, false));

    LOG_INFO("Created CLIP text encoder: %d layers, dim=%d, vocab=%d",
             cfg.num_layers, d, cfg.vocab_size);
    return (Module*)model;
}

Module* cml_zoo_stable_diffusion(const StableDiffusionConfig* config) {
    StableDiffusionConfig cfg = config ? *config : stable_diffusion_v1_config();

    Sequential* model = nn_sequential();

    Module* clip = cml_zoo_stable_diffusion_clip(&cfg.clip);
    sequential_add(model, clip);

    Module* unet = cml_zoo_stable_diffusion_unet(&cfg.unet);
    sequential_add(model, unet);

    Module* vae = cml_zoo_stable_diffusion_vae(&cfg.vae);
    sequential_add(model, vae);

    LOG_INFO("Created Stable Diffusion v1: CLIP + UNet + VAE, %d timesteps", cfg.num_timesteps);
    return (Module*)model;
}
