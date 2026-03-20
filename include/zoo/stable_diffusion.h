#ifndef CML_ZOO_STABLE_DIFFUSION_H
#define CML_ZOO_STABLE_DIFFUSION_H

#include "nn.h"
#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int latent_channels;
    int image_channels;
    int block_channels[4];
    int num_res_blocks;
    int attention_resolutions[2];
    int num_attention_resolutions;
    DType dtype;
    DeviceType device;
} VAEConfig;

typedef struct {
    int in_channels;
    int model_channels;
    int out_channels;
    int num_res_blocks;
    int attention_resolutions[3];
    int num_attention_resolutions;
    int channel_mult[4];
    int num_channel_mult;
    int num_heads;
    int context_dim;
    int time_embed_dim;
    DType dtype;
    DeviceType device;
} UNetConfig;

typedef struct {
    int vocab_size;
    int embed_dim;
    int num_heads;
    int num_layers;
    int max_seq_len;
    int projection_dim;
    DType dtype;
    DeviceType device;
} CLIPConfig;

typedef struct {
    VAEConfig vae;
    UNetConfig unet;
    CLIPConfig clip;
    int num_timesteps;
    float beta_start;
    float beta_end;
} StableDiffusionConfig;

StableDiffusionConfig stable_diffusion_v1_config(void);

Module* cml_zoo_stable_diffusion_vae(const VAEConfig* config);
Module* cml_zoo_stable_diffusion_unet(const UNetConfig* config);
Module* cml_zoo_stable_diffusion_clip(const CLIPConfig* config);
Module* cml_zoo_stable_diffusion(const StableDiffusionConfig* config);

#ifdef __cplusplus
}
#endif

#endif /* CML_ZOO_STABLE_DIFFUSION_H */
