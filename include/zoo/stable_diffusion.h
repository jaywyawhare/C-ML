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

/* --- DDIM scheduler + denoising loop ------------------------------------ */

typedef struct CMLSDScheduler {
    int num_train_timesteps;
    float* alphas_cumprod; /* [num_train_timesteps] */
    int* timesteps;        /* inference subsequence, descending */
    int num_inference_steps;
} CMLSDScheduler;

/* Scaled-linear beta schedule (SD v1 convention). */
CMLSDScheduler* cml_sd_scheduler_create(int num_timesteps, float beta_start, float beta_end);
void cml_sd_scheduler_free(CMLSDScheduler* sched);

/* Choose the evenly spaced inference timestep subsequence (descending). */
int cml_sd_scheduler_set_steps(CMLSDScheduler* sched, int num_inference_steps);

/* One deterministic DDIM (eta=0) update, in place on the latent buffer:
 * x0 = (x_t - sqrt(1-a_t)*eps) / sqrt(a_t);
 * x_prev = sqrt(a_prev)*x0 + sqrt(1-a_prev)*eps. */
int cml_sd_scheduler_step(CMLSDScheduler* sched, const float* eps, float* latent, size_t numel,
                          int step_index);

/* Denoising loop: iterates the scheduler over `num_inference_steps`, calling
 * `unet` on the current latent as the noise predictor each step, then applies
 * the DDIM update in place. Returns the final latent (caller frees).
 * Note: the Sequential UNet takes only the latent — timestep/context
 * conditioning needs a modular UNet forward and is not wired here. */
Tensor* cml_sd_generate(Module* unet, CMLSDScheduler* sched, Tensor* initial_latent,
                        int num_inference_steps);

#ifdef __cplusplus
}
#endif

#endif /* CML_ZOO_STABLE_DIFFUSION_H */
