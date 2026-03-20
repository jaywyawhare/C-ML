#include "cml.h"
#include "zoo/zoo.h"
#include <stdio.h>

int main(void) {
    cml_init();
    cml_seed(42);
    printf("Stable Diffusion Example\n\n");

    StableDiffusionConfig sdcfg = stable_diffusion_v1_config();
    printf("Stable Diffusion v1 config:\n");
    printf("  VAE: latent_ch=%d, image_ch=%d\n",
           sdcfg.vae.latent_channels, sdcfg.vae.image_channels);
    printf("  UNet: in=%d, model_ch=%d, out=%d, heads=%d\n",
           sdcfg.unet.in_channels, sdcfg.unet.model_channels,
           sdcfg.unet.out_channels, sdcfg.unet.num_heads);
    printf("  CLIP: vocab=%d, embed=%d, layers=%d\n",
           sdcfg.clip.vocab_size, sdcfg.clip.embed_dim, sdcfg.clip.num_layers);
    printf("  timesteps=%d, beta=[%.4f, %.4f]\n",
           sdcfg.num_timesteps, sdcfg.beta_start, sdcfg.beta_end);

    /* Create individual components */
    Module* vae = cml_zoo_stable_diffusion_vae(&sdcfg.vae);
    if (vae) {
        printf("\n  VAE created.\n");
        module_free(vae);
    }

    Module* clip = cml_zoo_stable_diffusion_clip(&sdcfg.clip);
    if (clip) {
        printf("  CLIP text encoder created.\n");
        module_free(clip);
    }

    printf("\nDone.\n");
    cml_cleanup();
    return 0;
}
