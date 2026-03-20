#include "cml.h"
#include "zoo/zoo.h"
#include <stdio.h>

int main(void) {
    cml_init();
    cml_seed(42);
    printf("Whisper Example\n\n");

    WhisperConfig wcfg = whisper_tiny_config();
    printf("Whisper-Tiny config:\n");
    printf("  audio: mels=%d, ctx=%d, state=%d, heads=%d, layers=%d\n",
           wcfg.n_mels, wcfg.n_audio_ctx, wcfg.n_audio_state,
           wcfg.n_audio_head, wcfg.n_audio_layer);
    printf("  text:  ctx=%d, state=%d, heads=%d, layers=%d, vocab=%d\n",
           wcfg.n_text_ctx, wcfg.n_text_state, wcfg.n_text_head,
           wcfg.n_text_layer, wcfg.n_vocab);

    Module* model = cml_zoo_whisper(&wcfg);
    if (!model) { printf("Failed to create Whisper\n"); return 1; }

    printf("\nWhisper-Tiny created successfully.\n");

    module_free(model);
    printf("\nDone.\n");
    cml_cleanup();
    return 0;
}
