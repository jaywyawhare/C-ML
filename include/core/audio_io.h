#ifndef CML_CORE_AUDIO_IO_H
#define CML_CORE_AUDIO_IO_H

#ifdef __cplusplus
extern "C" {
#endif

/* Minimal RIFF/WAVE PCM reader and writer; samples are interleaved floats in [-1, 1]. */

/**
 * Load a WAV file (8/16/24/32-bit PCM or 32-bit float) into an interleaved float buffer
 * normalized to [-1,1]. Sets frames (per channel), channels and sample_rate. Caller frees;
 * NULL on error or an unsupported format.
 * @param path (const char*)
 * @param frames (int*)
 * @param channels (int*)
 * @param sample_rate (int*)
 * @return (float*)
 */
float* cml_wav_load(const char* path, int* frames, int* channels, int* sample_rate);

/**
 * Save interleaved float samples (in [-1,1]) as a 16-bit PCM WAV file. Returns 0 on success.
 * @param path (const char*)
 * @param samples (const float*)
 * @param frames (int)
 * @param channels (int)
 * @param sample_rate (int)
 * @return (int)
 */
int cml_wav_save(const char* path, const float* samples, int frames, int channels, int sample_rate);

#ifdef __cplusplus
}
#endif

#endif /* CML_CORE_AUDIO_IO_H */
