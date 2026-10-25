/* WAV audio I/O, validated against a real file produced by Python's wave module (mono
 * 16-bit PCM, samples 0.5 and -0.5) and a save->load round-trip. */
#include <math.h>
#include <stdio.h>
#include "core/audio_io.h"
#include "alloc/cml_allocator.h"
#include "test_harness.h"

/* Exact bytes written by Python wave for one mono channel of [16384, -16384] at 8 kHz. */
static const unsigned char WAV_REF[] = {
    82, 73, 70, 70, 40,  0,  0, 0, 87, 65, 86, 69, 102, 109, 116, 32, 16, 0, 0, 0, 1, 0,  1, 0,
    64, 31, 0,  0,  128, 62, 0, 0, 2,  0,  16, 0,  100, 97,  116, 97, 4,  0, 0, 0, 0, 64, 0, 192};

static int test_load_python_ref(void) {
    FILE* f = fopen("/tmp/cml_test_ref.wav", "wb");
    if (!f)
        return 0;
    fwrite(WAV_REF, 1, sizeof WAV_REF, f);
    fclose(f);
    int frames = 0, ch = 0, rate = 0;
    float* s = cml_wav_load("/tmp/cml_test_ref.wav", &frames, &ch, &rate);
    int ok   = s && frames == 2 && ch == 1 && rate == 8000 && fabsf(s[0] - 0.5f) < 1e-4f &&
             fabsf(s[1] + 0.5f) < 1e-4f;
    cml_free(s);
    return ok;
}

static int test_round_trip(void) {
    float samples[] = {0.25f, -0.75f, 1.0f, -1.0f}; /* 2 stereo frames */
    int ok          = cml_wav_save("/tmp/cml_test_rt.wav", samples, 2, 2, 44100) == 0;
    int frames = 0, ch = 0, rate = 0;
    float* s = cml_wav_load("/tmp/cml_test_rt.wav", &frames, &ch, &rate);
    ok &= s && frames == 2 && ch == 2 && rate == 44100;
    for (int i = 0; ok && i < 4; i++)
        ok &= fabsf(s[i] - samples[i]) < 1e-4f;
    cml_free(s);
    return ok;
}

static int test_rejects_bad(void) {
    int frames = 0, ch = 0, rate = 0;
    return cml_wav_load("/tmp/does_not_exist_xyz.wav", &frames, &ch, &rate) == NULL;
}

int main(void) {
    printf("=== WAV audio I/O ===\n");
    TEST(load_python_ref);
    TEST(round_trip);
    TEST(rejects_bad);
    return TEST_SUMMARY();
}
