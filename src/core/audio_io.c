/* RIFF/WAVE PCM audio I/O. Reads 8/16/24/32-bit integer PCM and 32-bit float; writes
 * 16-bit PCM. Samples are exchanged as interleaved floats in [-1, 1]. Host is little-endian. */
#include "core/audio_io.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "alloc/cml_allocator.h"
#include "core/logging.h"

static uint32_t rd_u32(const unsigned char* p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static uint16_t rd_u16(const unsigned char* p) { return (uint16_t)(p[0] | p[1] << 8); }
static void wr_u32(FILE* f, uint32_t v) {
    unsigned char b[4] = {v & 0xff, v >> 8 & 0xff, v >> 16 & 0xff, v >> 24 & 0xff};
    fwrite(b, 1, 4, f);
}
static void wr_u16(FILE* f, uint16_t v) {
    unsigned char b[2] = {v & 0xff, v >> 8 & 0xff};
    fwrite(b, 1, 2, f);
}

float* cml_wav_load(const char* path, int* frames, int* channels, int* sample_rate) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        LOG_ERROR("wav: cannot open '%s'", path);
        return NULL;
    }
    unsigned char riff[12];
    if (fread(riff, 1, 12, f) != 12 || memcmp(riff, "RIFF", 4) != 0 ||
        memcmp(riff + 8, "WAVE", 4) != 0) {
        LOG_ERROR("wav: not a RIFF/WAVE file");
        fclose(f);
        return NULL;
    }
    uint16_t fmt = 0, nch = 0, bits = 0;
    uint32_t rate      = 0;
    long data_pos      = -1;
    uint32_t data_size = 0;
    unsigned char ch[8];
    while (fread(ch, 1, 8, f) == 8) {
        uint32_t sz = rd_u32(ch + 4);
        if (memcmp(ch, "fmt ", 4) == 0) {
            unsigned char fb[16];
            uint32_t want = sz < 16 ? sz : 16;
            if (fread(fb, 1, want, f) != want)
                break;
            fmt  = rd_u16(fb);
            nch  = rd_u16(fb + 2);
            rate = rd_u32(fb + 4);
            bits = rd_u16(fb + 14);
            if (sz > want)
                fseek(f, sz - want, SEEK_CUR);
        } else if (memcmp(ch, "data", 4) == 0) {
            data_pos  = ftell(f);
            data_size = sz;
            fseek(f, sz, SEEK_CUR);
        } else {
            fseek(f, sz, SEEK_CUR);
        }
        if (sz & 1)
            fseek(f, 1, SEEK_CUR); /* chunks are word-aligned */
    }
    if (data_pos < 0 || nch == 0 || bits == 0 || (fmt != 1 && fmt != 3)) {
        LOG_ERROR("wav: unsupported or incomplete format");
        fclose(f);
        return NULL;
    }
    int bytes_per      = bits / 8;
    size_t total       = data_size / bytes_per;
    unsigned char* raw = cml_malloc(data_size ? data_size : 1);
    if (!raw || fseek(f, data_pos, SEEK_SET) != 0 || fread(raw, 1, data_size, f) != data_size) {
        cml_free(raw);
        fclose(f);
        return NULL;
    }
    fclose(f);

    float* out = cml_malloc((total ? total : 1) * sizeof(float));
    if (!out) {
        cml_free(raw);
        return NULL;
    }
    for (size_t i = 0; i < total; i++) {
        const unsigned char* p = raw + i * bytes_per;
        float v;
        if (fmt == 3 && bits == 32) {
            memcpy(&v, p, 4);
        } else if (bits == 8) {
            v = ((int)p[0] - 128) / 128.0f; /* unsigned */
        } else if (bits == 16) {
            v = (int16_t)rd_u16(p) / 32768.0f;
        } else if (bits == 24) {
            int32_t s = (int32_t)(p[0] | p[1] << 8 | p[2] << 16);
            if (s & 0x800000)
                s |= ~0xffffff; /* sign-extend */
            v = s / 8388608.0f;
        } else if (bits == 32) {
            v = (int32_t)rd_u32(p) / 2147483648.0f;
        } else {
            cml_free(raw);
            cml_free(out);
            LOG_ERROR("wav: unsupported bit depth %d", bits);
            return NULL;
        }
        out[i] = v;
    }
    cml_free(raw);
    if (frames)
        *frames = (int)(total / nch);
    if (channels)
        *channels = nch;
    if (sample_rate)
        *sample_rate = (int)rate;
    return out;
}

int cml_wav_save(const char* path, const float* samples, int frames, int channels,
                 int sample_rate) {
    if (!path || !samples || frames < 0 || channels <= 0 || sample_rate <= 0)
        return -1;
    FILE* f = fopen(path, "wb");
    if (!f) {
        LOG_ERROR("wav: cannot open '%s' for writing", path);
        return -1;
    }
    uint32_t total     = (uint32_t)frames * (uint32_t)channels;
    uint32_t data_size = total * 2;
    uint32_t byte_rate = (uint32_t)sample_rate * (uint32_t)channels * 2;

    fwrite("RIFF", 1, 4, f);
    wr_u32(f, 36 + data_size);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    wr_u32(f, 16);
    wr_u16(f, 1); /* PCM */
    wr_u16(f, (uint16_t)channels);
    wr_u32(f, (uint32_t)sample_rate);
    wr_u32(f, byte_rate);
    wr_u16(f, (uint16_t)(channels * 2));
    wr_u16(f, 16);
    fwrite("data", 1, 4, f);
    wr_u32(f, data_size);
    for (uint32_t i = 0; i < total; i++) {
        float v = samples[i];
        if (v > 1.0f)
            v = 1.0f;
        if (v < -1.0f)
            v = -1.0f;
        wr_u16(f, (uint16_t)(int16_t)lrintf(v * 32767.0f));
    }
    int ok = !ferror(f);
    fclose(f);
    return ok ? 0 : -1;
}
