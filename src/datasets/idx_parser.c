#include "datasets/datasets.h"
#include "core/logging.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static uint32_t read_u32_be(FILE* f) {
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) |
           ((uint32_t)b[2] << 8) | (uint32_t)b[3];
}

float* cml_idx_load_images(const char* path, int* n, int* rows, int* cols) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        LOG_ERROR("[idx] Cannot open: %s", path);
        return NULL;
    }

    uint32_t magic = read_u32_be(f);
    if (magic != 2051) {
        LOG_ERROR("[idx] Invalid image magic: %u (expected 2051)", magic);
        fclose(f);
        return NULL;
    }

    *n    = (int)read_u32_be(f);
    *rows = (int)read_u32_be(f);
    *cols = (int)read_u32_be(f);

    int px = (*rows) * (*cols);
    float* data = malloc(sizeof(float) * (*n) * px);
    if (!data) { fclose(f); return NULL; }

    uint8_t* buf = malloc(px);
    if (!buf) { free(data); fclose(f); return NULL; }

    for (int i = 0; i < *n; i++) {
        if ((int)fread(buf, 1, px, f) != px) {
            LOG_ERROR("[idx] Truncated at image %d", i);
            free(data); free(buf); fclose(f);
            return NULL;
        }
        for (int j = 0; j < px; j++)
            data[i * px + j] = buf[j] / 255.0f;
    }

    free(buf);
    fclose(f);
    LOG_INFO("[idx] Loaded %d images (%dx%d) from %s", *n, *rows, *cols, path);
    return data;
}

float* cml_idx_load_labels(const char* path, int* n) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        LOG_ERROR("[idx] Cannot open: %s", path);
        return NULL;
    }

    uint32_t magic = read_u32_be(f);
    if (magic != 2049) {
        LOG_ERROR("[idx] Invalid label magic: %u (expected 2049)", magic);
        fclose(f);
        return NULL;
    }

    *n = (int)read_u32_be(f);

    float* data = malloc(sizeof(float) * (*n));
    if (!data) { fclose(f); return NULL; }

    for (int i = 0; i < *n; i++) {
        uint8_t label;
        if (fread(&label, 1, 1, f) != 1) {
            LOG_ERROR("[idx] Truncated at label %d", i);
            free(data); fclose(f);
            return NULL;
        }
        data[i] = (float)label;
    }

    fclose(f);
    LOG_INFO("[idx] Loaded %d labels from %s", *n, path);
    return data;
}
