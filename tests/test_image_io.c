/* Self-contained inflate + PNG decode, validated against Python zlib reference vectors:
 * a raw DEFLATE stream round-trips to its known payload, and a known 3x2 RGB PNG decodes
 * to its known pixels. */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "alloc/cml_allocator.h"
#include "core/image_io.h"
#include "test_harness.h"

static const unsigned char RAW_DEFLATE[] = {
    115, 246, 245, 81,  200, 204, 77,  76,  79,  85,  200, 201, 79,  76,  73, 45, 178, 82, 200,
    72,  205, 201, 201, 71,  33,  203, 243, 139, 114, 82,  20,  185, 156, 7,  90, 41,  0};
static const unsigned char PAYLOAD[] = {
    67,  77,  76,  32,  105, 109, 97,  103, 101, 32,  108, 111, 97,  100, 101, 114, 58,  32,
    104, 101, 108, 108, 111, 32,  104, 101, 108, 108, 111, 32,  104, 101, 108, 108, 111, 32,
    119, 111, 114, 108, 100, 33,  10,  67,  77,  76,  32,  105, 109, 97,  103, 101, 32,  108,
    111, 97,  100, 101, 114, 58,  32,  104, 101, 108, 108, 111, 32,  104, 101, 108, 108, 111,
    32,  104, 101, 108, 108, 111, 32,  119, 111, 114, 108, 100, 33,  10,  67,  77,  76,  32,
    105, 109, 97,  103, 101, 32,  108, 111, 97,  100, 101, 114, 58,  32,  104, 101, 108, 108,
    111, 32,  104, 101, 108, 108, 111, 32,  104, 101, 108, 108, 111, 32,  119, 111, 114, 108,
    100, 33,  10,  67,  77,  76,  32,  105, 109, 97,  103, 101, 32,  108, 111, 97,  100, 101,
    114, 58,  32,  104, 101, 108, 108, 111, 32,  104, 101, 108, 108, 111, 32,  104, 101, 108,
    108, 111, 32,  119, 111, 114, 108, 100, 33,  10};
static const unsigned char PNG[] = {
    137, 80,  78, 71,  13,  10,  26,  10,  0,   0,   0,   13,  73,  72,  68,  82,  0,
    0,   0,   3,  0,   0,   0,   2,   8,   2,   0,   0,   0,   18,  22,  241, 77,  0,
    0,   0,   28, 73,  68,  65,  84,  120, 156, 99,  224, 18,  145, 211, 48,  178, 113,
    11,  136, 98, 72,  201, 171, 104, 234, 153, 182, 96,  213, 22,  0,   46,  94,  6,
    175, 87,  66, 252, 147, 0,   0,   0,   0,   73,  69,  78,  68,  174, 66,  96,  130};
static const unsigned char PNG_EXPECT[] = {10,  20,  30,  40,  50,  60,  70,  80,  90,
                                           100, 110, 120, 130, 140, 150, 160, 170, 180};

static int test_inflate_raw(void) {
    size_t n           = 0;
    unsigned char* out = cml_inflate(RAW_DEFLATE, sizeof RAW_DEFLATE, &n);
    int ok             = out && n == sizeof PAYLOAD && memcmp(out, PAYLOAD, n) == 0;
    cml_free(out);
    return ok;
}

static int test_png_decode(void) {
    FILE* f = fopen("/tmp/cml_test_img.png", "wb");
    if (!f)
        return 0;
    fwrite(PNG, 1, sizeof PNG, f);
    fclose(f);
    int w = 0, h = 0, c = 0;
    float* px = cml_image_load_png("/tmp/cml_test_img.png", &w, &h, &c);
    int ok    = px && w == 3 && h == 2 && c == 3;
    for (size_t i = 0; ok && i < sizeof PNG_EXPECT; i++)
        ok &= fabsf(px[i] - PNG_EXPECT[i] / 255.0f) < 1e-6f;
    cml_free(px);
    return ok;
}

static int test_rejects_bad(void) {
    int w = 0, h = 0, c = 0;
    return cml_image_load_png("/tmp/does_not_exist_xyz.png", &w, &h, &c) == NULL;
}

int main(void) {
    printf("=== inflate + PNG decode ===\n");
    TEST(inflate_raw);
    TEST(png_decode);
    TEST(rejects_bad);
    return TEST_SUMMARY();
}
