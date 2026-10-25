#ifndef CML_CORE_IMAGE_IO_H
#define CML_CORE_IMAGE_IO_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Self-contained DEFLATE/zlib inflate and an 8-bit PNG decoder (no external libz). */

/**
 * Inflate a raw DEFLATE stream (RFC 1951) into a freshly allocated buffer; sets *out_len.
 * Returns the buffer (caller frees with cml_free) or NULL on malformed input.
 * @param data (const unsigned char*)
 * @param len (size_t)
 * @param out_len (size_t*)
 * @return (unsigned char*)
 */
unsigned char* cml_inflate(const unsigned char* data, size_t len, size_t* out_len);

/**
 * Inflate a zlib stream (RFC 1950: 2-byte header + DEFLATE + Adler-32) into a freshly
 * allocated buffer; sets *out_len. Returns the buffer (caller frees) or NULL.
 * @param data (const unsigned char*)
 * @param len (size_t)
 * @param out_len (size_t*)
 * @return (unsigned char*)
 */
unsigned char* cml_zlib_inflate(const unsigned char* data, size_t len, size_t* out_len);

/**
 * Load an 8-bit PNG (grayscale / gray+alpha / RGB / RGBA; no interlace or palette) into a
 * freshly allocated HWC float buffer normalized to [0,1]. Sets width/height/channels.
 * Returns the buffer (caller frees with cml_free) or NULL on error.
 * @param path (const char*)
 * @param width (int*)
 * @param height (int*)
 * @param channels (int*)
 * @return (float*)
 */
float* cml_image_load_png(const char* path, int* width, int* height, int* channels);

/**
 * Load an uncompressed 24/32-bit BMP into a freshly allocated HWC float buffer normalized to
 * [0,1] (3 or 4 channels). Sets width/height/channels. Caller frees; NULL on error.
 * @param path (const char*)
 * @param width (int*)
 * @param height (int*)
 * @param channels (int*)
 * @return (float*)
 */
float* cml_image_load_bmp(const char* path, int* width, int* height, int* channels);

/**
 * Save an HWC float image (in [0,1], >=3 channels) as a 24-bit BMP. Returns 0 on success.
 * @param path (const char*)
 * @param px (const float*)
 * @param width (int)
 * @param height (int)
 * @param channels (int)
 * @return (int)
 */
int cml_image_save_bmp(const char* path, const float* px, int width, int height, int channels);

/**
 * Horizontally mirror an HWC image into a freshly allocated buffer (caller frees).
 * @param px (const float*)
 * @param width (int)
 * @param height (int)
 * @param channels (int)
 * @return (float*)
 */
float* cml_image_flip_horizontal(const float* px, int width, int height, int channels);

/**
 * Center-crop an HWC image to crop_w x crop_h (must fit). Sets out_w/out_h. Caller frees.
 * @param px (const float*)
 * @param width (int)
 * @param height (int)
 * @param channels (int)
 * @param crop_w (int)
 * @param crop_h (int)
 * @param out_w (int*)
 * @param out_h (int*)
 * @return (float*)
 */
float* cml_image_center_crop(const float* px, int width, int height, int channels, int crop_w,
                             int crop_h, int* out_w, int* out_h);

/**
 * Bilinearly resize an HWC image to new_w x new_h into a freshly allocated buffer (caller frees).
 * @param px (const float*)
 * @param width (int)
 * @param height (int)
 * @param channels (int)
 * @param new_w (int)
 * @param new_h (int)
 * @return (float*)
 */
float* cml_image_resize_bilinear(const float* px, int width, int height, int channels, int new_w,
                                 int new_h);

#ifdef __cplusplus
}
#endif

#endif /* CML_CORE_IMAGE_IO_H */
