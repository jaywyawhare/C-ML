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

#ifdef __cplusplus
}
#endif

#endif /* CML_CORE_IMAGE_IO_H */
