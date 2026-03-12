/**
 * @file winograd.h
 * @brief Winograd convolution transforms for 3x3 kernel acceleration
 *
 * F(2x2,3x3) ~2.25x fewer multiplications, F(4x4,3x3) ~4x fewer.
 */

#ifndef CML_OPS_WINOGRAD_H
#define CML_OPS_WINOGRAD_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    WINOGRAD_F2x2_3x3 = 0,  /* output 2x2, kernel 3x3 → tile 4x4 */
    WINOGRAD_F4x4_3x3 = 1,  /* output 4x4, kernel 3x3 → tile 6x6 */
} WinogradVariant;

typedef struct {
    WinogradVariant variant;
    int tile_size;    /* tile dimension (4 or 6) */
    int output_tile;  /* output tile dimension (2 or 4) */
    int kernel_size;  /* 3 */
} WinogradConfig;

/**
 * @brief Check if Winograd is applicable for given conv parameters
 */
bool winograd_applicable(int kernel_h, int kernel_w, int stride_h, int stride_w,
                         int dilation_h, int dilation_w);

/**
 * @brief Select best Winograd variant for given spatial dimensions
 */
WinogradConfig winograd_select_variant(int height, int width);

/**
 * @brief Transform weight tensor to Winograd domain
 *
 * @param weight      [out_channels, in_channels, 3, 3]
 * @param out_channels Number of output channels
 * @param in_channels  Number of input channels
 * @param config       Winograd configuration
 * @param transformed  Output buffer (caller-allocated)
 * @return 0 on success
 */
int winograd_transform_weight(const float* weight, int out_channels, int in_channels,
                              const WinogradConfig* config, float* transformed);

/**
 * @brief Perform Winograd convolution
 *
 * @param input    [batch, in_channels, height, width]
 * @param weight   [out_channels, in_channels, 3, 3]
 * @param bias     [out_channels] or NULL
 * @param output   [batch, out_channels, out_h, out_w]
 * @param batch    Batch size
 * @param in_channels  Input channels
 * @param out_channels Output channels
 * @param height   Input height
 * @param width    Input width
 * @param padding_h Padding height
 * @param padding_w Padding width
 * @param groups   Number of groups (for grouped convolution)
 * @param config   Winograd configuration
 * @return 0 on success
 */
int winograd_conv2d(const float* input, const float* weight, const float* bias,
                    float* output, int batch, int in_channels, int out_channels,
                    int height, int width, int padding_h, int padding_w,
                    int groups, const WinogradConfig* config);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPS_WINOGRAD_H */
