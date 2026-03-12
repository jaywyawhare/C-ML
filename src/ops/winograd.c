/**
 * @file winograd.c
 * @brief Winograd convolution transforms for 3x3 kernel acceleration
 *
 * Implements F(2x2,3x3) and F(4x4,3x3) Winograd convolution variants.
 * F(2x2,3x3) uses 4x4 tiles and produces 2x2 output tiles (~2.25x speedup).
 * F(4x4,3x3) uses 6x6 tiles and produces 4x4 output tiles (~4x speedup).
 *
 * Reference: Lavin & Gray, "Fast Algorithms for Convolutional Neural Networks"
 */

#include "ops/winograd.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---------------------------------------------------------------------------
 * Static transform matrices for F(2x2, 3x3)
 * tile_size = 4, output_tile = 2, kernel = 3
 * -------------------------------------------------------------------------*/

/* G: weight transform matrix (4x3) */
static const float G_2x2[4][3] = {
    { 1.0f,  0.0f,  0.0f},
    { 0.5f,  0.5f,  0.5f},
    { 0.5f, -0.5f,  0.5f},
    { 0.0f,  0.0f,  1.0f}
};

/* B^T: input transform matrix (4x4) */
static const float BT_2x2[4][4] = {
    { 1,  0, -1,  0},
    { 0,  1,  1,  0},
    { 0, -1,  1,  0},
    { 0,  1,  0, -1}
};

/* A^T: output transform matrix (2x4) */
static const float AT_2x2[2][4] = {
    { 1,  1,  1,  0},
    { 0,  1, -1, -1}
};

/* ---------------------------------------------------------------------------
 * Static transform matrices for F(4x4, 3x3)
 * tile_size = 6, output_tile = 4, kernel = 3
 * -------------------------------------------------------------------------*/

/* G: weight transform matrix (6x3) */
static const float G_4x4[6][3] = {
    { 1.0f/4.0f,   0.0f,        0.0f       },
    {-1.0f/6.0f,  -1.0f/6.0f,  -1.0f/6.0f  },
    {-1.0f/6.0f,   1.0f/6.0f,  -1.0f/6.0f  },
    { 1.0f/24.0f,  1.0f/12.0f,  1.0f/6.0f  },
    { 1.0f/24.0f, -1.0f/12.0f,  1.0f/6.0f  },
    { 0.0f,        0.0f,        1.0f        }
};

/* B^T: input transform matrix (6x6) */
static const float BT_4x4[6][6] = {
    { 4,  0, -5,  0,  1,  0},
    { 0, -4, -4,  1,  1,  0},
    { 0,  4, -4, -1,  1,  0},
    { 0, -2, -1,  2,  1,  0},
    { 0,  2, -1, -2,  1,  0},
    { 0,  4,  0, -5,  0,  1}
};

/* A^T: output transform matrix (4x6) */
static const float AT_4x4[4][6] = {
    { 1,  1,  1,  1,  1,  0},
    { 0,  1, -1,  2, -2,  0},
    { 0,  1,  1,  4,  4,  0},
    { 0,  1, -1,  8, -8,  1}
};

/* ---------------------------------------------------------------------------
 * Helper: small matrix multiply  C[m x n] = A[m x k] * B[k x n]
 * All matrices stored as flat row-major arrays.
 * Designed for tile-sized matrices (up to 6x6).
 * -------------------------------------------------------------------------*/
static void mat_mul_small(const float *A, const float *B, float *C,
                          int m, int k, int n)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

/* ---------------------------------------------------------------------------
 * Helper: triple matrix multiply  D[m x n] = A[m x k1] * B[k1 x k2] * C[k2 x n]
 * Uses a caller-provided temporary buffer for the intermediate result.
 * -------------------------------------------------------------------------*/
static void mat_mul_triple(const float *A, const float *B, const float *C,
                           float *D, int m, int k1, int k2, int n, float *tmp)
{
    /* tmp[m x k2] = A * B */
    mat_mul_small(A, B, tmp, m, k1, k2);
    /* D[m x n] = tmp * C */
    mat_mul_small(tmp, C, D, m, k2, n);
}

/* ---------------------------------------------------------------------------
 * Helper: transpose a row-major matrix  B[n x m] = A^T where A is [m x n]
 * -------------------------------------------------------------------------*/
static void mat_transpose(const float *A, float *B, int m, int n)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            B[j * m + i] = A[i * n + j];
        }
    }
}

/* ---------------------------------------------------------------------------
 * Helper: extract a tile from a (possibly padded) 2D plane
 *
 * src:      pointer to the start of the channel plane [src_h x src_w]
 * tile:     output buffer [tile_size x tile_size]
 * tile_row: top-left row in the padded coordinate system
 * tile_col: top-left col in the padded coordinate system
 * src_h:    spatial height of src (un-padded)
 * src_w:    spatial width of src (un-padded)
 * pad_h:    padding applied on each side vertically
 * pad_w:    padding applied on each side horizontally
 * tile_size: side length of the tile
 * -------------------------------------------------------------------------*/
static void extract_tile(const float *src, float *tile,
                         int tile_row, int tile_col,
                         int src_h, int src_w,
                         int pad_h, int pad_w,
                         int tile_size)
{
    for (int i = 0; i < tile_size; i++) {
        int r = tile_row + i - pad_h;
        for (int j = 0; j < tile_size; j++) {
            int c = tile_col + j - pad_w;
            if (r >= 0 && r < src_h && c >= 0 && c < src_w) {
                tile[i * tile_size + j] = src[r * src_w + c];
            } else {
                tile[i * tile_size + j] = 0.0f;
            }
        }
    }
}

/* ---------------------------------------------------------------------------
 * Helper: apply input transform  V = B^T * d * B
 *
 * BT_flat:  flattened B^T matrix [ts x ts]
 * B_flat:   flattened B matrix [ts x ts]  (transpose of B^T)
 * tile:     input tile [ts x ts]
 * V:        output buffer [ts x ts]
 * ts:       tile size (4 or 6)
 * tmp:      scratch buffer [ts x ts]
 * -------------------------------------------------------------------------*/
static void transform_input_tile(const float *BT_flat, const float *B_flat,
                                 const float *tile, float *V,
                                 int ts, float *tmp)
{
    /* V = B^T * tile * B */
    mat_mul_triple(BT_flat, tile, B_flat, V, ts, ts, ts, ts, tmp);
}

/* ---------------------------------------------------------------------------
 * Helper: apply output transform  Y = A^T * M * A
 *
 * AT_flat:   flattened A^T matrix [out_tile x ts]
 * A_flat:    flattened A matrix [ts x out_tile]  (transpose of A^T)
 * M:         element-wise product accumulator [ts x ts]
 * Y:         output buffer [out_tile x out_tile]
 * out_tile:  output tile size (2 or 4)
 * ts:        tile size (4 or 6)
 * tmp:       scratch buffer [out_tile x ts]  (at least)
 * -------------------------------------------------------------------------*/
static void transform_output_tile(const float *AT_flat, const float *A_flat,
                                  const float *M, float *Y,
                                  int out_tile, int ts, float *tmp)
{
    /* Y = A^T * M * A */
    mat_mul_triple(AT_flat, M, A_flat, Y, out_tile, ts, ts, out_tile, tmp);
}

/* ---------------------------------------------------------------------------
 * Helper: write an output tile into the output plane, accumulating values
 *
 * dst:      pointer to the start of the output channel plane [dst_h x dst_w]
 * tile:     the output tile [out_tile x out_tile]
 * out_row:  top-left output row position
 * out_col:  top-left output col position
 * dst_h:    height of the output plane
 * dst_w:    width of the output plane
 * out_tile: side length of the output tile
 * -------------------------------------------------------------------------*/
static void write_output_tile(float *dst, const float *tile,
                              int out_row, int out_col,
                              int dst_h, int dst_w,
                              int out_tile)
{
    for (int i = 0; i < out_tile; i++) {
        int r = out_row + i;
        if (r >= dst_h) break;
        for (int j = 0; j < out_tile; j++) {
            int c = out_col + j;
            if (c >= dst_w) break;
            dst[r * dst_w + c] += tile[i * out_tile + j];
        }
    }
}

/* ---------------------------------------------------------------------------
 * Helper: flatten a 2D static array into a contiguous row-major buffer
 * -------------------------------------------------------------------------*/
static void flatten_G(const float *src, float *dst, int rows, int cols)
{
    memcpy(dst, src, (size_t)rows * cols * sizeof(float));
}

/* =========================================================================
 * Public API
 * ========================================================================*/

bool winograd_applicable(int kernel_h, int kernel_w,
                         int stride_h, int stride_w,
                         int dilation_h, int dilation_w)
{
    return (kernel_h == 3 && kernel_w == 3 &&
            stride_h == 1 && stride_w == 1 &&
            dilation_h == 1 && dilation_w == 1);
}

WinogradConfig winograd_select_variant(int height, int width)
{
    WinogradConfig cfg;
    cfg.kernel_size = 3;

    /* For small spatial dimensions (h <= 8 or w <= 8) use F(2x2,3x3).
     * Otherwise use F(4x4,3x3) which amortises transform overhead
     * across larger output tiles. */
    if (height <= 8 || width <= 8) {
        cfg.variant     = WINOGRAD_F2x2_3x3;
        cfg.tile_size   = 4;
        cfg.output_tile = 2;
    } else {
        cfg.variant     = WINOGRAD_F4x4_3x3;
        cfg.tile_size   = 6;
        cfg.output_tile = 4;
    }
    return cfg;
}

/* -------------------------------------------------------------------------
 * winograd_transform_weight
 *
 * For every (oc, ic) pair, compute:
 *   U[oc][ic] = G * g * G^T
 * where g is the [3x3] filter and U is [tile_size x tile_size].
 *
 * Output layout: transformed[oc * in_channels * ts*ts + ic * ts*ts + ...]
 * -----------------------------------------------------------------------*/
int winograd_transform_weight(const float *weight, int out_channels,
                              int in_channels, const WinogradConfig *config,
                              float *transformed)
{
    if (!weight || !config || !transformed) return -1;

    int ts = config->tile_size;    /* 4 or 6 */
    int ks = config->kernel_size;  /* 3      */

    /* Flatten the selected G matrix into a contiguous buffer */
    float G_flat[6 * 3];  /* max size needed (6x3 for F4x4) */
    float GT_flat[3 * 6]; /* G^T                             */

    if (config->variant == WINOGRAD_F2x2_3x3) {
        flatten_G(&G_2x2[0][0], G_flat, 4, 3);
        mat_transpose(G_flat, GT_flat, 4, 3);   /* GT is 3x4 */
    } else {
        flatten_G(&G_4x4[0][0], G_flat, 6, 3);
        mat_transpose(G_flat, GT_flat, 6, 3);   /* GT is 3x6 */
    }

    /* Temporary buffer for intermediate product in triple multiply */
    float tmp[6 * 3]; /* max intermediate: ts x ks */

    for (int oc = 0; oc < out_channels; oc++) {
        for (int ic = 0; ic < in_channels; ic++) {
            const float *g = weight + ((size_t)oc * in_channels + ic) * ks * ks;
            float *U = transformed + ((size_t)oc * in_channels + ic) * ts * ts;

            /* U = G * g * G^T */
            mat_mul_triple(G_flat, g, GT_flat, U, ts, ks, ks, ts, tmp);
        }
    }

    return 0;
}

/* -------------------------------------------------------------------------
 * winograd_conv2d
 *
 * Full Winograd convolution pipeline:
 *   1. Transform weights to Winograd domain: U = G * w * G^T
 *   2. For each tile in the input:
 *      a. Extract tile d (with implicit padding)
 *      b. Input transform: V = B^T * d * B
 *      c. Element-wise accumulate across input channels: M += V .* U
 *   3. Output transform: Y = A^T * M * A
 *   4. Write Y into the output plane
 *   5. Add bias if present
 *
 * Supports grouped convolution (groups > 1).
 * -----------------------------------------------------------------------*/
int winograd_conv2d(const float *input, const float *weight, const float *bias,
                    float *output, int batch, int in_channels, int out_channels,
                    int height, int width, int padding_h, int padding_w,
                    int groups, const WinogradConfig *config)
{
    if (!input || !weight || !output || !config) return -1;
    if (groups < 1) return -1;
    if (in_channels % groups != 0 || out_channels % groups != 0) return -1;

    int ts       = config->tile_size;    /* 4 or 6 */
    int out_tile = config->output_tile;  /* 2 or 4 */
    int ks       = config->kernel_size;  /* 3      */

    /* Output spatial dimensions (stride is always 1 for Winograd) */
    int out_h = height + 2 * padding_h - ks + 1;
    int out_w = width  + 2 * padding_w - ks + 1;

    if (out_h <= 0 || out_w <= 0) return -1;

    /* Number of tiles in each spatial direction (ceiling division) */
    int tiles_h = (out_h + out_tile - 1) / out_tile;
    int tiles_w = (out_w + out_tile - 1) / out_tile;

    /* Group partitioning */
    int in_channels_per_group  = in_channels  / groups;
    int out_channels_per_group = out_channels / groups;

    /* -------------------------------------------------------------------
     * Step 1: Transform all weight kernels to Winograd domain
     *
     * Layout: U[oc][ic_within_group][ts][ts]
     * Stored as: U[(oc * in_channels_per_group + ic_g) * ts * ts + ...]
     * -----------------------------------------------------------------*/
    size_t U_size = (size_t)out_channels * in_channels_per_group * ts * ts;
    float *U = (float *)malloc(U_size * sizeof(float));
    if (!U) return -1;

    {
        /* Flatten G and compute G^T for weight transform */
        float G_flat[6 * 3];
        float GT_flat[3 * 6];
        float tmp_w[6 * 3];

        if (config->variant == WINOGRAD_F2x2_3x3) {
            flatten_G(&G_2x2[0][0], G_flat, 4, 3);
            mat_transpose(G_flat, GT_flat, 4, 3);
        } else {
            flatten_G(&G_4x4[0][0], G_flat, 6, 3);
            mat_transpose(G_flat, GT_flat, 6, 3);
        }

        for (int g = 0; g < groups; g++) {
            for (int oc_g = 0; oc_g < out_channels_per_group; oc_g++) {
                int oc = g * out_channels_per_group + oc_g;
                for (int ic_g = 0; ic_g < in_channels_per_group; ic_g++) {
                    int ic = g * in_channels_per_group + ic_g;

                    /* weight layout: [out_channels, in_channels, 3, 3] */
                    const float *kern = weight +
                        ((size_t)oc * in_channels + ic) * ks * ks;
                    float *u_ptr = U +
                        ((size_t)oc * in_channels_per_group + ic_g) * ts * ts;

                    /* U = G * kern * G^T */
                    mat_mul_triple(G_flat, kern, GT_flat, u_ptr,
                                   ts, ks, ks, ts, tmp_w);
                }
            }
        }
    }

    /* -------------------------------------------------------------------
     * Flatten B^T, B, A^T, A for the chosen variant
     * -----------------------------------------------------------------*/
    float BT_flat[6 * 6];
    float B_flat[6 * 6];   /* B = (B^T)^T */
    float AT_flat[4 * 6];
    float A_flat[6 * 4];   /* A = (A^T)^T */

    if (config->variant == WINOGRAD_F2x2_3x3) {
        /* B^T is 4x4 */
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                BT_flat[i * 4 + j] = BT_2x2[i][j];
        mat_transpose(BT_flat, B_flat, 4, 4);

        /* A^T is 2x4 */
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 4; j++)
                AT_flat[i * 4 + j] = AT_2x2[i][j];
        mat_transpose(AT_flat, A_flat, 2, 4);   /* A is 4x2 */
    } else {
        /* B^T is 6x6 */
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 6; j++)
                BT_flat[i * 6 + j] = BT_4x4[i][j];
        mat_transpose(BT_flat, B_flat, 6, 6);

        /* A^T is 4x6 */
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 6; j++)
                AT_flat[i * 6 + j] = AT_4x4[i][j];
        mat_transpose(AT_flat, A_flat, 4, 6);   /* A is 6x4 */
    }

    /* -------------------------------------------------------------------
     * Allocate per-tile working buffers
     * -----------------------------------------------------------------*/
    size_t ts2  = (size_t)ts * ts;
    size_t ot2  = (size_t)out_tile * out_tile;

    float *tile_buf = (float *)malloc(ts2 * sizeof(float));   /* input tile d    */
    float *V_buf    = (float *)malloc(ts2 * sizeof(float));   /* B^T * d * B     */
    float *Y_buf    = (float *)malloc(ot2 * sizeof(float));   /* A^T * M * A     */
    float *tmp1     = (float *)malloc(ts2 * sizeof(float));   /* scratch for mat_mul_triple */
    float *acc_buf  = (float *)malloc(ts2 * sizeof(float));   /* accumulated M across ic    */

    if (!tile_buf || !V_buf || !Y_buf || !tmp1 || !acc_buf) {
        free(U);
        free(tile_buf);
        free(V_buf);
        free(Y_buf);
        free(tmp1);
        free(acc_buf);
        return -1;
    }

    /* -------------------------------------------------------------------
     * Zero the output buffer
     * -----------------------------------------------------------------*/
    size_t output_size = (size_t)batch * out_channels * out_h * out_w;
    memset(output, 0, output_size * sizeof(float));

    /* -------------------------------------------------------------------
     * Step 2: For each batch element, group, output channel, and tile,
     *         perform the Winograd convolution pipeline.
     * -----------------------------------------------------------------*/
    for (int b = 0; b < batch; b++) {
        for (int g = 0; g < groups; g++) {
            int ic_start = g * in_channels_per_group;
            int oc_start = g * out_channels_per_group;

            for (int oc_g = 0; oc_g < out_channels_per_group; oc_g++) {
                int oc = oc_start + oc_g;

                /* Pointer to this output channel plane */
                float *out_plane = output +
                    ((size_t)b * out_channels + oc) * out_h * out_w;

                for (int th = 0; th < tiles_h; th++) {
                    for (int tw = 0; tw < tiles_w; tw++) {
                        /* Top-left corner of the tile in padded coordinates */
                        int tile_row = th * out_tile;
                        int tile_col = tw * out_tile;

                        /* Clear the accumulator for this tile position */
                        memset(acc_buf, 0, ts2 * sizeof(float));

                        /* Accumulate element-wise products across all
                         * input channels in this group */
                        for (int ic_g = 0; ic_g < in_channels_per_group; ic_g++) {
                            int ic = ic_start + ic_g;

                            /* Pointer to this input channel plane */
                            const float *in_plane = input +
                                ((size_t)b * in_channels + ic) * height * width;

                            /* (a) Extract tile from input (handles padding) */
                            extract_tile(in_plane, tile_buf,
                                         tile_row, tile_col,
                                         height, width,
                                         padding_h, padding_w, ts);

                            /* (b) Input transform: V = B^T * tile * B */
                            transform_input_tile(BT_flat, B_flat,
                                                 tile_buf, V_buf, ts, tmp1);

                            /* Pointer to transformed weight for (oc, ic_g) */
                            const float *u_ptr = U +
                                ((size_t)oc * in_channels_per_group + ic_g) * ts * ts;

                            /* (c) Element-wise multiply and accumulate:
                             *     acc[i] += V[i] * U[i] */
                            for (int idx = 0; idx < (int)ts2; idx++) {
                                acc_buf[idx] += V_buf[idx] * u_ptr[idx];
                            }
                        }

                        /* (d) Output transform: Y = A^T * acc * A */
                        transform_output_tile(AT_flat, A_flat,
                                              acc_buf, Y_buf,
                                              out_tile, ts, tmp1);

                        /* Write output tile into the output plane */
                        int out_row = th * out_tile;
                        int out_col = tw * out_tile;
                        write_output_tile(out_plane, Y_buf,
                                          out_row, out_col,
                                          out_h, out_w, out_tile);
                    }
                }

                /* Add bias for this output channel if provided */
                if (bias) {
                    float b_val = bias[oc];
                    int plane_size = out_h * out_w;
                    for (int idx = 0; idx < plane_size; idx++) {
                        out_plane[idx] += b_val;
                    }
                }
            }
        }
    }

    /* -------------------------------------------------------------------
     * Cleanup
     * -----------------------------------------------------------------*/
    free(U);
    free(tile_buf);
    free(V_buf);
    free(Y_buf);
    free(tmp1);
    free(acc_buf);

    return 0;
}
