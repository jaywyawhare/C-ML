#ifndef CML_CORE_HEVC_H
#define CML_CORE_HEVC_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CMLHEVCParser CMLHEVCParser;

typedef struct CMLHEVCFrame {
    uint8_t* data;
    int width;
    int height;
    int stride;
    int64_t pts;
    int nal_type;
} CMLHEVCFrame;

#define HEVC_NAL_TRAIL_N 0
#define HEVC_NAL_TRAIL_R 1
#define HEVC_NAL_IDR_W_RADL 19
#define HEVC_NAL_IDR_N_LP 20
#define HEVC_NAL_VPS 32
#define HEVC_NAL_SPS 33
#define HEVC_NAL_PPS 34
#define HEVC_NAL_AUD 35
#define HEVC_NAL_SEI_PREFIX 39

typedef struct CMLHEVCNalUnit {
    const uint8_t* data;
    size_t size;
    int type;
    int temporal_id;
} CMLHEVCNalUnit;

CMLHEVCParser* cml_hevc_parser_create(void);
void cml_hevc_parser_free(CMLHEVCParser* parser);

int cml_hevc_parser_feed(CMLHEVCParser* parser, const uint8_t* data, size_t size);

/* Signal that no further bytes will be fed.
 *
 * A NAL unit is delimited by the start code that FOLLOWS it, so until the stream
 * ends the trailing NAL may still be incomplete and cannot be emitted. Call this
 * after the final cml_hevc_parser_feed() -- otherwise cml_hevc_next_nal() stops
 * one NAL early and the slice data is never returned. */
void cml_hevc_parser_end_of_stream(CMLHEVCParser* parser);

CMLHEVCNalUnit* cml_hevc_next_nal(CMLHEVCParser* parser);
void cml_hevc_nal_free(CMLHEVCNalUnit* nal);

int cml_hevc_parse_sps(const uint8_t* sps_data, size_t sps_size, int* width, int* height);

CMLHEVCFrame* cml_hevc_decode_iframe(CMLHEVCParser* parser, CMLHEVCNalUnit* nal);
void cml_hevc_frame_free(CMLHEVCFrame* frame);

#ifdef __cplusplus
}
#endif

#endif /* CML_CORE_HEVC_H */

/* ---- Inverse/forward core transforms (H.265 §8.6.4) -----------------------
 * The spec's integer DCT-II (4/8-point) and DST-VII (4-point) matrices, plus
 * the separable 1-D transforms. These are the standard's defined integer
 * constants (facts, not external reference data) and are self-verifiable: the
 * DCT matrices are exactly orthogonal and forward→inverse round-trips the block.
 * Larger (16/32) transforms and the CABAC/prediction stages still need verified
 * tables and reference streams - see docs/REMAINING_WORK.md. */

/* Returns the size x size transform matrix (row-major), or NULL if unsupported.
 * dst=0 => DCT-II (size 4 or 8); dst=1 => DST-VII (size 4 only). */
const int16_t* cml_hevc_transform_matrix(int size, int dst);

/* Separable 2-D forward/inverse transform of a size x size block (row-major
 * int32). `dst` selects DST-VII (size 4) vs DCT-II. Returns 0 on success. */
int cml_hevc_forward_transform(const int32_t* block, int32_t* coeffs, int size, int dst);
int cml_hevc_inverse_transform(const int32_t* coeffs, int32_t* block, int size, int dst);
