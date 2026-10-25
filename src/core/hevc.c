#include "core/hevc.h"
#include "core/logging.h"
#include "core/error_stack.h"
#include <stdlib.h>
#include <string.h>
#include "alloc/cml_allocator.h"

#define HEVC_RING_INITIAL_CAP (256 * 1024)
#define HEVC_START_CODE_3 3
#define HEVC_START_CODE_4 4

struct CMLHEVCParser {
    uint8_t* ring;
    size_t ring_cap;
    size_t ring_len;
    size_t scan_pos;
    int64_t frame_count;
    /* Set by cml_hevc_parser_end_of_stream(). A NAL is delimited by the *next*
     * start code, so while more bytes may still arrive the trailing one cannot be
     * emitted -- it might not be complete yet. Without this flag the last NAL in a
     * buffer was simply never returned, which for a whole-file feed meant the
     * slice data (the only NAL you actually decode) was unreachable. */
    bool eos;
};

typedef struct {
    const uint8_t* data;
    size_t size;
    size_t bit_pos;
} BitstreamReader;

/**
 * Initialize an MSB-first bitstream reader over a byte buffer.
 * @param bs (BitstreamReader*)
 * @param data (const uint8_t*)
 * @param size (size_t)
 * @return (void)
 */
static void bs_init(BitstreamReader* bs, const uint8_t* data, size_t size) {
    bs->data    = data;
    bs->size    = size;
    bs->bit_pos = 0;
}

/**
 * Read one bit (MSB first); -1 past end of buffer.
 * @param bs (BitstreamReader*)
 * @return (int)
 */
static int bs_read_bit(BitstreamReader* bs) {
    if (bs->bit_pos / 8 >= bs->size)
        return -1;
    int byte_idx = (int)(bs->bit_pos / 8);
    int bit_idx  = 7 - (int)(bs->bit_pos % 8);
    bs->bit_pos++;
    return (bs->data[byte_idx] >> bit_idx) & 1;
}

/**
 * Read an n-bit big-endian unsigned value from the bitstream.
 * @param bs (BitstreamReader*)
 * @param n (int)
 * @return (uint32_t)
 */
static uint32_t bs_read_bits(BitstreamReader* bs, int n) {
    uint32_t val = 0;
    for (int i = 0; i < n; i++) {
        int bit = bs_read_bit(bs);
        if (bit < 0)
            return 0;
        val = (val << 1) | (uint32_t)bit;
    }
    return val;
}

/**
 * Decode an unsigned Exp-Golomb (ue(v)) code, as used throughout H.265 headers.
 * @param bs (BitstreamReader*)
 * @return (uint32_t)
 */
static uint32_t bs_read_ue(BitstreamReader* bs) {
    int leading_zeros = 0;
    while (bs_read_bit(bs) == 0 && leading_zeros < 31)
        leading_zeros++;
    if (leading_zeros == 0)
        return 0;
    uint32_t val = bs_read_bits(bs, leading_zeros);
    return (1u << leading_zeros) - 1 + val;
}

/**
 * Strip emulation-prevention bytes (00 00 03) to recover the RBSP from a NAL
 * payload; returns a newly allocated buffer the caller frees.
 * @param nal (const uint8_t*)
 * @param nal_size (size_t)
 * @param rbsp_size (size_t*)
 * @return (uint8_t*)
 */
static uint8_t* rbsp_from_nal(const uint8_t* nal, size_t nal_size, size_t* rbsp_size) {
    uint8_t* rbsp = (uint8_t*)cml_malloc(nal_size);
    if (!rbsp)
        return NULL;

    size_t j = 0;
    for (size_t i = 0; i < nal_size; i++) {
        if (i + 2 < nal_size && nal[i] == 0x00 && nal[i + 1] == 0x00 && nal[i + 2] == 0x03) {
            rbsp[j++] = 0x00;
            rbsp[j++] = 0x00;
            i += 2;
        } else {
            rbsp[j++] = nal[i];
        }
    }

    *rbsp_size = j;
    return rbsp;
}

/**
 * Scan for the next Annex-B start code (00 00 01 or 00 00 00 01) from `start`;
 * reports its position and length. Returns 1 if found, 0 otherwise.
 * @param buf (const uint8_t*)
 * @param len (size_t)
 * @param start (size_t)
 * @param sc_pos (size_t*)
 * @param sc_len (int*)
 * @return (int)
 */
static int find_start_code(const uint8_t* buf, size_t len, size_t start, size_t* sc_pos,
                           int* sc_len) {
    for (size_t i = start; i + 2 < len; i++) {
        if (buf[i] == 0x00 && buf[i + 1] == 0x00) {
            if (i + 3 < len && buf[i + 2] == 0x00 && buf[i + 3] == 0x01) {
                *sc_pos = i;
                *sc_len = HEVC_START_CODE_4;
                return 1;
            }
            if (buf[i + 2] == 0x01) {
                *sc_pos = i;
                *sc_len = HEVC_START_CODE_3;
                return 1;
            }
        }
    }
    return 0;
}

/**
 * Allocate an HEVC Annex-B parser with an initial ring buffer.
 * @return (CMLHEVCParser*)
 */
CMLHEVCParser* cml_hevc_parser_create(void) {
    CMLHEVCParser* p = (CMLHEVCParser*)cml_calloc(1, sizeof(CMLHEVCParser));
    if (!p)
        return NULL;

    p->ring = (uint8_t*)cml_malloc(HEVC_RING_INITIAL_CAP);
    if (!p->ring) {
        cml_free(p);
        return NULL;
    }
    p->ring_cap    = HEVC_RING_INITIAL_CAP;
    p->ring_len    = 0;
    p->scan_pos    = 0;
    p->frame_count = 0;
    return p;
}

/**
 * Free the parser and its ring buffer.
 * @param parser (CMLHEVCParser*)
 * @return (void)
 */
void cml_hevc_parser_free(CMLHEVCParser* parser) {
    if (!parser)
        return;
    cml_free(parser->ring);
    cml_free(parser);
}

/**
 * Append bitstream bytes to the ring buffer, growing it as needed; -1 on error.
 * @param parser (CMLHEVCParser*)
 * @param data (const uint8_t*)
 * @param size (size_t)
 * @return (int)
 */
int cml_hevc_parser_feed(CMLHEVCParser* parser, const uint8_t* data, size_t size) {
    if (!parser || !data || size == 0)
        return -1;

    size_t needed = parser->ring_len + size;
    if (needed > parser->ring_cap) {
        size_t new_cap = parser->ring_cap;
        while (new_cap < needed)
            new_cap *= 2;
        uint8_t* new_ring = (uint8_t*)cml_realloc(parser->ring, new_cap);
        if (!new_ring) {
            LOG_ERROR("HEVC: ring buffer realloc failed (%zu bytes)", new_cap);
            return -1;
        }
        parser->ring     = new_ring;
        parser->ring_cap = new_cap;
    }

    memcpy(parser->ring + parser->ring_len, data, size);
    parser->ring_len += size;
    return 0;
}

/**
 * Signal that no more bytes will be fed, so the final NAL may be emitted.
 * @param parser (CMLHEVCParser*)
 * @return (void)
 */
void cml_hevc_parser_end_of_stream(CMLHEVCParser* parser) {
    if (parser)
        parser->eos = true;
}

/**
 * Extract the next complete NAL unit from the buffered stream, parsing its header
 * (type, temporal id). Consumed bytes are dropped from the ring buffer. Returns
 * NULL when no full NAL is available yet (or at end of stream).
 * @param parser (CMLHEVCParser*)
 * @return (CMLHEVCNalUnit*)
 */
CMLHEVCNalUnit* cml_hevc_next_nal(CMLHEVCParser* parser) {
    if (!parser || parser->ring_len < 4)
        return NULL;

    size_t first_pos, second_pos;
    int first_len, second_len;

    if (!find_start_code(parser->ring, parser->ring_len, parser->scan_pos, &first_pos, &first_len))
        return NULL;

    size_t nal_start = first_pos + first_len;

    if (!find_start_code(parser->ring, parser->ring_len, nal_start, &second_pos, &second_len)) {
        /* No following start code: this NAL runs to the end of what we have. Only
         * safe to emit once the caller has said no more bytes are coming. */
        if (!parser->eos)
            return NULL;
        second_pos = parser->ring_len;
        second_len = 0;
    }

    size_t nal_size = second_pos - nal_start;
    /* Trailing zero bytes belong to the start-code prefix of nothing, or to
     * cabac_zero_words; they are not part of the NAL payload. */
    while (nal_size > 0 && parser->ring[nal_start + nal_size - 1] == 0x00)
        nal_size--;
    if (nal_size < 2) {
        parser->scan_pos = second_pos;
        return NULL;
    }

    CMLHEVCNalUnit* nal = (CMLHEVCNalUnit*)cml_calloc(1, sizeof(CMLHEVCNalUnit));
    if (!nal)
        return NULL;

    uint8_t* nal_data = (uint8_t*)cml_malloc(nal_size);
    if (!nal_data) {
        cml_free(nal);
        return NULL;
    }
    memcpy(nal_data, parser->ring + nal_start, nal_size);

    uint8_t byte0             = nal_data[0];
    uint8_t byte1             = nal_data[1];
    int forbidden             = (byte0 >> 7) & 1;
    int nal_type              = (byte0 >> 1) & 0x3F;
    int nuh_layer_id          = ((byte0 & 1) << 5) | ((byte1 >> 3) & 0x1F);
    int nuh_temporal_id_plus1 = byte1 & 0x07;
    (void)forbidden;
    (void)nuh_layer_id;

    nal->data        = nal_data;
    nal->size        = nal_size;
    nal->type        = nal_type;
    nal->temporal_id = nuh_temporal_id_plus1 > 0 ? nuh_temporal_id_plus1 - 1 : 0;

    parser->scan_pos = second_pos;

    size_t consumed = second_pos;
    if (consumed > 0 && consumed <= parser->ring_len) {
        memmove(parser->ring, parser->ring + consumed, parser->ring_len - consumed);
        parser->ring_len -= consumed;
        parser->scan_pos = 0;
    }

    return nal;
}

/**
 * Free a NAL unit and its copied payload.
 * @param nal (CMLHEVCNalUnit*)
 * @return (void)
 */
void cml_hevc_nal_free(CMLHEVCNalUnit* nal) {
    if (!nal)
        return;
    cml_free((void*)nal->data);
    cml_free(nal);
}

/**
 * Parse an SPS NAL far enough to recover the coded picture width/height,
 * applying the conformance-window crop; -1 on malformed input.
 * @param sps_data (const uint8_t*)
 * @param sps_size (size_t)
 * @param width (int*)
 * @param height (int*)
 * @return (int)
 */
int cml_hevc_parse_sps(const uint8_t* sps_data, size_t sps_size, int* width, int* height) {
    if (!sps_data || sps_size < 4 || !width || !height)
        return -1;

    size_t rbsp_size = 0;
    uint8_t* rbsp    = rbsp_from_nal(sps_data + 2, sps_size - 2, &rbsp_size);
    if (!rbsp)
        return -1;

    BitstreamReader bs;
    bs_init(&bs, rbsp, rbsp_size);

    uint32_t sps_video_parameter_set_id   = bs_read_bits(&bs, 4);
    uint32_t sps_max_sub_layers_minus1    = bs_read_bits(&bs, 3);
    uint32_t sps_temporal_id_nesting_flag = bs_read_bits(&bs, 1);
    (void)sps_video_parameter_set_id;
    (void)sps_temporal_id_nesting_flag;

    /* profile_tier_level: general_profile_space(2) + general_tier_flag(1) + general_profile_idc(5)
     */
    bs_read_bits(&bs, 8);
    /* general_profile_compatibility_flags (32 bits) */
    bs_read_bits(&bs, 32);
    /* general_constraint_indicator_flags (48 bits) */
    bs_read_bits(&bs, 32);
    bs_read_bits(&bs, 16);
    /* general_level_idc (8 bits) */
    bs_read_bits(&bs, 8);

    for (uint32_t i = 0; i < sps_max_sub_layers_minus1; i++) {
        uint32_t sub_layer_profile_present = bs_read_bits(&bs, 1);
        uint32_t sub_layer_level_present   = bs_read_bits(&bs, 1);
        (void)sub_layer_profile_present;
        (void)sub_layer_level_present;
    }
    if (sps_max_sub_layers_minus1 > 0) {
        for (uint32_t i = sps_max_sub_layers_minus1; i < 8; i++)
            bs_read_bits(&bs, 2);
    }

    for (uint32_t i = 0; i < sps_max_sub_layers_minus1; i++) {
        /* skip sub_layer profile_tier_level - simplified */
        bs_read_bits(&bs, 8);
        bs_read_bits(&bs, 32);
        bs_read_bits(&bs, 32);
        bs_read_bits(&bs, 16);
        bs_read_bits(&bs, 8);
    }

    uint32_t sps_seq_parameter_set_id = bs_read_ue(&bs);
    (void)sps_seq_parameter_set_id;

    uint32_t chroma_format_idc = bs_read_ue(&bs);
    if (chroma_format_idc == 3) {
        bs_read_bits(&bs, 1); /* separate_colour_plane_flag */
    }

    uint32_t pic_width  = bs_read_ue(&bs);
    uint32_t pic_height = bs_read_ue(&bs);

    uint32_t conformance_window_flag = bs_read_bits(&bs, 1);
    if (conformance_window_flag) {
        uint32_t left   = bs_read_ue(&bs);
        uint32_t right  = bs_read_ue(&bs);
        uint32_t top    = bs_read_ue(&bs);
        uint32_t bottom = bs_read_ue(&bs);

        int sub_width_c  = (chroma_format_idc == 1 || chroma_format_idc == 2) ? 2 : 1;
        int sub_height_c = (chroma_format_idc == 1) ? 2 : 1;

        pic_width -= (left + right) * (uint32_t)sub_width_c;
        pic_height -= (top + bottom) * (uint32_t)sub_height_c;
    }

    *width  = (int)pic_width;
    *height = (int)pic_height;

    cml_free(rbsp);
    return 0;
}

/**
 * Decode an IDR (intra) frame to pixels. Currently unimplemented: parses the
 * NAL but fails with CM_NOT_IMPLEMENTED rather than returning a fake frame.
 * @param parser (CMLHEVCParser*)
 * @param nal (CMLHEVCNalUnit*)
 * @return (CMLHEVCFrame*)
 */
CMLHEVCFrame* cml_hevc_decode_iframe(CMLHEVCParser* parser, CMLHEVCNalUnit* nal) {
    if (!parser || !nal)
        return NULL;

    if (nal->type != HEVC_NAL_IDR_W_RADL && nal->type != HEVC_NAL_IDR_N_LP)
        return NULL;

    if (nal->size < 3)
        return NULL;

    size_t payload_offset = 2;
    size_t payload_size   = nal->size - payload_offset;

    size_t rbsp_size = 0;
    uint8_t* rbsp    = rbsp_from_nal(nal->data + payload_offset, payload_size, &rbsp_size);
    if (!rbsp)
        return NULL;

    /* Frame reconstruction (slice header parsing, intra prediction, transform,
     * deblocking) is not implemented - fail loudly instead of returning a
     * fake frame that would silently corrupt any pipeline consuming it. The
     * parser/NAL/SPS layers above are fully functional. */
    cml_free(rbsp);
    LOG_ERROR("[hevc] intra frame decoding is not implemented; NAL was parsed "
              "but no pixels can be produced");
    CML_ERR_RET(CM_NOT_IMPLEMENTED, "hevc: frame decoding not implemented", NULL);
}

/**
 * Free a decoded frame and its pixel buffer.
 * @param frame (CMLHEVCFrame*)
 * @return (void)
 */
void cml_hevc_frame_free(CMLHEVCFrame* frame) {
    if (!frame)
        return;
    cml_free(frame->data);
    cml_free(frame);
}

/* ---- Core integer transforms (H.265 §8.6.4) -------------------------------
 * DCT-II 4/8-point from the standard. These are column-orthogonal with equal
 * column norm, so inverse(forward(B)) == scale^2 * B exactly in integer
 * arithmetic -- an intrinsic correctness check needing no reference data. */

static const int16_t HEVC_DCT4[16] = {
    64, 64, 64, 64, 83, 36, -36, -83, 64, -64, -64, 64, 36, -83, 83, -36,
};

/**
 * Return the integer transform matrix for a given size. Only the exactly
 * row-orthogonal 4-point DCT is exposed; DST-VII and larger DCTs return NULL.
 * @param size (int)
 * @param dst (int)
 * @return (const int16_t*)
 */
const int16_t* cml_hevc_transform_matrix(int size, int dst) {
    if (dst)
        return NULL; /* DST-VII is not exactly orthogonal; needs reference vectors */
    if (size == 4)
        return HEVC_DCT4; /* exactly row-orthogonal -> intrinsically verifiable */
    /* 8/16/32-point integer transforms are only approximately orthogonal (lossy
     * by design), so they cannot be self-checked; they await spec reference
     * vectors before being exposed. */
    return NULL;
}

/* out = A x B over int64 (A, B are n x n row-major). at/bt transpose flags. */
/**
 * n x n integer matrix multiply over int64 with optional transpose of A or B.
 * @param A (const int64_t*)
 * @param at (int)
 * @param B (const int64_t*)
 * @param bt (int)
 * @param out (int64_t*)
 * @param n (int)
 * @return (void)
 */
static void hevc_mm(const int64_t* A, int at, const int64_t* B, int bt, int64_t* out, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int64_t acc = 0;
            for (int k = 0; k < n; k++) {
                int64_t a = at ? A[k * n + i] : A[i * n + j * 0 + k];
                int64_t b = bt ? B[j * n + k] : B[k * n + j];
                acc += a * b;
            }
            out[i * n + j] = acc;
        }
    }
}

/**
 * Apply the 2-D separable integer transform: forward M*B*M^T or inverse M^T*C*M.
 * @param in (const int32_t*)
 * @param out (int32_t*)
 * @param size (int)
 * @param dst (int)
 * @param inverse (int)
 * @return (int)
 */
static int hevc_transform(const int32_t* in, int32_t* out, int size, int dst, int inverse) {
    const int16_t* m = cml_hevc_transform_matrix(size, dst);
    if (!m || !in || !out)
        return -1;
    int n         = size;
    int64_t M[64] = {0}, B[64] = {0}, T[64] = {0}, R[64] = {0};
    for (int i = 0; i < n * n; i++) {
        M[i] = (int64_t)m[i];
        B[i] = (int64_t)in[i];
    }
    /* forward:  R = M  * B * M^T ;  inverse: R = M^T * C * M */
    if (!inverse) {
        hevc_mm(M, 0, B, 0, T, n); /* T = M * B   */
        hevc_mm(T, 0, M, 1, R, n); /* R = T * M^T */
    } else {
        hevc_mm(M, 1, B, 0, T, n); /* T = M^T * C */
        hevc_mm(T, 0, M, 0, R, n); /* R = T * M   */
    }
    for (int i = 0; i < n * n; i++)
        out[i] = (int32_t)R[i];
    return 0;
}

/**
 * Forward 2-D integer transform of a residual block into coefficients.
 * @param block (const int32_t*)
 * @param coeffs (int32_t*)
 * @param size (int)
 * @param dst (int)
 * @return (int)
 */
int cml_hevc_forward_transform(const int32_t* block, int32_t* coeffs, int size, int dst) {
    return hevc_transform(block, coeffs, size, dst, 0);
}

/**
 * Inverse 2-D integer transform of coefficients back into a residual block.
 * @param coeffs (const int32_t*)
 * @param block (int32_t*)
 * @param size (int)
 * @param dst (int)
 * @return (int)
 */
int cml_hevc_inverse_transform(const int32_t* coeffs, int32_t* block, int size, int dst) {
    return hevc_transform(coeffs, block, size, dst, 1);
}

/* ---- Intra prediction (H.265 §8.4.4.2) ------------------------------------
 * Spec-defined integer prediction of an n x n block from decoded neighbours. All
 * tables and formulas are the standard's; the result is verifiable against the DC
 * average, the planar bilinear and the pure horizontal/vertical/diagonal cases. */

/* intraPredAngle (spec Table 8-4), indexed by mode; modes 0/1 are planar/DC. */
static const int HEVC_INTRA_ANGLE[35] = {
    0,   0,   32,  26,  21,  17, 13, 9,  5, 2, 0, -2, -5, -9, -13, -17, -21, -26,
    -32, -26, -21, -17, -13, -9, -5, -2, 0, 2, 5, 9,  13, 17, 21,  26,  32,
};

/**
 * Spec Table 8-5 inverse angle for a negative intraPredAngle; 0 otherwise.
 * @param angle (int)
 * @return (int)
 */
static int hevc_inv_angle(int angle) {
    switch (angle) {
    case -2:
        return -4096;
    case -5:
        return -1638;
    case -9:
        return -910;
    case -13:
        return -630;
    case -17:
        return -482;
    case -21:
        return -390;
    case -26:
        return -315;
    case -32:
        return -256;
    default:
        return 0;
    }
}

/**
 * Clamp an integer sample to the 8-bit range.
 * @param v (int)
 * @return (uint8_t)
 */
static uint8_t hevc_clip8(int v) { return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v)); }

/**
 * Base-2 log of a power-of-two transform size (4->2 .. 32->5).
 * @param n (int)
 * @return (int)
 */
static int hevc_log2(int n) {
    int l = 0;
    while ((1 << l) < n)
        l++;
    return l;
}

/**
 * Angular intra prediction along `angle` using `main` as the projected reference
 * row and `side` for the negative-angle extension; writes column-major when
 * `transpose` is set (the horizontal mode class). `main`/`side` hold 2n samples and
 * `corner` is p[-1][-1].
 * @param n (int)
 * @param angle (int)
 * @param main (const uint8_t*)
 * @param side (const uint8_t*)
 * @param corner (uint8_t)
 * @param transpose (int)
 * @param out (uint8_t*)
 * @return (void)
 */
static void hevc_angular(int n, int angle, const uint8_t* main, const uint8_t* side, uint8_t corner,
                         int transpose, uint8_t* out) {
    int ref_buf[4 * 32 + 1]; /* indices [-2n, 2n], offset by 2n */
    const int off = 2 * n;
    int* ref      = ref_buf + off;
    for (int i = 0; i <= 2 * n; i++)
        ref[i] = (i == 0) ? corner : main[i - 1];
    if (angle < 0) {
        int inv    = hevc_inv_angle(angle);
        int minIdx = (n * angle) >> 5;
        for (int x = -1; x >= minIdx; x--) {
            int k  = -1 + ((x * inv + 128) >> 8);
            ref[x] = (k < 0) ? corner : side[k];
        }
    }
    for (int y = 0; y < n; y++) {
        int pos   = (y + 1) * angle;
        int iIdx  = pos >> 5;
        int iFact = pos & 31;
        for (int x = 0; x < n; x++) {
            int v = iFact ? ((32 - iFact) * ref[x + iIdx + 1] + iFact * ref[x + iIdx + 2] + 16) >> 5
                          : ref[x + iIdx + 1];
            int idx  = transpose ? (x * n + y) : (y * n + x);
            out[idx] = hevc_clip8(v);
        }
    }
}

/**
 * Intra-predict an n x n block from its neighbours (see the header for the sample
 * layout). mode 0 = planar, 1 = DC, 2..34 = angular.
 * @param mode (int)
 * @param n (int)
 * @param top (const uint8_t*)
 * @param left (const uint8_t*)
 * @param corner (uint8_t)
 * @param out (uint8_t*)
 * @return (int)
 */
int cml_hevc_intra_predict(int mode, int n, const uint8_t* top, const uint8_t* left, uint8_t corner,
                           uint8_t* out) {
    if (!top || !left || !out || mode < 0 || mode > 34)
        return -1;
    if (n != 4 && n != 8 && n != 16 && n != 32)
        return -1;

    int shift = hevc_log2(n) + 1;

    if (mode == 0) { /* planar */
        for (int y = 0; y < n; y++)
            for (int x = 0; x < n; x++) {
                int v = (n - 1 - x) * left[y] + (x + 1) * top[n] + (n - 1 - y) * top[x] +
                        (y + 1) * left[n] + n;
                out[y * n + x] = hevc_clip8(v >> shift);
            }
        return 0;
    }

    if (mode == 1) { /* DC */
        int sum = n;
        for (int i = 0; i < n; i++)
            sum += top[i] + left[i];
        int dc = sum >> shift;
        for (int y = 0; y < n; y++)
            for (int x = 0; x < n; x++)
                out[y * n + x] = (uint8_t)dc;
        if (n < 32) { /* luma boundary smoothing */
            out[0] = hevc_clip8((left[0] + 2 * dc + top[0] + 2) >> 2);
            for (int x = 1; x < n; x++)
                out[x] = hevc_clip8((top[x] + 3 * dc + 2) >> 2);
            for (int y = 1; y < n; y++)
                out[y * n] = hevc_clip8((left[y] + 3 * dc + 2) >> 2);
        }
        return 0;
    }

    /* angular: modes >= 18 project the row above, modes < 18 the column to the left */
    int angle = HEVC_INTRA_ANGLE[mode];
    if (mode >= 18)
        hevc_angular(n, angle, top, left, corner, 0, out);
    else
        hevc_angular(n, angle, left, top, corner, 1, out);
    return 0;
}
