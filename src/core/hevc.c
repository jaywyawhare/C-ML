#include "core/hevc.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>

#define HEVC_RING_INITIAL_CAP (256 * 1024)
#define HEVC_START_CODE_3     3
#define HEVC_START_CODE_4     4

struct CMLHEVCParser {
    uint8_t* ring;
    size_t ring_cap;
    size_t ring_len;
    size_t scan_pos;
    int64_t frame_count;
};

typedef struct {
    const uint8_t* data;
    size_t size;
    size_t bit_pos;
} BitstreamReader;

static void bs_init(BitstreamReader* bs, const uint8_t* data, size_t size) {
    bs->data = data;
    bs->size = size;
    bs->bit_pos = 0;
}

static int bs_read_bit(BitstreamReader* bs) {
    if (bs->bit_pos / 8 >= bs->size) return -1;
    int byte_idx = (int)(bs->bit_pos / 8);
    int bit_idx = 7 - (int)(bs->bit_pos % 8);
    bs->bit_pos++;
    return (bs->data[byte_idx] >> bit_idx) & 1;
}

static uint32_t bs_read_bits(BitstreamReader* bs, int n) {
    uint32_t val = 0;
    for (int i = 0; i < n; i++) {
        int bit = bs_read_bit(bs);
        if (bit < 0) return 0;
        val = (val << 1) | (uint32_t)bit;
    }
    return val;
}

static uint32_t bs_read_ue(BitstreamReader* bs) {
    int leading_zeros = 0;
    while (bs_read_bit(bs) == 0 && leading_zeros < 31)
        leading_zeros++;
    if (leading_zeros == 0) return 0;
    uint32_t val = bs_read_bits(bs, leading_zeros);
    return (1u << leading_zeros) - 1 + val;
}

static uint8_t* rbsp_from_nal(const uint8_t* nal, size_t nal_size, size_t* rbsp_size) {
    uint8_t* rbsp = (uint8_t*)malloc(nal_size);
    if (!rbsp) return NULL;

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

static int find_start_code(const uint8_t* buf, size_t len, size_t start, size_t* sc_pos, int* sc_len) {
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

CMLHEVCParser* cml_hevc_parser_create(void) {
    CMLHEVCParser* p = (CMLHEVCParser*)calloc(1, sizeof(CMLHEVCParser));
    if (!p) return NULL;

    p->ring = (uint8_t*)malloc(HEVC_RING_INITIAL_CAP);
    if (!p->ring) {
        free(p);
        return NULL;
    }
    p->ring_cap = HEVC_RING_INITIAL_CAP;
    p->ring_len = 0;
    p->scan_pos = 0;
    p->frame_count = 0;
    return p;
}

void cml_hevc_parser_free(CMLHEVCParser* parser) {
    if (!parser) return;
    free(parser->ring);
    free(parser);
}

int cml_hevc_parser_feed(CMLHEVCParser* parser, const uint8_t* data, size_t size) {
    if (!parser || !data || size == 0) return -1;

    size_t needed = parser->ring_len + size;
    if (needed > parser->ring_cap) {
        size_t new_cap = parser->ring_cap;
        while (new_cap < needed) new_cap *= 2;
        uint8_t* new_ring = (uint8_t*)realloc(parser->ring, new_cap);
        if (!new_ring) {
            LOG_ERROR("HEVC: ring buffer realloc failed (%zu bytes)", new_cap);
            return -1;
        }
        parser->ring = new_ring;
        parser->ring_cap = new_cap;
    }

    memcpy(parser->ring + parser->ring_len, data, size);
    parser->ring_len += size;
    return 0;
}

CMLHEVCNalUnit* cml_hevc_next_nal(CMLHEVCParser* parser) {
    if (!parser || parser->ring_len < 4) return NULL;

    size_t first_pos, second_pos;
    int first_len, second_len;

    if (!find_start_code(parser->ring, parser->ring_len, parser->scan_pos, &first_pos, &first_len))
        return NULL;

    size_t nal_start = first_pos + first_len;

    if (!find_start_code(parser->ring, parser->ring_len, nal_start, &second_pos, &second_len)) {
        return NULL;
    }

    size_t nal_size = second_pos - nal_start;
    if (nal_size < 2) {
        parser->scan_pos = second_pos;
        return NULL;
    }

    CMLHEVCNalUnit* nal = (CMLHEVCNalUnit*)calloc(1, sizeof(CMLHEVCNalUnit));
    if (!nal) return NULL;

    uint8_t* nal_data = (uint8_t*)malloc(nal_size);
    if (!nal_data) {
        free(nal);
        return NULL;
    }
    memcpy(nal_data, parser->ring + nal_start, nal_size);

    uint8_t byte0 = nal_data[0];
    uint8_t byte1 = nal_data[1];
    int forbidden = (byte0 >> 7) & 1;
    int nal_type = (byte0 >> 1) & 0x3F;
    int nuh_layer_id = ((byte0 & 1) << 5) | ((byte1 >> 3) & 0x1F);
    int nuh_temporal_id_plus1 = byte1 & 0x07;
    (void)forbidden;
    (void)nuh_layer_id;

    nal->data = nal_data;
    nal->size = nal_size;
    nal->type = nal_type;
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

void cml_hevc_nal_free(CMLHEVCNalUnit* nal) {
    if (!nal) return;
    free((void*)nal->data);
    free(nal);
}

int cml_hevc_parse_sps(const uint8_t* sps_data, size_t sps_size, int* width, int* height) {
    if (!sps_data || sps_size < 4 || !width || !height) return -1;

    size_t rbsp_size = 0;
    uint8_t* rbsp = rbsp_from_nal(sps_data + 2, sps_size - 2, &rbsp_size);
    if (!rbsp) return -1;

    BitstreamReader bs;
    bs_init(&bs, rbsp, rbsp_size);

    uint32_t sps_video_parameter_set_id = bs_read_bits(&bs, 4);
    uint32_t sps_max_sub_layers_minus1 = bs_read_bits(&bs, 3);
    uint32_t sps_temporal_id_nesting_flag = bs_read_bits(&bs, 1);
    (void)sps_video_parameter_set_id;
    (void)sps_temporal_id_nesting_flag;

    /* profile_tier_level: general_profile_space(2) + general_tier_flag(1) + general_profile_idc(5) */
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
        uint32_t sub_layer_level_present = bs_read_bits(&bs, 1);
        (void)sub_layer_profile_present;
        (void)sub_layer_level_present;
    }
    if (sps_max_sub_layers_minus1 > 0) {
        for (uint32_t i = sps_max_sub_layers_minus1; i < 8; i++)
            bs_read_bits(&bs, 2);
    }

    for (uint32_t i = 0; i < sps_max_sub_layers_minus1; i++) {
        /* skip sub_layer profile_tier_level — simplified */
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

    uint32_t pic_width = bs_read_ue(&bs);
    uint32_t pic_height = bs_read_ue(&bs);

    uint32_t conformance_window_flag = bs_read_bits(&bs, 1);
    if (conformance_window_flag) {
        uint32_t left = bs_read_ue(&bs);
        uint32_t right = bs_read_ue(&bs);
        uint32_t top = bs_read_ue(&bs);
        uint32_t bottom = bs_read_ue(&bs);

        int sub_width_c = (chroma_format_idc == 1 || chroma_format_idc == 2) ? 2 : 1;
        int sub_height_c = (chroma_format_idc == 1) ? 2 : 1;

        pic_width -= (left + right) * (uint32_t)sub_width_c;
        pic_height -= (top + bottom) * (uint32_t)sub_height_c;
    }

    *width = (int)pic_width;
    *height = (int)pic_height;

    free(rbsp);
    return 0;
}

CMLHEVCFrame* cml_hevc_decode_iframe(CMLHEVCParser* parser, CMLHEVCNalUnit* nal) {
    if (!parser || !nal) return NULL;

    if (nal->type != HEVC_NAL_IDR_W_RADL && nal->type != HEVC_NAL_IDR_N_LP)
        return NULL;

    if (nal->size < 3) return NULL;

    size_t payload_offset = 2;
    size_t payload_size = nal->size - payload_offset;

    size_t rbsp_size = 0;
    uint8_t* rbsp = rbsp_from_nal(nal->data + payload_offset, payload_size, &rbsp_size);
    if (!rbsp) return NULL;

    /* Stub: no actual HEVC decoding — allocate a placeholder luma plane.
     * Real decode requires slice header parsing, prediction, transform, deblocking. */
    int w = 64, h = 64;
    int stride = w;

    CMLHEVCFrame* frame = (CMLHEVCFrame*)calloc(1, sizeof(CMLHEVCFrame));
    if (!frame) {
        free(rbsp);
        return NULL;
    }

    frame->data = (uint8_t*)calloc((size_t)(stride * h), 1);
    if (!frame->data) {
        free(rbsp);
        free(frame);
        return NULL;
    }

    size_t copy_len = rbsp_size < (size_t)(stride * h) ? rbsp_size : (size_t)(stride * h);
    memcpy(frame->data, rbsp, copy_len);

    frame->width = w;
    frame->height = h;
    frame->stride = stride;
    frame->pts = parser->frame_count++;
    frame->nal_type = nal->type;

    free(rbsp);
    return frame;
}

void cml_hevc_frame_free(CMLHEVCFrame* frame) {
    if (!frame) return;
    free(frame->data);
    free(frame);
}
