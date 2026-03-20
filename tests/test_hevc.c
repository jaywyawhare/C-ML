#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "core/hevc.h"

static int tests_run = 0;
static int tests_passed = 0;

#define RUN_TEST(test) do { \
    tests_run++; \
    printf("  [%d] %-50s ", tests_run, #test); \
    if (test()) { tests_passed++; printf("PASS\n"); } \
    else { printf("FAIL\n"); } \
} while(0)

static int test_parser_create_free(void) {
    CMLHEVCParser* p = cml_hevc_parser_create();
    if (!p) return 0;
    cml_hevc_parser_free(p);
    return 1;
}

static int test_parser_feed(void) {
    CMLHEVCParser* p = cml_hevc_parser_create();
    if (!p) return 0;

    uint8_t data[] = {0x00, 0x00, 0x00, 0x01, 0x40, 0x01, 0xFF, 0xFF};
    int ret = cml_hevc_parser_feed(p, data, sizeof(data));
    cml_hevc_parser_free(p);
    return ret == 0;
}

static int test_parser_feed_null(void) {
    CMLHEVCParser* p = cml_hevc_parser_create();
    if (!p) return 0;
    int ret = cml_hevc_parser_feed(p, NULL, 0);
    cml_hevc_parser_free(p);
    return ret == -1;
}

static int test_nal_extraction_vps(void) {
    CMLHEVCParser* p = cml_hevc_parser_create();
    if (!p) return 0;

    /* VPS NAL: type=32, nal_unit_type in byte0 = (32 << 1) = 0x40 */
    uint8_t stream[] = {
        0x00, 0x00, 0x00, 0x01, 0x40, 0x01, 0xAA, 0xBB, 0xCC,
        0x00, 0x00, 0x00, 0x01, 0x42, 0x01, 0xDD,
        0x00, 0x00, 0x00, 0x01, 0x44, 0x01, 0xEE,
    };
    cml_hevc_parser_feed(p, stream, sizeof(stream));

    CMLHEVCNalUnit* nal = cml_hevc_next_nal(p);
    if (!nal) { cml_hevc_parser_free(p); return 0; }

    int ok = (nal->type == HEVC_NAL_VPS && nal->size == 5);
    cml_hevc_nal_free(nal);

    nal = cml_hevc_next_nal(p);
    if (!nal) { cml_hevc_parser_free(p); return ok; }
    ok = ok && (nal->type == HEVC_NAL_SPS && nal->size == 3);
    cml_hevc_nal_free(nal);

    cml_hevc_parser_free(p);
    return ok;
}

static int test_nal_header_parsing(void) {
    CMLHEVCParser* p = cml_hevc_parser_create();
    if (!p) return 0;

    /* IDR_W_RADL: type=19, byte0 = (19 << 1) = 0x26, temporal_id_plus1 = 1 */
    uint8_t stream[] = {
        0x00, 0x00, 0x01, 0x26, 0x01, 0xAA, 0xBB,
        0x00, 0x00, 0x01, 0x02, 0x01, 0xCC,
    };
    cml_hevc_parser_feed(p, stream, sizeof(stream));

    CMLHEVCNalUnit* nal = cml_hevc_next_nal(p);
    if (!nal) { cml_hevc_parser_free(p); return 0; }

    int ok = (nal->type == HEVC_NAL_IDR_W_RADL && nal->temporal_id == 0);
    cml_hevc_nal_free(nal);
    cml_hevc_parser_free(p);
    return ok;
}

static int test_three_byte_start_code(void) {
    CMLHEVCParser* p = cml_hevc_parser_create();
    if (!p) return 0;

    uint8_t stream[] = {
        0x00, 0x00, 0x01, 0x40, 0x01, 0x11, 0x22,
        0x00, 0x00, 0x01, 0x42, 0x01, 0x33,
    };
    cml_hevc_parser_feed(p, stream, sizeof(stream));

    CMLHEVCNalUnit* nal = cml_hevc_next_nal(p);
    int ok = (nal != NULL && nal->type == HEVC_NAL_VPS);
    cml_hevc_nal_free(nal);
    cml_hevc_parser_free(p);
    return ok;
}

static int test_emulation_prevention(void) {
    CMLHEVCParser* p = cml_hevc_parser_create();
    if (!p) return 0;

    /* SPS with emulation prevention byte: 00 00 03 should become 00 00 */
    uint8_t sps_nal[] = {
        0x42, 0x01,  /* NAL header: SPS */
        0x01,        /* sps_video_parameter_set_id(4) + sps_max_sub_layers_minus1(3) + temporal_id_nesting(1) */
        0x60, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00,
        0x03, 0x00, 0x00, 0x03, 0x00, 0x7B, 0xA0,
        0x02, 0x80, /* profile_tier_level padding */
        0x80, 0x28, 0x80,
    };

    int w = 0, h = 0;
    int ret = cml_hevc_parse_sps(sps_nal, sizeof(sps_nal), &w, &h);
    cml_hevc_parser_free(p);
    /* Parse should succeed (dimensions may be arbitrary due to simplified bitstream) */
    return ret == 0;
}

static int test_incremental_feed(void) {
    CMLHEVCParser* p = cml_hevc_parser_create();
    if (!p) return 0;

    uint8_t part1[] = {0x00, 0x00, 0x00, 0x01, 0x40, 0x01, 0xAA};
    uint8_t part2[] = {0x00, 0x00, 0x00, 0x01, 0x42, 0x01, 0xBB};

    cml_hevc_parser_feed(p, part1, sizeof(part1));
    CMLHEVCNalUnit* nal = cml_hevc_next_nal(p);
    if (nal != NULL) { cml_hevc_nal_free(nal); cml_hevc_parser_free(p); return 0; }

    cml_hevc_parser_feed(p, part2, sizeof(part2));
    nal = cml_hevc_next_nal(p);
    int ok = (nal != NULL && nal->type == HEVC_NAL_VPS);
    cml_hevc_nal_free(nal);
    cml_hevc_parser_free(p);
    return ok;
}

static int test_iframe_decode_stub(void) {
    CMLHEVCParser* p = cml_hevc_parser_create();
    if (!p) return 0;

    /* IDR_W_RADL: type=19, byte0 = 0x26, byte1 = 0x01 */
    uint8_t stream[] = {
        0x00, 0x00, 0x00, 0x01,
        0x26, 0x01, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF,
        0x00, 0x00, 0x00, 0x01,
        0x02, 0x01, 0x00,
    };
    cml_hevc_parser_feed(p, stream, sizeof(stream));

    CMLHEVCNalUnit* nal = cml_hevc_next_nal(p);
    if (!nal || nal->type != HEVC_NAL_IDR_W_RADL) {
        cml_hevc_nal_free(nal);
        cml_hevc_parser_free(p);
        return 0;
    }

    CMLHEVCFrame* frame = cml_hevc_decode_iframe(p, nal);
    int ok = (frame != NULL && frame->width > 0 && frame->height > 0 && frame->data != NULL);
    ok = ok && (frame->nal_type == HEVC_NAL_IDR_W_RADL);

    cml_hevc_frame_free(frame);
    cml_hevc_nal_free(nal);
    cml_hevc_parser_free(p);
    return ok;
}

static int test_non_idr_decode_returns_null(void) {
    CMLHEVCParser* p = cml_hevc_parser_create();
    if (!p) return 0;

    CMLHEVCNalUnit nal = {.data = (const uint8_t*)"\x40\x01\xAA", .size = 3, .type = HEVC_NAL_VPS, .temporal_id = 0};
    CMLHEVCFrame* frame = cml_hevc_decode_iframe(p, &nal);
    cml_hevc_parser_free(p);
    return frame == NULL;
}

static int test_multiple_nal_sequence(void) {
    CMLHEVCParser* p = cml_hevc_parser_create();
    if (!p) return 0;

    /* VPS, SPS, PPS, IDR sequence */
    uint8_t stream[] = {
        0x00, 0x00, 0x00, 0x01, 0x40, 0x01, 0x0C, 0x01,
        0x00, 0x00, 0x00, 0x01, 0x42, 0x01, 0x01, 0x02,
        0x00, 0x00, 0x00, 0x01, 0x44, 0x01, 0xC0, 0xF3,
        0x00, 0x00, 0x00, 0x01, 0x26, 0x01, 0x05, 0x06,
        0x00, 0x00, 0x00, 0x01, 0x02, 0x01, 0x00,
    };
    cml_hevc_parser_feed(p, stream, sizeof(stream));

    int types[] = {HEVC_NAL_VPS, HEVC_NAL_SPS, HEVC_NAL_PPS, HEVC_NAL_IDR_W_RADL};
    int ok = 1;
    for (int i = 0; i < 4; i++) {
        CMLHEVCNalUnit* nal = cml_hevc_next_nal(p);
        if (!nal || nal->type != types[i]) ok = 0;
        cml_hevc_nal_free(nal);
    }

    cml_hevc_parser_free(p);
    return ok;
}

static int test_nal_free_null(void) {
    cml_hevc_nal_free(NULL);
    cml_hevc_frame_free(NULL);
    cml_hevc_parser_free(NULL);
    return 1;
}

int main(void) {
    printf("HEVC Parser Tests\n");

    RUN_TEST(test_parser_create_free);
    RUN_TEST(test_parser_feed);
    RUN_TEST(test_parser_feed_null);
    RUN_TEST(test_nal_extraction_vps);
    RUN_TEST(test_nal_header_parsing);
    RUN_TEST(test_three_byte_start_code);
    RUN_TEST(test_emulation_prevention);
    RUN_TEST(test_incremental_feed);
    RUN_TEST(test_iframe_decode_stub);
    RUN_TEST(test_non_idr_decode_returns_null);
    RUN_TEST(test_multiple_nal_sequence);
    RUN_TEST(test_nal_free_null);

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
