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

#define HEVC_NAL_TRAIL_N     0
#define HEVC_NAL_TRAIL_R     1
#define HEVC_NAL_IDR_W_RADL 19
#define HEVC_NAL_IDR_N_LP   20
#define HEVC_NAL_VPS        32
#define HEVC_NAL_SPS        33
#define HEVC_NAL_PPS        34
#define HEVC_NAL_AUD        35
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
CMLHEVCNalUnit* cml_hevc_next_nal(CMLHEVCParser* parser);
void cml_hevc_nal_free(CMLHEVCNalUnit* nal);

int cml_hevc_parse_sps(const uint8_t* sps_data, size_t sps_size, int* width, int* height);

CMLHEVCFrame* cml_hevc_decode_iframe(CMLHEVCParser* parser, CMLHEVCNalUnit* nal);
void cml_hevc_frame_free(CMLHEVCFrame* frame);

#ifdef __cplusplus
}
#endif

#endif /* CML_CORE_HEVC_H */
