/**
 * @file protobuf_mini.h
 * @brief Minimal C11 protobuf wire-format decoder
 *
 * Supports varint, fixed32, fixed64, and length-delimited fields.
 * Used by the ONNX runtime to parse .onnx model files.
 */

#ifndef CML_CORE_PROTOBUF_MINI_H
#define CML_CORE_PROTOBUF_MINI_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Wire types ── */

typedef enum {
    PB_WIRE_VARINT  = 0,
    PB_WIRE_FIXED64 = 1,
    PB_WIRE_LEN     = 2,
    PB_WIRE_FIXED32 = 5,
} PBWireType;

/* ── Field value ── */

typedef struct {
    uint32_t field_number;
    PBWireType wire_type;
    union {
        uint64_t varint;
        uint32_t fixed32;
        uint64_t fixed64;
        float    float_val;
        double   double_val;
        struct {
            const uint8_t* data;
            size_t length;
        } bytes;
    } value;
} PBField;

/* ── Reader ── */

typedef struct {
    const uint8_t* data;
    size_t length;
    size_t pos;
} PBReader;

/* ── API ── */

void pb_reader_init(PBReader* reader, const uint8_t* data, size_t length);

/**
 * @brief Read next field from protobuf stream
 * @return true if field read, false if end of data
 */
bool pb_read_field(PBReader* reader, PBField* field);

/**
 * @brief Read a varint from the current position
 */
uint64_t pb_read_varint(PBReader* reader);

/**
 * @brief Read a signed varint (zigzag decoded)
 */
int64_t pb_read_svarint(PBReader* reader);

/**
 * @brief Read fixed32
 */
uint32_t pb_read_fixed32(PBReader* reader);

/**
 * @brief Read fixed64
 */
uint64_t pb_read_fixed64(PBReader* reader);

/**
 * @brief Read a float (fixed32)
 */
float pb_read_float(PBReader* reader);

/**
 * @brief Read a double (fixed64)
 */
double pb_read_double(PBReader* reader);

/**
 * @brief Create a sub-reader for a length-delimited field
 */
PBReader pb_reader_sub(const PBField* field);

/**
 * @brief Read a string from a length-delimited field (not null-terminated)
 */
const char* pb_field_string(const PBField* field, size_t* out_len);

/**
 * @brief Check if reader has more data
 */
bool pb_reader_has_data(const PBReader* reader);

/**
 * @brief Skip current field value
 */
void pb_skip_field(PBReader* reader, const PBField* field);

#ifdef __cplusplus
}
#endif

#endif /* CML_CORE_PROTOBUF_MINI_H */
