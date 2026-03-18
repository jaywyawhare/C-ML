#ifndef CML_CORE_PROTOBUF_MINI_H
#define CML_CORE_PROTOBUF_MINI_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PB_WIRE_VARINT  = 0,
    PB_WIRE_FIXED64 = 1,
    PB_WIRE_LEN     = 2,
    PB_WIRE_FIXED32 = 5,
} PBWireType;

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

typedef struct {
    const uint8_t* data;
    size_t length;
    size_t pos;
} PBReader;

void pb_reader_init(PBReader* reader, const uint8_t* data, size_t length);
bool pb_read_field(PBReader* reader, PBField* field);
uint64_t pb_read_varint(PBReader* reader);
int64_t pb_read_svarint(PBReader* reader);
uint32_t pb_read_fixed32(PBReader* reader);
uint64_t pb_read_fixed64(PBReader* reader);
float pb_read_float(PBReader* reader);
double pb_read_double(PBReader* reader);
PBReader pb_reader_sub(const PBField* field);

/* Returned string is not null-terminated */
const char* pb_field_string(const PBField* field, size_t* out_len);

bool pb_reader_has_data(const PBReader* reader);
void pb_skip_field(PBReader* reader, const PBField* field);

#ifdef __cplusplus
}
#endif

#endif /* CML_CORE_PROTOBUF_MINI_H */
