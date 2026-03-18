#include "core/protobuf_mini.h"
#include <string.h>

void pb_reader_init(PBReader *reader, const uint8_t *data, size_t length)
{
    if (!reader) return;
    reader->data   = data;
    reader->length = length;
    reader->pos    = 0;
}

uint64_t pb_read_varint(PBReader *reader)
{
    uint64_t result = 0;
    int shift = 0;

    while (reader->pos < reader->length) {
        uint8_t byte = reader->data[reader->pos++];
        result |= (uint64_t)(byte & 0x7F) << shift;
        if ((byte & 0x80) == 0) {
            break;
        }
        shift += 7;
        if (shift >= 64) {
            break;
        }
    }

    return result;
}

int64_t pb_read_svarint(PBReader *reader)
{
    uint64_t n = pb_read_varint(reader);
    /* Zigzag decode: (n >> 1) ^ -(n & 1) */
    return (int64_t)((n >> 1) ^ (-(n & 1)));
}

uint32_t pb_read_fixed32(PBReader *reader)
{
    if (reader->pos + 4 > reader->length) return 0;

    uint32_t v = 0;
    v |= (uint32_t)reader->data[reader->pos + 0];
    v |= (uint32_t)reader->data[reader->pos + 1] << 8;
    v |= (uint32_t)reader->data[reader->pos + 2] << 16;
    v |= (uint32_t)reader->data[reader->pos + 3] << 24;
    reader->pos += 4;
    return v;
}

uint64_t pb_read_fixed64(PBReader *reader)
{
    if (reader->pos + 8 > reader->length) return 0;

    uint64_t v = 0;
    for (int i = 0; i < 8; i++) {
        v |= (uint64_t)reader->data[reader->pos + (size_t)i] << (i * 8);
    }
    reader->pos += 8;
    return v;
}

float pb_read_float(PBReader *reader)
{
    uint32_t bits = pb_read_fixed32(reader);
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

double pb_read_double(PBReader *reader)
{
    uint64_t bits = pb_read_fixed64(reader);
    double d;
    memcpy(&d, &bits, sizeof(d));
    return d;
}

bool pb_read_field(PBReader *reader, PBField *field)
{
    if (!reader || !field) return false;
    if (reader->pos >= reader->length) return false;

    uint64_t tag = pb_read_varint(reader);
    field->field_number = (uint32_t)(tag >> 3);
    field->wire_type    = (PBWireType)(tag & 7);

    if (field->field_number == 0) return false;

    switch (field->wire_type) {

    case PB_WIRE_VARINT:
        field->value.varint = pb_read_varint(reader);
        break;

    case PB_WIRE_FIXED64:
        field->value.fixed64 = pb_read_fixed64(reader);
        break;

    case PB_WIRE_LEN: {
        uint64_t len = pb_read_varint(reader);
        if (reader->pos + len > reader->length) {
            reader->pos = reader->length;
            return false;
        }
        field->value.bytes.data   = reader->data + reader->pos;
        field->value.bytes.length = (size_t)len;
        reader->pos += (size_t)len;
        break;
    }

    case PB_WIRE_FIXED32:
        field->value.fixed32 = pb_read_fixed32(reader);
        break;

    default:
        return false;
    }

    return true;
}

PBReader pb_reader_sub(const PBField *field)
{
    PBReader sub;
    if (field && field->wire_type == PB_WIRE_LEN) {
        sub.data   = field->value.bytes.data;
        sub.length = field->value.bytes.length;
    } else {
        sub.data   = NULL;
        sub.length = 0;
    }
    sub.pos = 0;
    return sub;
}

const char *pb_field_string(const PBField *field, size_t *out_len)
{
    if (!field || field->wire_type != PB_WIRE_LEN) {
        if (out_len) *out_len = 0;
        return NULL;
    }
    if (out_len) *out_len = field->value.bytes.length;
    return (const char *)field->value.bytes.data;
}

bool pb_reader_has_data(const PBReader *reader)
{
    if (!reader) return false;
    return reader->pos < reader->length;
}

void pb_skip_field(PBReader *reader, const PBField *field)
{
    if (!reader || !field) return;

    /*
     * After pb_read_field() the reader position has already been advanced
     * past the field value for all wire types we handle (varint, fixed32,
     * fixed64, LEN).  There is nothing more to skip -- this function is
     * provided for explicitness in calling code.
     *
     * Wire types 3/4 (start/end group, deprecated) require recursion,
     * but ONNX never uses groups.
     */
    (void)reader;
    (void)field;
}
