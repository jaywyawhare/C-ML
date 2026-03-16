/**
 * @file onnx.c
 * @brief ONNX protobuf parsing
 *
 * Parses an ONNX ModelProto from a file or memory buffer using the minimal
 * protobuf decoder (protobuf_mini.h).  Produces a CMLONNXModel that the
 * ONNX operator mapping layer (onnx_ops.c) can execute.
 *
 * Protobuf field numbers follow the official ONNX IR specification:
 *   https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
 */

#include "core/onnx.h"
#include "core/protobuf_mini.h"
#include "core/logging.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void copy_pb_string(char *dst, size_t dst_size, const PBField *field)
{
    size_t len = 0;
    const char *src = pb_field_string(field, &len);
    if (!src || len == 0) {
        dst[0] = '\0';
        return;
    }
    if (len >= dst_size) len = dst_size - 1;
    memcpy(dst, src, len);
    dst[len] = '\0';
}

/**
 * Allocate a NUL-terminated heap copy of a protobuf string field.
 * The caller must free() the result.
 */
static char *dup_pb_string(const PBField *field)
{
    size_t len = 0;
    const char *src = pb_field_string(field, &len);
    if (!src || len == 0) return NULL;

    char *s = (char *)malloc(len + 1);
    if (!s) return NULL;
    memcpy(s, src, len);
    s[len] = '\0';
    return s;
}

static DType onnx_dtype_to_cml(int onnx_dtype)
{
    switch (onnx_dtype) {
    case 1:  return DTYPE_FLOAT32;   /* FLOAT  */
    case 2:  return DTYPE_UINT8;     /* UINT8  */
    case 3:  return DTYPE_INT8;      /* INT8   */
    case 5:  return DTYPE_INT16;     /* INT16  */
    case 6:  return DTYPE_INT32;     /* INT32  */
    case 7:  return DTYPE_INT64;     /* INT64  */
    case 10: return DTYPE_FLOAT16;   /* FLOAT16 */
    case 11: return DTYPE_FLOAT64;   /* DOUBLE */
    case 12: return DTYPE_UINT32;    /* UINT32 */
    case 13: return DTYPE_UINT64;    /* UINT64 */
    case 16: return DTYPE_BFLOAT16;  /* BFLOAT16 */
    default: return DTYPE_FLOAT32;
    }
}

/*
 * AttributeProto field numbers (onnx.proto):
 *   1: name           (string)
 *   2: f              (float)
 *   3: i              (int64)
 *   4: s              (bytes/string)
 *   6: floats         (repeated float, packed)
 *   7: ints           (repeated int64, packed)
 *  20: type           (int32, AttributeType enum)
 */

static void parse_attribute(PBReader *rd, CMLONNXAttribute *attr)
{
    memset(attr, 0, sizeof(*attr));

    PBField f;
    while (pb_read_field(rd, &f)) {
        switch (f.field_number) {
        case 1: /* name */
            copy_pb_string(attr->name, sizeof(attr->name), &f);
            break;

        case 2: /* f (float, stored as fixed32) */
            if (f.wire_type == PB_WIRE_FIXED32) {
                memcpy(&attr->value.f, &f.value.fixed32, sizeof(float));
                attr->type = CML_ONNX_ATTR_FLOAT;
            }
            break;

        case 3: /* i (int64, varint) */
            attr->value.i = (int64_t)f.value.varint;
            attr->type = CML_ONNX_ATTR_INT;
            break;

        case 4: /* s (string / bytes) */
            if (f.wire_type == PB_WIRE_LEN) {
                size_t slen = f.value.bytes.length;
                if (slen >= sizeof(attr->value.s.data))
                    slen = sizeof(attr->value.s.data) - 1;
                memcpy(attr->value.s.data, f.value.bytes.data, slen);
                attr->value.s.data[slen] = '\0';
                attr->value.s.len = slen;
                attr->type = CML_ONNX_ATTR_STRING;
            }
            break;

        case 6: /* floats (packed repeated float) */
            if (f.wire_type == PB_WIRE_LEN) {
                int count = (int)(f.value.bytes.length / sizeof(float));
                attr->value.floats.data = (float *)malloc(sizeof(float) * (size_t)count);
                if (attr->value.floats.data) {
                    PBReader sub = pb_reader_sub(&f);
                    for (int i = 0; i < count; i++) {
                        attr->value.floats.data[i] = pb_read_float(&sub);
                    }
                    attr->value.floats.count = count;
                }
                attr->type = CML_ONNX_ATTR_FLOATS;
            }
            break;

        case 7: /* ints (packed repeated int64 -- but also non-packed) */
            if (f.wire_type == PB_WIRE_LEN) {
                /* Packed encoding: each element is a varint */
                PBReader sub = pb_reader_sub(&f);
                /* First pass: count */
                PBReader counter = sub;
                int count = 0;
                while (pb_reader_has_data(&counter)) {
                    pb_read_varint(&counter);
                    count++;
                }
                attr->value.ints.data = (int64_t *)malloc(sizeof(int64_t) * (size_t)count);
                if (attr->value.ints.data) {
                    for (int i = 0; i < count; i++) {
                        attr->value.ints.data[i] = (int64_t)pb_read_varint(&sub);
                    }
                    attr->value.ints.count = count;
                }
                attr->type = CML_ONNX_ATTR_INTS;
            } else if (f.wire_type == PB_WIRE_VARINT) {
                /* Non-packed single int in repeated field -- append */
                int cur = attr->value.ints.count;
                int64_t *tmp = (int64_t *)realloc(attr->value.ints.data,
                                                   sizeof(int64_t) * (size_t)(cur + 1));
                if (tmp) {
                    tmp[cur] = (int64_t)f.value.varint;
                    attr->value.ints.data = tmp;
                    attr->value.ints.count = cur + 1;
                }
                attr->type = CML_ONNX_ATTR_INTS;
            }
            break;

        default:
            pb_skip_field(rd, &f);
            break;
        }
    }
}

/*
 * NodeProto field numbers:
 *   1: input     (repeated string)
 *   2: output    (repeated string)
 *   3: name      (string)
 *   4: op_type   (string)
 *   5: attribute (repeated AttributeProto, LEN)
 */

static void parse_node(PBReader *rd, CMLONNXNode *node)
{
    memset(node, 0, sizeof(*node));

    PBField f;
    while (pb_read_field(rd, &f)) {
        switch (f.field_number) {
        case 1: /* input (repeated string) */
            if (node->num_inputs < CML_ONNX_MAX_INPUTS) {
                node->inputs[node->num_inputs] = dup_pb_string(&f);
                node->num_inputs++;
            }
            break;

        case 2: /* output (repeated string) */
            if (node->num_outputs < CML_ONNX_MAX_OUTPUTS) {
                node->outputs[node->num_outputs] = dup_pb_string(&f);
                node->num_outputs++;
            }
            break;

        case 3: /* name */
            copy_pb_string(node->name, sizeof(node->name), &f);
            break;

        case 4: /* op_type */
            copy_pb_string(node->op_type, sizeof(node->op_type), &f);
            break;

        case 5: /* attribute (LEN-delimited repeated) */
            if (f.wire_type == PB_WIRE_LEN && node->num_attrs < CML_ONNX_MAX_ATTRS) {
                PBReader sub = pb_reader_sub(&f);
                parse_attribute(&sub, &node->attrs[node->num_attrs]);
                node->num_attrs++;
            }
            break;

        default:
            pb_skip_field(rd, &f);
            break;
        }
    }
}

/*
 * TensorProto field numbers:
 *   1: dims        (repeated int64)
 *   2: data_type   (int32)
 *   4: raw_data    (bytes)
 *   5: float_data  (packed repeated float)
 *   8: name        (string)
 */

static void parse_tensor_proto(PBReader *rd, CMLONNXInitializer *init)
{
    memset(init, 0, sizeof(*init));

    int dims[8];
    int ndim = 0;
    int onnx_dtype = 1; /* default FLOAT */
    const uint8_t *raw_data = NULL;
    size_t raw_len = 0;
    float *float_data = NULL;
    int float_count = 0;

    PBField f;
    while (pb_read_field(rd, &f)) {
        switch (f.field_number) {
        case 1: /* dims (repeated int64) */
            if (f.wire_type == PB_WIRE_VARINT) {
                /* Non-packed */
                if (ndim < 8) dims[ndim++] = (int)f.value.varint;
            } else if (f.wire_type == PB_WIRE_LEN) {
                /* Packed */
                PBReader sub = pb_reader_sub(&f);
                while (pb_reader_has_data(&sub) && ndim < 8) {
                    dims[ndim++] = (int)pb_read_varint(&sub);
                }
            }
            break;

        case 2: /* data_type */
            onnx_dtype = (int)f.value.varint;
            break;

        case 4: /* raw_data */
            if (f.wire_type == PB_WIRE_LEN) {
                raw_data = f.value.bytes.data;
                raw_len  = f.value.bytes.length;
            }
            break;

        case 5: /* float_data (packed repeated float) */
            if (f.wire_type == PB_WIRE_LEN) {
                float_count = (int)(f.value.bytes.length / sizeof(float));
                float_data = (float *)malloc(sizeof(float) * (size_t)float_count);
                if (float_data) {
                    PBReader sub = pb_reader_sub(&f);
                    for (int i = 0; i < float_count; i++) {
                        float_data[i] = pb_read_float(&sub);
                    }
                }
            }
            break;

        case 8: /* name */
            copy_pb_string(init->name, sizeof(init->name), &f);
            break;

        default:
            pb_skip_field(rd, &f);
            break;
        }
    }

    /* Scalar tensors with no dims */
    if (ndim == 0) {
        ndim = 1;
        dims[0] = 1;
    }

    DType dtype = onnx_dtype_to_cml(onnx_dtype);
    TensorConfig cfg = { .dtype = dtype, .has_dtype = true };

    if (raw_data && raw_len > 0) {
        init->tensor = tensor_from_data(raw_data, dims, ndim, &cfg);
    } else if (float_data && float_count > 0) {
        init->tensor = tensor_from_data(float_data, dims, ndim, &cfg);
    } else {
        /* Empty initializer -- create zeros */
        init->tensor = tensor_zeros(dims, ndim, &cfg);
    }

    free(float_data);
}

/*
 * ValueInfoProto:
 *   1: name   (string)
 *   2: type   (TypeProto, LEN)
 *
 * TypeProto:
 *   1: tensor_type (TypeProto.Tensor, LEN)
 *
 * TypeProto.Tensor:
 *   1: elem_type (int32 / varint)
 *   2: shape     (TensorShapeProto, LEN)
 *
 * TensorShapeProto:
 *   1: dim (repeated TensorShapeProto.Dimension, LEN)
 *
 * TensorShapeProto.Dimension:
 *   1: dim_value (int64)
 *   2: dim_param (string -- symbolic)
 */

static void parse_tensor_shape_dim(PBReader *rd, int *dim_out)
{
    *dim_out = -1; /* dynamic / unknown */
    PBField f;
    while (pb_read_field(rd, &f)) {
        if (f.field_number == 1) { /* dim_value */
            *dim_out = (int)f.value.varint;
        }
        /* field 2: dim_param (symbolic, ignored) */
    }
}

static void parse_tensor_shape(PBReader *rd, CMLONNXTensorInfo *info)
{
    PBField f;
    while (pb_read_field(rd, &f)) {
        if (f.field_number == 1 && f.wire_type == PB_WIRE_LEN) { /* dim */
            PBReader sub = pb_reader_sub(&f);
            if (info->ndim < 8) {
                parse_tensor_shape_dim(&sub, &info->shape[info->ndim]);
                info->ndim++;
            }
        }
    }
}

static void parse_tensor_type(PBReader *rd, CMLONNXTensorInfo *info)
{
    PBField f;
    while (pb_read_field(rd, &f)) {
        switch (f.field_number) {
        case 1: /* elem_type */
            info->dtype = onnx_dtype_to_cml((int)f.value.varint);
            break;
        case 2: /* shape */
            if (f.wire_type == PB_WIRE_LEN) {
                PBReader sub = pb_reader_sub(&f);
                parse_tensor_shape(&sub, info);
            }
            break;
        default:
            break;
        }
    }
}

static void parse_type_proto(PBReader *rd, CMLONNXTensorInfo *info)
{
    PBField f;
    while (pb_read_field(rd, &f)) {
        if (f.field_number == 1 && f.wire_type == PB_WIRE_LEN) { /* tensor_type */
            PBReader sub = pb_reader_sub(&f);
            parse_tensor_type(&sub, info);
        }
    }
}

static void parse_value_info(PBReader *rd, CMLONNXTensorInfo *info)
{
    memset(info, 0, sizeof(*info));
    info->dtype = DTYPE_FLOAT32;

    PBField f;
    while (pb_read_field(rd, &f)) {
        switch (f.field_number) {
        case 1: /* name */
            copy_pb_string(info->name, sizeof(info->name), &f);
            break;
        case 2: /* type (TypeProto) */
            if (f.wire_type == PB_WIRE_LEN) {
                PBReader sub = pb_reader_sub(&f);
                parse_type_proto(&sub, info);
            }
            break;
        default:
            break;
        }
    }
}

/*
 * GraphProto field numbers:
 *   1: node         (repeated NodeProto, LEN)
 *   2: name         (string)
 *   5: initializer  (repeated TensorProto, LEN)
 *  11: input        (repeated ValueInfoProto, LEN)
 *  12: output       (repeated ValueInfoProto, LEN)
 */

static int parse_graph(PBReader *rd, CMLONNXGraph *graph)
{
    memset(graph, 0, sizeof(*graph));

    /* Pre-allocate arrays */
    graph->nodes        = (CMLONNXNode *)calloc(CML_ONNX_MAX_NODES, sizeof(CMLONNXNode));
    graph->inputs       = (CMLONNXTensorInfo *)calloc(CML_ONNX_MAX_INPUTS, sizeof(CMLONNXTensorInfo));
    graph->outputs      = (CMLONNXTensorInfo *)calloc(CML_ONNX_MAX_OUTPUTS, sizeof(CMLONNXTensorInfo));
    graph->initializers = (CMLONNXInitializer *)calloc(CML_ONNX_MAX_NODES, sizeof(CMLONNXInitializer));

    if (!graph->nodes || !graph->inputs || !graph->outputs || !graph->initializers) {
        return -1;
    }

    PBField f;
    while (pb_read_field(rd, &f)) {
        switch (f.field_number) {

        case 1: /* node */
            if (f.wire_type == PB_WIRE_LEN && graph->num_nodes < CML_ONNX_MAX_NODES) {
                PBReader sub = pb_reader_sub(&f);
                parse_node(&sub, &graph->nodes[graph->num_nodes]);
                graph->num_nodes++;
            }
            break;

        case 2: /* name */
            copy_pb_string(graph->name, sizeof(graph->name), &f);
            break;

        case 5: /* initializer */
            if (f.wire_type == PB_WIRE_LEN && graph->num_initializers < CML_ONNX_MAX_NODES) {
                PBReader sub = pb_reader_sub(&f);
                parse_tensor_proto(&sub, &graph->initializers[graph->num_initializers]);
                graph->num_initializers++;
            }
            break;

        case 11: /* input */
            if (f.wire_type == PB_WIRE_LEN && graph->num_inputs < CML_ONNX_MAX_INPUTS) {
                PBReader sub = pb_reader_sub(&f);
                parse_value_info(&sub, &graph->inputs[graph->num_inputs]);
                graph->num_inputs++;
            }
            break;

        case 12: /* output */
            if (f.wire_type == PB_WIRE_LEN && graph->num_outputs < CML_ONNX_MAX_OUTPUTS) {
                PBReader sub = pb_reader_sub(&f);
                parse_value_info(&sub, &graph->outputs[graph->num_outputs]);
                graph->num_outputs++;
            }
            break;

        default:
            pb_skip_field(rd, &f);
            break;
        }
    }

    return 0;
}

/*
 * OpsetIdProto:
 *   1: domain   (string, "" = default ai.onnx)
 *   2: version  (int64)
 */

static int64_t parse_opset(PBReader *rd)
{
    int64_t version = 0;
    PBField f;
    while (pb_read_field(rd, &f)) {
        if (f.field_number == 2) {
            version = (int64_t)f.value.varint;
        }
    }
    return version;
}

/*
 * ModelProto field numbers:
 *   1: ir_version    (int64)
 *   2: producer_name (string)
 *   3: producer_version (string, ignored)
 *   4: domain        (string)
 *   7: graph         (GraphProto, LEN)
 *   8: opset_import  (repeated OpsetIdProto, LEN)
 */

CMLONNXModel *cml_onnx_load_buffer(const uint8_t *data, size_t length)
{
    if (!data || length == 0) {
        LOG_ERROR("onnx: NULL or empty buffer");
        return NULL;
    }

    CMLONNXModel *model = (CMLONNXModel *)calloc(1, sizeof(CMLONNXModel));
    if (!model) return NULL;

    PBReader reader;
    pb_reader_init(&reader, data, length);

    PBField f;
    while (pb_read_field(&reader, &f)) {
        switch (f.field_number) {

        case 1: /* ir_version */
            model->ir_version = (int64_t)f.value.varint;
            break;

        case 2: /* producer_name */
            copy_pb_string(model->producer_name, sizeof(model->producer_name), &f);
            break;

        case 4: /* domain */
            copy_pb_string(model->domain, sizeof(model->domain), &f);
            break;

        case 7: /* graph (GraphProto) */
            if (f.wire_type == PB_WIRE_LEN) {
                PBReader sub = pb_reader_sub(&f);
                if (parse_graph(&sub, &model->graph) != 0) {
                    LOG_ERROR("onnx: failed to parse graph");
                    cml_onnx_free(model);
                    return NULL;
                }
            }
            break;

        case 8: /* opset_import */
            if (f.wire_type == PB_WIRE_LEN) {
                PBReader sub = pb_reader_sub(&f);
                int64_t v = parse_opset(&sub);
                if (v > model->opset_version) {
                    model->opset_version = v;
                }
            }
            break;

        default:
            pb_skip_field(&reader, &f);
            break;
        }
    }

    LOG_INFO("onnx: loaded model ir=%lld opset=%lld producer=%s nodes=%d inits=%d",
             (long long)model->ir_version, (long long)model->opset_version,
             model->producer_name, model->graph.num_nodes,
             model->graph.num_initializers);

    return model;
}

CMLONNXModel *cml_onnx_load(const char *filepath)
{
    if (!filepath) return NULL;

    FILE *fp = fopen(filepath, "rb");
    if (!fp) {
        LOG_ERROR("onnx: cannot open file '%s'", filepath);
        return NULL;
    }

    /* Determine file size */
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (fsize <= 0) {
        LOG_ERROR("onnx: empty or unreadable file '%s'", filepath);
        fclose(fp);
        return NULL;
    }

    uint8_t *buf = (uint8_t *)malloc((size_t)fsize);
    if (!buf) {
        fclose(fp);
        return NULL;
    }

    size_t n = fread(buf, 1, (size_t)fsize, fp);
    fclose(fp);

    if ((long)n != fsize) {
        LOG_ERROR("onnx: short read on '%s'", filepath);
        free(buf);
        return NULL;
    }

    CMLONNXModel *model = cml_onnx_load_buffer(buf, (size_t)fsize);
    free(buf);
    return model;
}

void cml_onnx_free(CMLONNXModel *model)
{
    if (!model) return;

    CMLONNXGraph *g = &model->graph;

    /* Free node strings */
    if (g->nodes) {
        for (int i = 0; i < g->num_nodes; i++) {
            CMLONNXNode *nd = &g->nodes[i];
            for (int j = 0; j < nd->num_inputs; j++) {
                free(nd->inputs[j]);
            }
            for (int j = 0; j < nd->num_outputs; j++) {
                free(nd->outputs[j]);
            }
            /* Free attribute heap data */
            for (int j = 0; j < nd->num_attrs; j++) {
                CMLONNXAttribute *a = &nd->attrs[j];
                if (a->type == CML_ONNX_ATTR_INTS) {
                    free(a->value.ints.data);
                } else if (a->type == CML_ONNX_ATTR_FLOATS) {
                    free(a->value.floats.data);
                } else if (a->type == CML_ONNX_ATTR_TENSOR && a->value.tensor) {
                    tensor_free(a->value.tensor);
                }
            }
        }
        free(g->nodes);
    }

    /* Free initializer tensors */
    if (g->initializers) {
        for (int i = 0; i < g->num_initializers; i++) {
            if (g->initializers[i].tensor) {
                tensor_free(g->initializers[i].tensor);
            }
        }
        free(g->initializers);
    }

    free(g->inputs);
    free(g->outputs);

    free(model);
}
