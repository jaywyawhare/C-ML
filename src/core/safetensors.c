/* Format: [8-byte header_size][JSON header][raw tensor data]
 * JSON header maps tensor names to {dtype, shape, data_offsets: [start, end]} */

#include "core/safetensors.h"
#include "core/serialization.h"
#include "core/logging.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TENSORS 4096

typedef struct SafeTensorInfo {
    char* name;
    char* dtype_str;
    int ndim;
    int shape[8];
    size_t data_start;
    size_t data_end;
} SafeTensorInfo;

struct SafeTensorsContext {
    FILE* file;
    char* filepath;
    bool is_write;
    int num_tensors;
    SafeTensorInfo* tensors;
    uint64_t header_size;
    // Write buffer
    int write_count;
    uint8_t* write_data;
    size_t write_data_size;
    size_t write_data_cap;
};

static const char* dtype_to_safetensor_str(DType dtype) {
    switch (dtype) {
        case DTYPE_FLOAT32: return "F32";
        case DTYPE_FLOAT64: return "F64";
        case DTYPE_FLOAT16: return "F16";
        case DTYPE_BFLOAT16: return "BF16";
        case DTYPE_INT8:    return "I8";
        case DTYPE_UINT8:   return "U8";
        case DTYPE_INT16:   return "I16";
        case DTYPE_UINT16:  return "U16";
        case DTYPE_INT32:   return "I32";
        case DTYPE_INT64:   return "I64";
        case DTYPE_UINT32:  return "U32";
        case DTYPE_UINT64:  return "U64";
        case DTYPE_BOOL:    return "BOOL";
        case DTYPE_FLOAT8_E4M3: return "F8_E4M3";
        case DTYPE_FLOAT8_E5M2: return "F8_E5M2";
        default: return "F32";
    }
}

static DType safetensor_str_to_dtype(const char* str) {
    if (!str) return DTYPE_FLOAT32;
    if (strcmp(str, "F32") == 0) return DTYPE_FLOAT32;
    if (strcmp(str, "F64") == 0) return DTYPE_FLOAT64;
    if (strcmp(str, "F16") == 0) return DTYPE_FLOAT16;
    if (strcmp(str, "BF16") == 0) return DTYPE_BFLOAT16;
    if (strcmp(str, "I8") == 0) return DTYPE_INT8;
    if (strcmp(str, "U8") == 0) return DTYPE_UINT8;
    if (strcmp(str, "I16") == 0) return DTYPE_INT16;
    if (strcmp(str, "U16") == 0) return DTYPE_UINT16;
    if (strcmp(str, "I32") == 0) return DTYPE_INT32;
    if (strcmp(str, "I64") == 0) return DTYPE_INT64;
    if (strcmp(str, "U32") == 0) return DTYPE_UINT32;
    if (strcmp(str, "U64") == 0) return DTYPE_UINT64;
    if (strcmp(str, "BOOL") == 0) return DTYPE_BOOL;
    if (strcmp(str, "F8_E4M3") == 0) return DTYPE_FLOAT8_E4M3;
    if (strcmp(str, "F8_E5M2") == 0) return DTYPE_FLOAT8_E5M2;
    return DTYPE_FLOAT32;
}

static char* skip_ws(char* p) {
    while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
    return p;
}

static char* parse_string(char* p, char** out) {
    if (*p != '"') return NULL;
    p++;
    char* start = p;
    while (*p && *p != '"') {
        if (*p == '\\') p++;
        p++;
    }
    size_t len = p - start;
    *out = malloc(len + 1);
    memcpy(*out, start, len);
    (*out)[len] = '\0';
    if (*p == '"') p++;
    return p;
}

static char* parse_int(char* p, int64_t* out) {
    *out = 0;
    int neg = 0;
    if (*p == '-') { neg = 1; p++; }
    while (*p >= '0' && *p <= '9') {
        *out = *out * 10 + (*p - '0');
        p++;
    }
    if (neg) *out = -*out;
    return p;
}

SafeTensorsContext* safetensors_open_read(const char* filepath) {
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        LOG_ERROR("safetensors_open_read: cannot open %s", filepath);
        return NULL;
    }

    uint64_t header_size;
    if (fread(&header_size, 8, 1, f) != 1) { fclose(f); return NULL; }
    if (header_size > 100 * 1024 * 1024) { fclose(f); return NULL; } // Sanity check

    char* header = malloc(header_size + 1);
    if (!header) { fclose(f); return NULL; }
    if (fread(header, 1, header_size, f) != header_size) { free(header); fclose(f); return NULL; }
    header[header_size] = '\0';

    SafeTensorsContext* ctx = calloc(1, sizeof(SafeTensorsContext));
    if (!ctx) { free(header); fclose(f); return NULL; }
    ctx->file = f;
    ctx->filepath = strdup(filepath);
    ctx->is_write = false;
    ctx->header_size = header_size;
    ctx->tensors = calloc(MAX_TENSORS, sizeof(SafeTensorInfo));
    if (!ctx->tensors) { free(header); safetensors_close(ctx); return NULL; }

    char* p = header;
    p = skip_ws(p);
    if (*p == '{') p++;

    while (*p && *p != '}' && ctx->num_tensors < MAX_TENSORS) {
        p = skip_ws(p);
        if (*p == '}') break;
        if (*p == ',') p++;
        p = skip_ws(p);

        char* name = NULL;
        p = parse_string(p, &name);
        if (!p || !name) break;

        if (strcmp(name, "__metadata__") == 0) {
            free(name);
            p = skip_ws(p);
            if (*p == ':') p++;
            int depth = 0;
            do {
                if (*p == '{') depth++;
                else if (*p == '}') depth--;
                p++;
            } while (*p && depth > 0);
            continue;
        }

        SafeTensorInfo* info = &ctx->tensors[ctx->num_tensors];
        info->name = name;

        p = skip_ws(p);
        if (*p == ':') p++;
        p = skip_ws(p);

        if (*p != '{') { free(name); break; }
        p++;

        while (*p && *p != '}') {
            p = skip_ws(p);
            if (*p == ',') p++;
            p = skip_ws(p);
            if (*p == '}') break;

            char* key = NULL;
            p = parse_string(p, &key);
            if (!p || !key) break;
            p = skip_ws(p);
            if (*p == ':') p++;
            p = skip_ws(p);

            if (strcmp(key, "dtype") == 0) {
                char* dtype_str = NULL;
                p = parse_string(p, &dtype_str);
                info->dtype_str = dtype_str;
            } else if (strcmp(key, "shape") == 0) {
                if (*p == '[') p++;
                info->ndim = 0;
                while (*p && *p != ']') {
                    p = skip_ws(p);
                    if (*p == ',') p++;
                    p = skip_ws(p);
                    if (*p == ']') break;
                    int64_t dim;
                    p = parse_int(p, &dim);
                    if (info->ndim < 8) info->shape[info->ndim++] = (int)dim;
                }
                if (*p == ']') p++;
            } else if (strcmp(key, "data_offsets") == 0) {
                if (*p == '[') p++;
                p = skip_ws(p);
                int64_t start, end;
                p = parse_int(p, &start);
                p = skip_ws(p);
                if (*p == ',') p++;
                p = skip_ws(p);
                p = parse_int(p, &end);
                p = skip_ws(p);
                if (*p == ']') p++;
                info->data_start = (size_t)start;
                info->data_end = (size_t)end;
            }
            free(key);
        }
        if (*p == '}') p++;
        ctx->num_tensors++;
    }

    free(header);
    return ctx;
}

SafeTensorsContext* safetensors_open_write(const char* filepath) {
    SafeTensorsContext* ctx = calloc(1, sizeof(SafeTensorsContext));
    if (!ctx) return NULL;
    ctx->filepath = strdup(filepath);
    ctx->is_write = true;
    ctx->tensors = calloc(MAX_TENSORS, sizeof(SafeTensorInfo));
    ctx->write_data_cap = 4096;
    ctx->write_data = malloc(ctx->write_data_cap);
    if (!ctx->tensors || !ctx->write_data) { safetensors_close(ctx); return NULL; }
    return ctx;
}

void safetensors_close(SafeTensorsContext* ctx) {
    if (!ctx) return;

    if (ctx->is_write && ctx->filepath && ctx->write_count > 0) {
        size_t json_cap = 4096;
        char* json = malloc(json_cap);
        if (json) {
            size_t pos = 0;
            json[pos++] = '{';

            for (int i = 0; i < ctx->write_count; i++) {
                SafeTensorInfo* info = &ctx->tensors[i];
                while (pos + 512 > json_cap) {
                    json_cap *= 2;
                    json = realloc(json, json_cap);
                }
                if (i > 0) json[pos++] = ',';
                pos += snprintf(json + pos, json_cap - pos,
                    "\"%s\":{\"dtype\":\"%s\",\"shape\":[",
                    info->name, info->dtype_str);
                for (int d = 0; d < info->ndim; d++) {
                    if (d > 0) json[pos++] = ',';
                    pos += snprintf(json + pos, json_cap - pos, "%d", info->shape[d]);
                }
                pos += snprintf(json + pos, json_cap - pos,
                    "],\"data_offsets\":[%zu,%zu]}",
                    info->data_start, info->data_end);
            }
            json[pos++] = '}';

            FILE* f = fopen(ctx->filepath, "wb");
            if (f) {
                uint64_t header_size = pos;
                fwrite(&header_size, 8, 1, f);
                fwrite(json, 1, pos, f);
                fwrite(ctx->write_data, 1, ctx->write_data_size, f);
                fclose(f);
            }
            free(json);
        }
    }

    if (ctx->file) fclose(ctx->file);
    if (ctx->tensors) {
        int count = ctx->is_write ? ctx->write_count : ctx->num_tensors;
        for (int i = 0; i < count; i++) {
            free(ctx->tensors[i].name);
            free(ctx->tensors[i].dtype_str);
        }
        free(ctx->tensors);
    }
    free(ctx->filepath);
    free(ctx->write_data);
    free(ctx);
}

int safetensors_get_num_tensors(SafeTensorsContext* ctx) {
    return ctx ? ctx->num_tensors : 0;
}

const char* safetensors_get_tensor_name(SafeTensorsContext* ctx, int index) {
    if (!ctx || index < 0 || index >= ctx->num_tensors) return NULL;
    return ctx->tensors[index].name;
}

Tensor* safetensors_read_tensor(SafeTensorsContext* ctx, const char* name) {
    if (!ctx || !name || ctx->is_write) return NULL;

    int idx = -1;
    for (int i = 0; i < ctx->num_tensors; i++) {
        if (ctx->tensors[i].name && strcmp(ctx->tensors[i].name, name) == 0) {
            idx = i; break;
        }
    }
    if (idx < 0) return NULL;

    SafeTensorInfo* info = &ctx->tensors[idx];
    DType dtype = safetensor_str_to_dtype(info->dtype_str);
    size_t data_size = info->data_end - info->data_start;

    TensorConfig config = {.dtype = dtype, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* t = tensor_empty(info->shape, info->ndim, &config);
    if (!t) return NULL;
    tensor_ensure_executed(t);

    long data_offset = 8 + (long)ctx->header_size + (long)info->data_start;
    fseek(ctx->file, data_offset, SEEK_SET);
    if (fread(t->data, 1, data_size, ctx->file) != data_size) {
        tensor_free(t);
        return NULL;
    }
    return t;
}

int safetensors_write_tensor(SafeTensorsContext* ctx, const char* name, Tensor* tensor) {
    if (!ctx || !name || !tensor || !ctx->is_write) return -1;
    if (ctx->write_count >= MAX_TENSORS) return -1;

    tensor_ensure_executed(tensor);
    if (!tensor->data) return -1;

    size_t data_size = tensor->numel * cml_dtype_size(tensor->dtype);

    while (ctx->write_data_size + data_size > ctx->write_data_cap) {
        ctx->write_data_cap *= 2;
        ctx->write_data = realloc(ctx->write_data, ctx->write_data_cap);
        if (!ctx->write_data) return -1;
    }

    SafeTensorInfo* info = &ctx->tensors[ctx->write_count];
    info->name = strdup(name);
    info->dtype_str = strdup(dtype_to_safetensor_str(tensor->dtype));
    info->ndim = tensor->ndim;
    for (int d = 0; d < tensor->ndim; d++) info->shape[d] = tensor->shape[d];
    info->data_start = ctx->write_data_size;
    info->data_end = ctx->write_data_size + data_size;

    memcpy(ctx->write_data + ctx->write_data_size, tensor->data, data_size);
    ctx->write_data_size += data_size;
    ctx->write_count++;
    return 0;
}

int module_save_safetensors(Module* module, const char* filepath) {
    if (!module || !filepath) return -1;

    NamedParameter* named_params = NULL;
    int num_params = 0;
    if (module_named_parameters(module, &named_params, &num_params) != 0) return -1;

    SafeTensorsContext* ctx = safetensors_open_write(filepath);
    if (!ctx) { module_named_parameters_free(named_params, num_params); return -1; }

    for (int i = 0; i < num_params; i++) {
        if (named_params[i].parameter && named_params[i].parameter->tensor) {
            safetensors_write_tensor(ctx, named_params[i].name, named_params[i].parameter->tensor);
        }
    }

    safetensors_close(ctx);
    module_named_parameters_free(named_params, num_params);
    return 0;
}

int module_load_safetensors(Module* module, const char* filepath) {
    if (!module || !filepath) return -1;

    SafeTensorsContext* ctx = safetensors_open_read(filepath);
    if (!ctx) return -1;

    NamedParameter* named_params = NULL;
    int num_params = 0;
    if (module_named_parameters(module, &named_params, &num_params) != 0) {
        safetensors_close(ctx);
        return -1;
    }

    for (int i = 0; i < num_params; i++) {
        Tensor* loaded = safetensors_read_tensor(ctx, named_params[i].name);
        if (loaded && named_params[i].parameter && named_params[i].parameter->tensor) {
            Tensor* target = named_params[i].parameter->tensor;
            tensor_ensure_executed(target);
            if (target->data && loaded->data && target->numel == loaded->numel) {
                memcpy(target->data, loaded->data, target->numel * cml_dtype_size(target->dtype));
            }
            tensor_free(loaded);
        }
    }

    safetensors_close(ctx);
    module_named_parameters_free(named_params, num_params);
    return 0;
}
