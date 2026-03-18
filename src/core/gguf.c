#include "core/gguf.h"
#include "core/gguf_quant.h"
#include "core/serialization.h"
#include "core/logging.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GGUF_VERSION 3
#define MAX_TENSORS 4096

typedef struct GGUFTensorInfo {
    char* name;
    int ndim;
    int64_t shape[8];
    GGUFTensorType type;
    uint64_t offset;
    size_t data_size;
} GGUFTensorInfo;

struct GGUFContext {
    FILE* file;
    char* filepath;
    bool is_write;
    int num_tensors;
    int num_metadata;
    GGUFTensorInfo* tensors;
    uint64_t data_offset;  // Start of tensor data section
    // Write buffer
    int write_count;
    uint8_t* write_data;
    size_t write_data_size;
    size_t write_data_cap;
};

static DType gguf_type_to_dtype(GGUFTensorType type) {
    switch (type) {
        case GGUF_TENSOR_F32: return DTYPE_FLOAT32;
        case GGUF_TENSOR_F16: return DTYPE_FLOAT16;
        case GGUF_TENSOR_I8:  return DTYPE_INT8;
        case GGUF_TENSOR_I16: return DTYPE_INT16;
        case GGUF_TENSOR_I32: return DTYPE_INT32;
        default: return DTYPE_FLOAT32;
    }
}

static GGUFTensorType dtype_to_gguf_type(DType dtype) {
    switch (dtype) {
        case DTYPE_FLOAT32: return GGUF_TENSOR_F32;
        case DTYPE_FLOAT16: return GGUF_TENSOR_F16;
        case DTYPE_INT8:    return GGUF_TENSOR_I8;
        case DTYPE_INT16:   return GGUF_TENSOR_I16;
        case DTYPE_INT32:   return GGUF_TENSOR_I32;
        default:            return GGUF_TENSOR_F32;
    }
}

static char* read_gguf_string(FILE* f) {
    uint64_t len;
    if (fread(&len, 8, 1, f) != 1) return NULL;
    if (len > 65536) return NULL;
    char* str = malloc(len + 1);
    if (!str) return NULL;
    if (fread(str, 1, len, f) != len) { free(str); return NULL; }
    str[len] = '\0';
    return str;
}

static void write_gguf_string(FILE* f, const char* str) {
    uint64_t len = strlen(str);
    fwrite(&len, 8, 1, f);
    fwrite(str, 1, len, f);
}

static void skip_gguf_value(FILE* f, uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:   fseek(f, 1, SEEK_CUR); break;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:  fseek(f, 2, SEEK_CUR); break;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32: fseek(f, 4, SEEK_CUR); break;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64: fseek(f, 8, SEEK_CUR); break;
        case GGUF_TYPE_STRING: {
            char* s = read_gguf_string(f);
            free(s);
            break;
        }
        case GGUF_TYPE_ARRAY: {
            uint32_t arr_type;
            uint64_t arr_len;
            if (fread(&arr_type, 4, 1, f) != 1) break;
            if (fread(&arr_len, 8, 1, f) != 1) break;
            for (uint64_t i = 0; i < arr_len; i++)
                skip_gguf_value(f, arr_type);
            break;
        }
    }
}

GGUFContext* gguf_open_read(const char* filepath) {
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        LOG_ERROR("gguf_open_read: cannot open %s", filepath);
        return NULL;
    }

    uint32_t magic;
    if (fread(&magic, 4, 1, f) != 1 || magic != GGUF_MAGIC) {
        LOG_ERROR("gguf_open_read: invalid GGUF magic");
        fclose(f);
        return NULL;
    }

    uint32_t version;
    if (fread(&version, 4, 1, f) != 1) { fclose(f); return NULL; }

    uint64_t num_tensors, num_metadata;
    if (fread(&num_tensors, 8, 1, f) != 1) { fclose(f); return NULL; }
    if (fread(&num_metadata, 8, 1, f) != 1) { fclose(f); return NULL; }

    GGUFContext* ctx = calloc(1, sizeof(GGUFContext));
    if (!ctx) { fclose(f); return NULL; }
    ctx->file = f;
    ctx->filepath = strdup(filepath);
    ctx->is_write = false;
    ctx->num_tensors = (int)num_tensors;
    ctx->num_metadata = (int)num_metadata;

    for (uint64_t i = 0; i < num_metadata; i++) {
        char* key = read_gguf_string(f);
        free(key);
        uint32_t val_type;
        if (fread(&val_type, 4, 1, f) != 1) break;
        skip_gguf_value(f, val_type);
    }

    ctx->tensors = calloc(num_tensors, sizeof(GGUFTensorInfo));
    if (!ctx->tensors) { gguf_close(ctx); return NULL; }

    for (uint64_t i = 0; i < num_tensors; i++) {
        ctx->tensors[i].name = read_gguf_string(f);
        uint32_t ndim;
        if (fread(&ndim, 4, 1, f) != 1) break;
        ctx->tensors[i].ndim = (int)ndim;
        for (uint32_t d = 0; d < ndim; d++) {
            uint64_t dim;
            if (fread(&dim, 8, 1, f) != 1) break;
            ctx->tensors[i].shape[d] = (int64_t)dim;
        }
        uint32_t type;
        if (fread(&type, 4, 1, f) != 1) break;
        ctx->tensors[i].type = (GGUFTensorType)type;
        if (fread(&ctx->tensors[i].offset, 8, 1, f) != 1) break;

        size_t numel = 1;
        for (int d = 0; d < ctx->tensors[i].ndim; d++)
            numel *= (size_t)ctx->tensors[i].shape[d];
        if (gguf_type_is_quantized(ctx->tensors[i].type)) {
            int block_size = gguf_quant_block_size(ctx->tensors[i].type);
            size_t type_size = gguf_quant_type_size(ctx->tensors[i].type);
            size_t num_blocks = (numel + (size_t)block_size - 1) / (size_t)block_size;
            ctx->tensors[i].data_size = num_blocks * type_size;
        } else {
            ctx->tensors[i].data_size = numel * cml_dtype_size(gguf_type_to_dtype(ctx->tensors[i].type));
        }
    }

    // Align to 32 bytes for tensor data
    long pos = ftell(f);
    long aligned = (pos + 31) & ~31L;
    ctx->data_offset = (uint64_t)aligned;

    return ctx;
}

GGUFContext* gguf_open_write(const char* filepath) {
    GGUFContext* ctx = calloc(1, sizeof(GGUFContext));
    if (!ctx) return NULL;
    ctx->filepath = strdup(filepath);
    ctx->is_write = true;
    ctx->tensors = calloc(MAX_TENSORS, sizeof(GGUFTensorInfo));
    ctx->write_data_cap = 4096;
    ctx->write_data = malloc(ctx->write_data_cap);
    if (!ctx->tensors || !ctx->write_data) { gguf_close(ctx); return NULL; }
    return ctx;
}

void gguf_close(GGUFContext* ctx) {
    if (!ctx) return;

    if (ctx->is_write && ctx->filepath && ctx->write_count > 0) {
        FILE* f = fopen(ctx->filepath, "wb");
        if (f) {
            uint32_t magic = GGUF_MAGIC;
            uint32_t version = GGUF_VERSION;
            uint64_t nt = ctx->write_count;
            uint64_t nm = 0;
            fwrite(&magic, 4, 1, f);
            fwrite(&version, 4, 1, f);
            fwrite(&nt, 8, 1, f);
            fwrite(&nm, 8, 1, f);

            for (int i = 0; i < ctx->write_count; i++) {
                write_gguf_string(f, ctx->tensors[i].name);
                uint32_t ndim = (uint32_t)ctx->tensors[i].ndim;
                fwrite(&ndim, 4, 1, f);
                for (int d = 0; d < ctx->tensors[i].ndim; d++) {
                    uint64_t dim = (uint64_t)ctx->tensors[i].shape[d];
                    fwrite(&dim, 8, 1, f);
                }
                uint32_t type = (uint32_t)ctx->tensors[i].type;
                fwrite(&type, 4, 1, f);
                fwrite(&ctx->tensors[i].offset, 8, 1, f);
            }

            // Align to 32 bytes
            long pos = ftell(f);
            long aligned = (pos + 31) & ~31L;
            while (pos < aligned) { uint8_t z = 0; fwrite(&z, 1, 1, f); pos++; }

            fwrite(ctx->write_data, 1, ctx->write_data_size, f);
            fclose(f);
        }
    }

    if (ctx->file) fclose(ctx->file);
    if (ctx->tensors) {
        for (int i = 0; i < ctx->num_tensors || i < ctx->write_count; i++) {
            free(ctx->tensors[i].name);
        }
        free(ctx->tensors);
    }
    free(ctx->filepath);
    free(ctx->write_data);
    free(ctx);
}

int gguf_get_num_tensors(GGUFContext* ctx) {
    return ctx ? ctx->num_tensors : 0;
}

const char* gguf_get_tensor_name(GGUFContext* ctx, int index) {
    if (!ctx || index < 0 || index >= ctx->num_tensors) return NULL;
    return ctx->tensors[index].name;
}

Tensor* gguf_read_tensor(GGUFContext* ctx, const char* name) {
    if (!ctx || !name || ctx->is_write) return NULL;

    int idx = -1;
    for (int i = 0; i < ctx->num_tensors; i++) {
        if (ctx->tensors[i].name && strcmp(ctx->tensors[i].name, name) == 0) {
            idx = i; break;
        }
    }
    if (idx < 0) return NULL;

    GGUFTensorInfo* info = &ctx->tensors[idx];

    int shape[8];
    for (int d = 0; d < info->ndim; d++) shape[d] = (int)info->shape[d];

    size_t numel = 1;
    for (int d = 0; d < info->ndim; d++) numel *= (size_t)info->shape[d];

    if (gguf_type_is_quantized(info->type)) {
        /* Quantized: read raw block data, dequantize to float32 */
        void* raw = malloc(info->data_size);
        if (!raw) return NULL;

        fseek(ctx->file, (long)(ctx->data_offset + info->offset), SEEK_SET);
        if (fread(raw, 1, info->data_size, ctx->file) != info->data_size) {
            free(raw);
            return NULL;
        }

        TensorConfig config = {.dtype = DTYPE_FLOAT32, .device = DEVICE_CPU,
                               .has_dtype = true, .has_device = true};
        Tensor* t = tensor_empty(shape, info->ndim, &config);
        if (!t) { free(raw); return NULL; }
        tensor_ensure_executed(t);

        if (gguf_dequantize(info->type, raw, (float*)t->data, numel) != 0) {
            free(raw);
            tensor_free(t);
            return NULL;
        }
        free(raw);
        return t;
    }

    /* Non-quantized: read directly */
    DType dtype = gguf_type_to_dtype(info->type);
    TensorConfig config = {.dtype = dtype, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* t = tensor_empty(shape, info->ndim, &config);
    if (!t) return NULL;
    tensor_ensure_executed(t);

    fseek(ctx->file, (long)(ctx->data_offset + info->offset), SEEK_SET);
    if (fread(t->data, 1, info->data_size, ctx->file) != info->data_size) {
        tensor_free(t);
        return NULL;
    }
    return t;
}

int gguf_write_tensor(GGUFContext* ctx, const char* name, Tensor* tensor) {
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

    GGUFTensorInfo* info = &ctx->tensors[ctx->write_count];
    info->name = strdup(name);
    info->ndim = tensor->ndim;
    for (int d = 0; d < tensor->ndim; d++) info->shape[d] = tensor->shape[d];
    info->type = dtype_to_gguf_type(tensor->dtype);
    info->offset = ctx->write_data_size;
    info->data_size = data_size;

    memcpy(ctx->write_data + ctx->write_data_size, tensor->data, data_size);
    ctx->write_data_size += data_size;
    ctx->write_count++;
    return 0;
}

int module_save_gguf(Module* module, const char* filepath) {
    if (!module || !filepath) return -1;

    NamedParameter* named_params = NULL;
    int num_params = 0;
    if (module_named_parameters(module, &named_params, &num_params) != 0) return -1;

    GGUFContext* ctx = gguf_open_write(filepath);
    if (!ctx) { module_named_parameters_free(named_params, num_params); return -1; }

    for (int i = 0; i < num_params; i++) {
        if (named_params[i].parameter && named_params[i].parameter->tensor) {
            gguf_write_tensor(ctx, named_params[i].name, named_params[i].parameter->tensor);
        }
    }

    gguf_close(ctx);
    module_named_parameters_free(named_params, num_params);
    return 0;
}

int module_load_gguf(Module* module, const char* filepath) {
    if (!module || !filepath) return -1;

    GGUFContext* ctx = gguf_open_read(filepath);
    if (!ctx) return -1;

    NamedParameter* named_params = NULL;
    int num_params = 0;
    if (module_named_parameters(module, &named_params, &num_params) != 0) {
        gguf_close(ctx);
        return -1;
    }

    for (int i = 0; i < num_params; i++) {
        Tensor* loaded = gguf_read_tensor(ctx, named_params[i].name);
        if (loaded && named_params[i].parameter && named_params[i].parameter->tensor) {
            Tensor* target = named_params[i].parameter->tensor;
            tensor_ensure_executed(target);
            if (target->data && loaded->data && target->numel == loaded->numel) {
                memcpy(target->data, loaded->data, target->numel * cml_dtype_size(target->dtype));
            }
            tensor_free(loaded);
        }
    }

    gguf_close(ctx);
    module_named_parameters_free(named_params, num_params);
    return 0;
}
