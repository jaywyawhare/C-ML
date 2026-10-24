#include "core/gguf.h"
#include "core/gguf_quant.h"
#include "core/serialization.h"
#include "core/logging.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "alloc/cml_allocator.h"

#define GGUF_VERSION 3
#define MAX_TENSORS 4096
#define GGUF_MAX_DIMS 8
#define GGUF_MAX_ARRAY_DEPTH 8
#define GGUF_DEFAULT_ALIGNMENT 32

typedef struct GGUFTensorInfo {
    char* name;
    int ndim;
    int shape[GGUF_MAX_DIMS];
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
    uint64_t data_offset; // Start of tensor data section
    uint32_t alignment;   // Tensor data alignment (general.alignment)
    // Tokenizer metadata (tokenizer.ggml.tokens / .merges), owned by ctx
    char** tok_tokens;
    int num_tok_tokens;
    char** tok_merges;
    int num_tok_merges;
    // Write buffer
    int write_count;
    uint8_t* write_data;
    size_t write_data_size;
    size_t write_data_cap;
};

/** Plain (non-quantized) GGUF tensor types and the DType each one stores. */
static const struct {
    GGUFTensorType gguf;
    DType dtype;
} GGUF_PLAIN_TYPES[] = {
    {GGUF_TENSOR_F32, DTYPE_FLOAT32},   {GGUF_TENSOR_F16, DTYPE_FLOAT16},
    {GGUF_TENSOR_BF16, DTYPE_BFLOAT16}, {GGUF_TENSOR_F64, DTYPE_FLOAT64},
    {GGUF_TENSOR_I8, DTYPE_INT8},       {GGUF_TENSOR_I16, DTYPE_INT16},
    {GGUF_TENSOR_I32, DTYPE_INT32},     {GGUF_TENSOR_I64, DTYPE_INT64},
};
#define GGUF_NUM_PLAIN_TYPES (sizeof(GGUF_PLAIN_TYPES) / sizeof(GGUF_PLAIN_TYPES[0]))

/** DType stored by a plain GGUF tensor type; false for quantized or unsupported types. */
static bool gguf_type_to_dtype(GGUFTensorType type, DType* out) {
    for (size_t i = 0; i < GGUF_NUM_PLAIN_TYPES; i++) {
        if (GGUF_PLAIN_TYPES[i].gguf == type) {
            *out = GGUF_PLAIN_TYPES[i].dtype;
            return true;
        }
    }
    return false;
}

/** GGUF tensor type that stores `dtype`; false when GGUF has no plain encoding for it. */
static bool dtype_to_gguf_type(DType dtype, GGUFTensorType* out) {
    for (size_t i = 0; i < GGUF_NUM_PLAIN_TYPES; i++) {
        if (GGUF_PLAIN_TYPES[i].dtype == dtype) {
            *out = GGUF_PLAIN_TYPES[i].gguf;
            return true;
        }
    }
    return false;
}

/** Read a GGUF length-prefixed string (u64 len + bytes) into a NUL-terminated buffer. */
static char* read_gguf_string(FILE* f) {
    uint64_t len;
    if (fread(&len, 8, 1, f) != 1)
        return NULL;
    if (len > 65536)
        return NULL;
    char* str = cml_malloc(len + 1);
    if (!str)
        return NULL;
    if (fread(str, 1, len, f) != len) {
        cml_free(str);
        return NULL;
    }
    str[len] = '\0';
    return str;
}

/** Write a GGUF length-prefixed string (u64 len + bytes). */
static void write_gguf_string(FILE* f, const char* str) {
    uint64_t len = strlen(str);
    fwrite(&len, 8, 1, f);
    fwrite(str, 1, len, f);
}

/** Byte width of a fixed-size GGUF metadata type; 0 for strings, arrays and unknown types. */
static size_t gguf_scalar_size(uint32_t type) {
    switch (type) {
    case GGUF_TYPE_UINT8:
    case GGUF_TYPE_INT8:
    case GGUF_TYPE_BOOL:
        return 1;
    case GGUF_TYPE_UINT16:
    case GGUF_TYPE_INT16:
        return 2;
    case GGUF_TYPE_UINT32:
    case GGUF_TYPE_INT32:
    case GGUF_TYPE_FLOAT32:
        return 4;
    case GGUF_TYPE_UINT64:
    case GGUF_TYPE_INT64:
    case GGUF_TYPE_FLOAT64:
        return 8;
    default:
        return 0;
    }
}

static bool skip_gguf_value(FILE* f, uint32_t type, int depth);

/** Seek past the `arr_len` elements of a GGUF array whose header was already read. Fixed-size
 *  elements are skipped in one seek, so a hostile length cannot spin. False on failure. */
static bool skip_gguf_array(FILE* f, uint32_t arr_type, uint64_t arr_len, int depth) {
    size_t width = gguf_scalar_size(arr_type);
    if (width)
        return arr_len <= LONG_MAX / width && fseek(f, (long)(arr_len * width), SEEK_CUR) == 0;
    for (uint64_t i = 0; i < arr_len; i++)
        if (!skip_gguf_value(f, arr_type, depth + 1))
            return false;
    return true;
}

/** Seek past a GGUF metadata value of the given type, with array nesting capped at
 *  GGUF_MAX_ARRAY_DEPTH. Returns false on a truncated file or an invalid value. */
static bool skip_gguf_value(FILE* f, uint32_t type, int depth) {
    size_t width = gguf_scalar_size(type);
    if (width)
        return fseek(f, (long)width, SEEK_CUR) == 0;
    if (type == GGUF_TYPE_STRING) {
        uint64_t len;
        return fread(&len, 8, 1, f) == 1 && len <= LONG_MAX && fseek(f, (long)len, SEEK_CUR) == 0;
    }
    if (type != GGUF_TYPE_ARRAY || depth >= GGUF_MAX_ARRAY_DEPTH)
        return false;
    uint32_t arr_type;
    uint64_t arr_len;
    return fread(&arr_type, 4, 1, f) == 1 && fread(&arr_len, 8, 1, f) == 1 &&
           skip_gguf_array(f, arr_type, arr_len, depth);
}

/** Free a string list of `n` entries (NULL-safe). */
static void free_string_list(char** items, int n) {
    if (!items)
        return;
    for (int i = 0; i < n; i++)
        cml_free(items[i]);
    cml_free(items);
}

/**
 * Open a GGUF file for reading: validates the magic/version, captures the tokenizer
 * vocab and BPE merges from the metadata (skipping everything else), and records each
 * tensor's shape/type/offset/size. The data section is 32-byte aligned.
 */
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
    if (fread(&version, 4, 1, f) != 1) {
        fclose(f);
        return NULL;
    }

    uint64_t num_tensors, num_metadata;
    if (fread(&num_tensors, 8, 1, f) != 1) {
        fclose(f);
        return NULL;
    }
    if (fread(&num_metadata, 8, 1, f) != 1) {
        fclose(f);
        return NULL;
    }

    if (num_tensors > INT_MAX || num_metadata > INT_MAX) {
        LOG_ERROR("gguf_open_read: implausible tensor/metadata counts in %s", filepath);
        fclose(f);
        return NULL;
    }

    GGUFContext* ctx = cml_calloc(1, sizeof(GGUFContext));
    if (!ctx) {
        fclose(f);
        return NULL;
    }
    ctx->file         = f;
    ctx->filepath     = cml_strdup(filepath);
    ctx->is_write     = false;
    ctx->num_tensors  = (int)num_tensors;
    ctx->num_metadata = (int)num_metadata;
    ctx->alignment    = GGUF_DEFAULT_ALIGNMENT;

    for (uint64_t i = 0; i < num_metadata; i++) {
        char* key = read_gguf_string(f);
        uint32_t val_type;
        if (fread(&val_type, 4, 1, f) != 1) {
            cml_free(key);
            break;
        }

        /* Capture the tokenizer vocab and BPE merge list; everything else is
         * skipped as before. */
        bool is_tokens = key && strcmp(key, "tokenizer.ggml.tokens") == 0;
        bool is_merges = key && strcmp(key, "tokenizer.ggml.merges") == 0;
        bool is_align  = key && strcmp(key, "general.alignment") == 0;
        cml_free(key);
        if (is_align && val_type == GGUF_TYPE_UINT32) {
            uint32_t align;
            if (fread(&align, 4, 1, f) != 1)
                break;
            if (align && (align & (align - 1)) == 0)
                ctx->alignment = align;
        } else if ((is_tokens || is_merges) && val_type == GGUF_TYPE_ARRAY) {
            uint32_t arr_type;
            uint64_t arr_len;
            if (fread(&arr_type, 4, 1, f) != 1 || fread(&arr_len, 8, 1, f) != 1)
                break;
            if (arr_type == GGUF_TYPE_STRING && arr_len > 0 && arr_len < 10000000ULL) {
                char** items = cml_calloc((size_t)arr_len, sizeof(char*));
                if (!items)
                    break;
                for (uint64_t k = 0; k < arr_len; k++) {
                    items[k] = read_gguf_string(f);
                    if (!items[k])
                        items[k] = cml_strdup("");
                }
                if (is_tokens) {
                    free_string_list(ctx->tok_tokens, ctx->num_tok_tokens);
                    ctx->tok_tokens     = items;
                    ctx->num_tok_tokens = (int)arr_len;
                } else {
                    free_string_list(ctx->tok_merges, ctx->num_tok_merges);
                    ctx->tok_merges     = items;
                    ctx->num_tok_merges = (int)arr_len;
                }
            } else if (!skip_gguf_array(f, arr_type, arr_len, 0)) {
                break;
            }
        } else if (!skip_gguf_value(f, val_type, 0)) {
            break;
        }
    }

    ctx->tensors = cml_calloc(num_tensors, sizeof(GGUFTensorInfo));
    if (!ctx->tensors) {
        gguf_close(ctx);
        return NULL;
    }

    for (uint64_t i = 0; i < num_tensors; i++) {
        GGUFTensorInfo* info = &ctx->tensors[i];
        info->name           = read_gguf_string(f);
        uint32_t ndim;
        if (fread(&ndim, 4, 1, f) != 1)
            break;
        if (ndim > GGUF_MAX_DIMS)
            goto malformed;
        info->ndim = (int)ndim;
        for (uint32_t d = 0; d < ndim; d++) {
            uint64_t dim;
            if (fread(&dim, 8, 1, f) != 1)
                break;
            if (dim > INT_MAX)
                goto malformed;
            info->shape[ndim - 1 - d] = (int)dim;
        }
        uint32_t type;
        if (fread(&type, 4, 1, f) != 1)
            break;
        info->type = (GGUFTensorType)type;
        if (fread(&info->offset, 8, 1, f) != 1)
            break;

        size_t numel;
        DType dtype;
        if (!tensor_numel_checked(info->shape, info->ndim, &numel))
            goto malformed;
        if (gguf_type_is_quantized(info->type)) {
            size_t block_size = (size_t)gguf_quant_block_size(info->type);
            size_t num_blocks = numel / block_size + (numel % block_size != 0);
            if (__builtin_mul_overflow(num_blocks, gguf_quant_type_size(info->type),
                                       &info->data_size))
                goto malformed;
        } else if (gguf_type_to_dtype(info->type, &dtype) &&
                   !tensor_nbytes_checked(numel, dtype, &info->data_size)) {
            goto malformed;
        }
    }

    long pos         = ftell(f);
    ctx->data_offset = ((uint64_t)pos + ctx->alignment - 1) & ~(uint64_t)(ctx->alignment - 1);

    return ctx;

malformed:
    LOG_ERROR("gguf_open_read: malformed tensor info in %s", filepath);
    gguf_close(ctx);
    return NULL;
}

/** Create a write context that buffers tensor data in memory until close. */
GGUFContext* gguf_open_write(const char* filepath) {
    GGUFContext* ctx = cml_calloc(1, sizeof(GGUFContext));
    if (!ctx)
        return NULL;
    ctx->filepath       = cml_strdup(filepath);
    ctx->is_write       = true;
    ctx->tensors        = cml_calloc(MAX_TENSORS, sizeof(GGUFTensorInfo));
    ctx->write_data_cap = 4096;
    ctx->write_data     = cml_malloc(ctx->write_data_cap);
    if (!ctx->tensors || !ctx->write_data) {
        gguf_close(ctx);
        return NULL;
    }
    return ctx;
}

/** Close the context. For write contexts, emits the header, tensor infos, 32-byte
 *  alignment padding, and buffered data. Frees all owned resources. */
void gguf_close(GGUFContext* ctx) {
    if (!ctx)
        return;

    if (ctx->is_write && ctx->filepath && ctx->write_count > 0) {
        FILE* f = fopen(ctx->filepath, "wb");
        if (f) {
            uint32_t magic   = GGUF_MAGIC;
            uint32_t version = GGUF_VERSION;
            uint64_t nt      = ctx->write_count;
            uint64_t nm      = 0;
            fwrite(&magic, 4, 1, f);
            fwrite(&version, 4, 1, f);
            fwrite(&nt, 8, 1, f);
            fwrite(&nm, 8, 1, f);

            for (int i = 0; i < ctx->write_count; i++) {
                write_gguf_string(f, ctx->tensors[i].name);
                uint32_t ndim = (uint32_t)ctx->tensors[i].ndim;
                fwrite(&ndim, 4, 1, f);
                for (int d = ctx->tensors[i].ndim - 1; d >= 0; d--) {
                    uint64_t dim = (uint64_t)ctx->tensors[i].shape[d];
                    fwrite(&dim, 8, 1, f);
                }
                uint32_t type = (uint32_t)ctx->tensors[i].type;
                fwrite(&type, 4, 1, f);
                fwrite(&ctx->tensors[i].offset, 8, 1, f);
            }

            long pos     = ftell(f);
            long aligned = (pos + GGUF_DEFAULT_ALIGNMENT - 1) & ~(long)(GGUF_DEFAULT_ALIGNMENT - 1);
            while (pos < aligned) {
                uint8_t z = 0;
                fwrite(&z, 1, 1, f);
                pos++;
            }

            fwrite(ctx->write_data, 1, ctx->write_data_size, f);
            fclose(f);
        }
    }

    if (ctx->file)
        fclose(ctx->file);
    if (ctx->tensors) {
        for (int i = 0; i < ctx->num_tensors || i < ctx->write_count; i++) {
            cml_free(ctx->tensors[i].name);
        }
        cml_free(ctx->tensors);
    }
    free_string_list(ctx->tok_tokens, ctx->num_tok_tokens);
    free_string_list(ctx->tok_merges, ctx->num_tok_merges);
    cml_free(ctx->filepath);
    cml_free(ctx->write_data);
    cml_free(ctx);
}

/** Hand back the context-owned tokenizer vocab and merge lists (borrowed, not copied);
 *  -1 if no tokenizer metadata was present. */
int gguf_get_tokenizer(GGUFContext* ctx, char*** tokens_out, int* num_tokens_out,
                       char*** merges_out, int* num_merges_out) {
    if (tokens_out)
        *tokens_out = NULL;
    if (num_tokens_out)
        *num_tokens_out = 0;
    if (merges_out)
        *merges_out = NULL;
    if (num_merges_out)
        *num_merges_out = 0;
    if (!ctx || !ctx->tok_tokens || ctx->num_tok_tokens <= 0)
        return -1;
    if (tokens_out)
        *tokens_out = ctx->tok_tokens;
    if (num_tokens_out)
        *num_tokens_out = ctx->num_tok_tokens;
    if (merges_out)
        *merges_out = ctx->tok_merges;
    if (num_merges_out)
        *num_merges_out = ctx->num_tok_merges;
    return 0;
}

/** Number of tensors in an opened-for-read file. */
int gguf_get_num_tensors(GGUFContext* ctx) { return ctx ? ctx->num_tensors : 0; }

/** Name of the tensor at the given index, or NULL if out of range. */
const char* gguf_get_tensor_name(GGUFContext* ctx, int index) {
    if (!ctx || index < 0 || index >= ctx->num_tensors)
        return NULL;
    return ctx->tensors[index].name;
}

/** Read a named tensor. Quantized tensors are dequantized to float32 while the raw
 *  blocks are retained on the tensor (quant_data) for requantized compute; NULL if absent. */
Tensor* gguf_read_tensor(GGUFContext* ctx, const char* name) {
    if (!ctx || !name || ctx->is_write)
        return NULL;

    int idx = -1;
    for (int i = 0; i < ctx->num_tensors; i++) {
        if (ctx->tensors[i].name && strcmp(ctx->tensors[i].name, name) == 0) {
            idx = i;
            break;
        }
    }
    if (idx < 0)
        return NULL;

    GGUFTensorInfo* info = &ctx->tensors[idx];
    if (info->offset > (uint64_t)LONG_MAX - ctx->data_offset)
        return NULL;
    long data_pos = (long)(ctx->data_offset + info->offset);

    if (gguf_type_is_quantized(info->type)) {
        /* Quantized: read raw block data, dequantize to float32 */
        void* raw = cml_malloc(info->data_size);
        if (!raw)
            return NULL;

        fseek(ctx->file, data_pos, SEEK_SET);
        if (fread(raw, 1, info->data_size, ctx->file) != info->data_size) {
            cml_free(raw);
            return NULL;
        }

        TensorConfig config = {
            .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
        Tensor* t = tensor_empty(info->shape, info->ndim, &config);
        if (!t) {
            cml_free(raw);
            return NULL;
        }
        tensor_ensure_executed(t);

        if (gguf_dequantize(info->type, raw, (float*)t->data, t->numel) != 0) {
            cml_free(raw);
            tensor_free(t);
            return NULL;
        }

        if (info->type == GGUF_TENSOR_Q8_0) {
            t->quant_type = CML_QUANT_GGUF_Q8_0;
        } else if (info->type == GGUF_TENSOR_Q4_0) {
            t->quant_type = CML_QUANT_GGUF_Q4_0;
        }
        t->quant_data       = raw;
        t->quant_data_bytes = info->data_size;
        return t;
    }

    DType dtype;
    if (!gguf_type_to_dtype(info->type, &dtype)) {
        LOG_ERROR("gguf_read_tensor: '%s' has unsupported GGUF tensor type %d", name,
                  (int)info->type);
        return NULL;
    }
    TensorConfig config = {
        .dtype = dtype, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* t = tensor_empty(info->shape, info->ndim, &config);
    if (!t)
        return NULL;
    tensor_ensure_executed(t);

    fseek(ctx->file, data_pos, SEEK_SET);
    if (fread(t->data, 1, info->data_size, ctx->file) != info->data_size) {
        tensor_free(t);
        return NULL;
    }
    return t;
}

/** Append a tensor's raw bytes to the write buffer, recording its metadata; -1 on error. */
int gguf_write_tensor(GGUFContext* ctx, const char* name, Tensor* tensor) {
    if (!ctx || !name || !tensor || !ctx->is_write)
        return -1;
    if (ctx->write_count >= MAX_TENSORS)
        return -1;

    GGUFTensorType type;
    if (!dtype_to_gguf_type(tensor->dtype, &type) || tensor->ndim > GGUF_MAX_DIMS) {
        LOG_ERROR("gguf_write_tensor: '%s' (dtype %d, %d dims) has no GGUF encoding", name,
                  (int)tensor->dtype, tensor->ndim);
        return -1;
    }

    tensor_ensure_executed(tensor);
    if (!tensor->data)
        return -1;

    size_t data_size = tensor->numel * cml_dtype_size(tensor->dtype);
    size_t offset =
        (ctx->write_data_size + GGUF_DEFAULT_ALIGNMENT - 1) & ~(size_t)(GGUF_DEFAULT_ALIGNMENT - 1);

    while (offset + data_size > ctx->write_data_cap) {
        uint8_t* grown = cml_realloc(ctx->write_data, ctx->write_data_cap * 2);
        if (!grown)
            return -1;
        ctx->write_data = grown;
        ctx->write_data_cap *= 2;
    }

    GGUFTensorInfo* info = &ctx->tensors[ctx->write_count];
    info->name           = cml_strdup(name);
    info->ndim           = tensor->ndim;
    for (int d = 0; d < tensor->ndim; d++)
        info->shape[d] = tensor->shape[d];
    info->type      = type;
    info->offset    = offset;
    info->data_size = data_size;

    memset(ctx->write_data + ctx->write_data_size, 0, offset - ctx->write_data_size);
    memcpy(ctx->write_data + offset, tensor->data, data_size);
    ctx->write_data_size = offset + data_size;
    ctx->write_count++;
    return 0;
}

/** Save all of a module's named parameters to a GGUF file. */
int module_save_gguf(Module* module, const char* filepath) {
    if (!module || !filepath)
        return -1;

    NamedParameter* named_params = NULL;
    int num_params               = 0;
    if (module_named_parameters(module, &named_params, &num_params) != 0)
        return -1;

    GGUFContext* ctx = gguf_open_write(filepath);
    if (!ctx) {
        module_named_parameters_free(named_params, num_params);
        return -1;
    }

    for (int i = 0; i < num_params; i++) {
        if (named_params[i].parameter && named_params[i].parameter->tensor) {
            gguf_write_tensor(ctx, named_params[i].name, named_params[i].parameter->tensor);
        }
    }

    gguf_close(ctx);
    module_named_parameters_free(named_params, num_params);
    return 0;
}

/** Load a module's parameters from a GGUF file, copying only where element counts match. */
int module_load_gguf(Module* module, const char* filepath) {
    if (!module || !filepath)
        return -1;

    GGUFContext* ctx = gguf_open_read(filepath);
    if (!ctx)
        return -1;

    NamedParameter* named_params = NULL;
    int num_params               = 0;
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
