#include "nn/model_io.h"
#include "nn.h"
#include "optim.h"
#include "tensor/tensor.h"
#include "core/logging.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "alloc/cml_allocator.h"

#define CML_MODEL_MAGIC "CML\0"
#define CML_CKPT_MAGIC "CKP\0"
#define CML_MODEL_VERSION 1

/* Sanity caps for file-supplied counts: every length/rank/count below comes
 * straight off disk, so each read is bounded before it drives an allocation
 * or a loop. */
#define CML_IO_MAX_NAME_LEN 4096
#define CML_IO_MAX_NDIM 16
#define CML_IO_MAX_PARAMS (1 << 20)

/* Checked stdio wrappers: a short read/write or stream error fails loudly
 * instead of silently corrupting the file or continuing past garbage. */
static bool io_write(FILE* f, const void* buf, size_t n) {
    return n == 0 || (fwrite(buf, 1, n, f) == n && !ferror(f));
}

static bool io_read(FILE* f, void* buf, size_t n) {
    return n == 0 || (fread(buf, 1, n, f) == n && !ferror(f));
}

static bool io_skip(FILE* f, uint64_t n) {
    /* Bounded skip: refuse to seek past what could plausibly be in the file
     * (a corrupt size field would otherwise "skip" into next century). */
    long cur = ftell(f);
    if (cur < 0)
        return false;
    fseek(f, 0, SEEK_END);
    long end = ftell(f);
    if (end < 0 || (uint64_t)(end - cur) < n) {
        fseek(f, cur, SEEK_SET);
        return false;
    }
    fseek(f, cur + (long)n, SEEK_SET);
    return true;
}

/* Serialize a module's parameters to an open file: count, then each param's
 * name, dtype, shape and raw data. */
static bool write_model_params(FILE* f, Module* model) {
    int32_t num_params = model->num_parameters;
    if (!io_write(f, &num_params, sizeof(num_params)))
        return false;

    for (int i = 0; i < model->num_parameters; i++) {
        Parameter* param = model->parameters[i];
        if (!param || !param->tensor)
            continue;

        Tensor* t = param->tensor;
        tensor_ensure_executed(t);
        const char* name   = param->name ? param->name : "";
        int32_t name_len   = (int32_t)strlen(name);
        int32_t dtype      = (int32_t)t->dtype;
        int32_t ndim       = t->ndim;
        size_t elem_size   = cml_dtype_size(t->dtype);
        uint64_t data_size = (uint64_t)(t->numel * elem_size);
        if (!io_write(f, &name_len, sizeof(name_len)) || !io_write(f, name, (size_t)name_len) ||
            !io_write(f, &dtype, sizeof(dtype)) || !io_write(f, &ndim, sizeof(ndim)))
            return false;
        for (int d = 0; d < t->ndim; d++) {
            int32_t dim = t->shape[d];
            if (!io_write(f, &dim, sizeof(dim)))
                return false;
        }
        if (!io_write(f, &data_size, sizeof(data_size)))
            return false;
        void* data = tensor_data_ptr(t);
        if (data && !io_write(f, data, (size_t)data_size))
            return false;
    }
    return true;
}

int model_save(Module* model, const char* filepath) {
    if (!model || !filepath)
        return -1;

    FILE* f = fopen(filepath, "wb");
    if (!f) {
        LOG_ERROR("Failed to open file for writing: %s", filepath);
        return -1;
    }
    bool ok          = io_write(f, "CML", 4); // magic (includes null terminator)
    uint32_t version = CML_MODEL_VERSION;
    ok               = ok && io_write(f, &version, sizeof(version));
    ok               = ok && write_model_params(f, model);

    if (fclose(f) != 0)
        ok = false;
    if (!ok) {
        LOG_ERROR("Failed to write model file: %s", filepath);
        return -1;
    }
    return 0;
}

/* Read one serialized parameter's header -- length-prefixed name, dtype and
 * rank -- returning the freshly allocated name, or NULL on malformed input or
 * allocation failure. */
static char* read_param_header(FILE* f, int32_t* dtype, int32_t* ndim) {
    int32_t name_len;
    if (!io_read(f, &name_len, sizeof(name_len)) || name_len < 0 || name_len > CML_IO_MAX_NAME_LEN)
        return NULL;

    char* name = cml_malloc((size_t)name_len + 1);
    if (!name)
        return NULL;
    if (!io_read(f, name, (size_t)name_len) || !io_read(f, dtype, sizeof(*dtype)) ||
        !io_read(f, ndim, sizeof(*ndim)) || *ndim < 0 || *ndim > CML_IO_MAX_NDIM) {
        cml_free(name);
        return NULL;
    }
    name[name_len] = '\0';
    return name;
}

int model_load(Module* model, const char* filepath) {
    if (!model || !filepath)
        return -1;

    FILE* f = fopen(filepath, "rb");
    if (!f) {
        LOG_ERROR("Failed to open file for reading: %s", filepath);
        return -1;
    }
    char magic[4];
    int32_t num_params = 0;
    if (!io_read(f, magic, 4) || memcmp(magic, "CML", 3) != 0) {
        LOG_ERROR("Invalid model file magic");
        fclose(f);
        return -1;
    }

    uint32_t version;
    if (!io_read(f, &version, sizeof(version)) || version != CML_MODEL_VERSION) {
        LOG_ERROR("Unsupported model file version %u (expected %d)", version, CML_MODEL_VERSION);
        fclose(f);
        return -1;
    }

    if (!io_read(f, &num_params, sizeof(num_params)) || num_params < 0 ||
        num_params > CML_IO_MAX_PARAMS) {
        LOG_ERROR("Corrupt model file: bad parameter count");
        fclose(f);
        return -1;
    }

    for (int i = 0; i < num_params; i++) {
        int32_t dtype, ndim;
        char* name = read_param_header(f, &dtype, &ndim);
        if (!name) {
            LOG_ERROR("Corrupt model file: unreadable parameter header");
            fclose(f);
            return -1;
        }

        /* Shapes are matched by parameter name, so the stored dims are read
         * only to advance the stream (bounded by read_param_header). */
        for (int d = 0; d < ndim; d++) {
            int32_t dim;
            if (!io_read(f, &dim, sizeof(dim))) {
                cml_free(name);
                LOG_ERROR("Corrupt model file: truncated shape");
                fclose(f);
                return -1;
            }
        }

        uint64_t data_size;
        if (!io_read(f, &data_size, sizeof(data_size))) {
            cml_free(name);
            LOG_ERROR("Corrupt model file: truncated data size");
            fclose(f);
            return -1;
        }

        Parameter* param = module_get_parameter(model, name);
        bool consumed    = false;
        if (param && param->tensor) {
            tensor_ensure_executed(param->tensor);
            void* data           = tensor_data_ptr(param->tensor);
            size_t expected_size = param->tensor->numel * cml_dtype_size(param->tensor->dtype);
            if (data && expected_size == (size_t)data_size && io_read(f, data, (size_t)data_size)) {
                consumed = true;
            } else {
                /* Size mismatch or unreadable tensor: skip the payload. */
                consumed = io_skip(f, data_size);
                if (consumed)
                    LOG_WARNING("Parameter '%s' size mismatch, skipping", name);
            }
        } else {
            consumed = io_skip(f, data_size);
            if (consumed)
                LOG_WARNING("Parameter '%s' not found in model, skipping", name);
        }

        if (!consumed) {
            LOG_ERROR("Corrupt model file: parameter '%s' extends past end of file", name);
            cml_free(name);
            fclose(f);
            return -1;
        }
        cml_free(name);
    }

    fclose(f);
    return 0;
}

int model_save_checkpoint(Module* model, Optimizer* optimizer, int epoch, float loss,
                          const char* filepath) {
    if (!model || !filepath)
        return -1;

    FILE* f = fopen(filepath, "wb");
    if (!f) {
        LOG_ERROR("Failed to open checkpoint file: %s", filepath);
        return -1;
    }
    bool ok          = io_write(f, "CKP", 4);
    uint32_t version = CML_MODEL_VERSION;
    ok               = ok && io_write(f, &version, sizeof(version));

    int32_t ep        = epoch;
    ok                = ok && io_write(f, &ep, sizeof(ep));
    ok                = ok && io_write(f, &loss, sizeof(loss));
    ok                = ok && write_model_params(f, model);
    int32_t has_optim = optimizer ? 1 : 0;
    ok                = ok && io_write(f, &has_optim, sizeof(has_optim));

    if (optimizer && ok) {
        int32_t num_groups = optimizer->num_param_groups;
        ok                 = io_write(f, &num_groups, sizeof(num_groups));
        for (int i = 0; ok && i < optimizer->num_param_groups; i++) {
            ParameterGroup* g = &optimizer->param_groups[i];
            ok                = io_write(f, &g->lr, sizeof(g->lr)) &&
                 io_write(f, &g->step_count, sizeof(g->step_count));
        }
        /* Per-parameter moments (Adam m/v, momentum buffers, ...) so resuming
         * matches the pre-checkpoint optimization trajectory. */
        if (ok)
            optimizer_state_save(optimizer, f);
    }

    if (fclose(f) != 0)
        ok = false;
    if (!ok) {
        LOG_ERROR("Failed to write checkpoint file: %s", filepath);
        return -1;
    }
    return 0;
}

int model_load_checkpoint(Module* model, Optimizer* optimizer, int* epoch, float* loss,
                          const char* filepath) {
    if (!model || !filepath)
        return -1;

    FILE* f = fopen(filepath, "rb");
    if (!f) {
        LOG_ERROR("Failed to open checkpoint: %s", filepath);
        return -1;
    }

    char magic[4];
    if (!io_read(f, magic, 4) || memcmp(magic, "CKP", 3) != 0) {
        LOG_ERROR("Invalid checkpoint magic");
        fclose(f);
        return -1;
    }

    uint32_t version;
    if (!io_read(f, &version, sizeof(version)) || version != CML_MODEL_VERSION) {
        LOG_ERROR("Unsupported checkpoint version %u (expected %d)", version, CML_MODEL_VERSION);
        fclose(f);
        return -1;
    }

    int32_t ep;
    float l;
    int32_t num_params = 0;
    if (!io_read(f, &ep, sizeof(ep)) || !io_read(f, &l, sizeof(l))) {
        LOG_ERROR("Corrupt checkpoint: truncated header");
        fclose(f);
        return -1;
    }
    if (epoch)
        *epoch = ep;
    if (loss)
        *loss = l;

    if (!io_read(f, &num_params, sizeof(num_params)) || num_params < 0 ||
        num_params > CML_IO_MAX_PARAMS) {
        LOG_ERROR("Corrupt checkpoint: bad parameter count");
        fclose(f);
        return -1;
    }

    for (int i = 0; i < num_params; i++) {
        int32_t dtype, ndim;
        char* name = read_param_header(f, &dtype, &ndim);
        if (!name) {
            LOG_ERROR("Corrupt checkpoint: unreadable parameter header");
            fclose(f);
            return -1;
        }
        for (int d = 0; d < ndim; d++) {
            int32_t dim;
            if (!io_read(f, &dim, sizeof(dim))) {
                cml_free(name);
                LOG_ERROR("Corrupt checkpoint: truncated shape");
                fclose(f);
                return -1;
            }
        }

        uint64_t data_size;
        if (!io_read(f, &data_size, sizeof(data_size))) {
            cml_free(name);
            LOG_ERROR("Corrupt checkpoint: truncated data size");
            fclose(f);
            return -1;
        }

        Parameter* param = module_get_parameter(model, name);
        if (param && param->tensor) {
            tensor_ensure_executed(param->tensor);
            void* data      = tensor_data_ptr(param->tensor);
            size_t expected = param->tensor->numel * cml_dtype_size(param->tensor->dtype);
            if (data && expected == (size_t)data_size) {
                if (!io_read(f, data, (size_t)data_size)) {
                    cml_free(name);
                    LOG_ERROR("Corrupt checkpoint: truncated parameter '%s' data", name);
                    fclose(f);
                    return -1;
                }
            } else if (!io_skip(f, data_size)) {
                cml_free(name);
                LOG_ERROR("Corrupt checkpoint: parameter '%s' extends past end of file", name);
                fclose(f);
                return -1;
            }
        } else if (!io_skip(f, data_size)) {
            cml_free(name);
            LOG_ERROR("Corrupt checkpoint: unknown parameter '%s' extends past end of "
                      "file",
                      name);
            fclose(f);
            return -1;
        }
        cml_free(name);
    }
    int32_t has_optim;
    if (!io_read(f, &has_optim, sizeof(has_optim))) {
        LOG_ERROR("Corrupt checkpoint: truncated optimizer marker");
        fclose(f);
        return -1;
    }

    if (has_optim && optimizer) {
        int32_t num_groups;
        if (!io_read(f, &num_groups, sizeof(num_groups)) || num_groups < 0 ||
            num_groups > CML_IO_MAX_PARAMS) {
            LOG_ERROR("Corrupt checkpoint: bad optimizer group count");
            fclose(f);
            return -1;
        }
        int groups_to_read =
            num_groups < optimizer->num_param_groups ? num_groups : optimizer->num_param_groups;
        for (int i = 0; i < groups_to_read; i++) {
            float lr;
            int32_t step;
            if (!io_read(f, &lr, sizeof(lr)) || !io_read(f, &step, sizeof(step))) {
                LOG_ERROR("Corrupt checkpoint: truncated optimizer group");
                fclose(f);
                return -1;
            }
            optimizer->param_groups[i].lr         = lr;
            optimizer->param_groups[i].step_count = step;
        }
        for (int i = groups_to_read; i < num_groups; i++) {
            float lr;
            int32_t step;
            if (!io_read(f, &lr, sizeof(lr)) || !io_read(f, &step, sizeof(step))) {
                LOG_ERROR("Corrupt checkpoint: truncated optimizer group");
                fclose(f);
                return -1;
            }
        }
        if (num_groups == optimizer->num_param_groups && optimizer_state_load(optimizer, f) != 0) {
            LOG_WARNING("Checkpoint optimizer state could not be restored; "
                        "moments start from zero");
        }
    }

    fclose(f);
    return 0;
}
