#include "nn/model_io.h"
#include "nn.h"
#include "optim.h"
#include "tensor/tensor.h"
#include "core/logging.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define CML_MODEL_MAGIC "CML\0"
#define CML_CKPT_MAGIC  "CKP\0"
#define CML_MODEL_VERSION 1

int model_save(Module* model, const char* filepath) {
    if (!model || !filepath) return -1;

    FILE* f = fopen(filepath, "wb");
    if (!f) {
        LOG_ERROR("Failed to open file for writing: %s", filepath);
        return -1;
    }
    fwrite("CML", 1, 4, f); // magic (includes null terminator)
    uint32_t version = CML_MODEL_VERSION;
    fwrite(&version, sizeof(uint32_t), 1, f);

    int32_t num_params = model->num_parameters;
    fwrite(&num_params, sizeof(int32_t), 1, f);

    for (int i = 0; i < model->num_parameters; i++) {
        Parameter* param = model->parameters[i];
        if (!param || !param->tensor) continue;

        Tensor* t = param->tensor;
        tensor_ensure_executed(t);
        const char* name = param->name ? param->name : "";
        int32_t name_len = (int32_t)strlen(name);
        fwrite(&name_len, sizeof(int32_t), 1, f);
        fwrite(name, 1, (size_t)name_len, f);
        int32_t dtype = (int32_t)t->dtype;
        fwrite(&dtype, sizeof(int32_t), 1, f);

        int32_t ndim = t->ndim;
        fwrite(&ndim, sizeof(int32_t), 1, f);

        for (int d = 0; d < t->ndim; d++) {
            int32_t dim = t->shape[d];
            fwrite(&dim, sizeof(int32_t), 1, f);
        }
        size_t elem_size = cml_dtype_size(t->dtype);
        uint64_t data_size = (uint64_t)(t->numel * elem_size);
        fwrite(&data_size, sizeof(uint64_t), 1, f);

        void* data = tensor_data_ptr(t);
        if (data) {
            fwrite(data, 1, (size_t)data_size, f);
        }
    }

    fclose(f);
    return 0;
}

int model_load(Module* model, const char* filepath) {
    if (!model || !filepath) return -1;

    FILE* f = fopen(filepath, "rb");
    if (!f) {
        LOG_ERROR("Failed to open file for reading: %s", filepath);
        return -1;
    }
    char magic[4];
    fread(magic, 1, 4, f);
    if (memcmp(magic, "CML", 3) != 0) {
        LOG_ERROR("Invalid model file magic");
        fclose(f);
        return -1;
    }

    uint32_t version;
    fread(&version, sizeof(uint32_t), 1, f);

    int32_t num_params;
    fread(&num_params, sizeof(int32_t), 1, f);

    for (int i = 0; i < num_params; i++) {
        int32_t name_len;
        fread(&name_len, sizeof(int32_t), 1, f);
        char* name = malloc((size_t)(name_len + 1));
        if (!name) { fclose(f); return -1; }
        fread(name, 1, (size_t)name_len, f);
        name[name_len] = '\0';
        int32_t dtype;
        fread(&dtype, sizeof(int32_t), 1, f);

        int32_t ndim;
        fread(&ndim, sizeof(int32_t), 1, f);

        int* shape = malloc((size_t)ndim * sizeof(int));
        for (int d = 0; d < ndim; d++) {
            int32_t dim;
            fread(&dim, sizeof(int32_t), 1, f);
            shape[d] = dim;
        }

        uint64_t data_size;
        fread(&data_size, sizeof(uint64_t), 1, f);
        Parameter* param = module_get_parameter(model, name);
        if (param && param->tensor) {
            tensor_ensure_executed(param->tensor);
            void* data = tensor_data_ptr(param->tensor);
            size_t expected_size = param->tensor->numel * cml_dtype_size(param->tensor->dtype);
            if (data && expected_size == (size_t)data_size) {
                fread(data, 1, (size_t)data_size, f);
            } else {
                fseek(f, (long)data_size, SEEK_CUR);
                LOG_WARNING("Parameter '%s' size mismatch, skipping", name);
            }
        } else {
            fseek(f, (long)data_size, SEEK_CUR);
            LOG_WARNING("Parameter '%s' not found in model, skipping", name);
        }

        free(name);
        free(shape);
    }

    fclose(f);
    return 0;
}

int model_save_checkpoint(Module* model, Optimizer* optimizer, int epoch, float loss,
                          const char* filepath) {
    if (!model || !filepath) return -1;

    FILE* f = fopen(filepath, "wb");
    if (!f) {
        LOG_ERROR("Failed to open checkpoint file: %s", filepath);
        return -1;
    }
    fwrite("CKP", 1, 4, f);
    uint32_t version = CML_MODEL_VERSION;
    fwrite(&version, sizeof(uint32_t), 1, f);

    int32_t ep = epoch;
    fwrite(&ep, sizeof(int32_t), 1, f);
    fwrite(&loss, sizeof(float), 1, f);
    int32_t num_params = model->num_parameters;
    fwrite(&num_params, sizeof(int32_t), 1, f);

    for (int i = 0; i < model->num_parameters; i++) {
        Parameter* param = model->parameters[i];
        if (!param || !param->tensor) continue;

        Tensor* t = param->tensor;
        tensor_ensure_executed(t);

        const char* name = param->name ? param->name : "";
        int32_t name_len = (int32_t)strlen(name);
        fwrite(&name_len, sizeof(int32_t), 1, f);
        fwrite(name, 1, (size_t)name_len, f);

        int32_t dtype = (int32_t)t->dtype;
        fwrite(&dtype, sizeof(int32_t), 1, f);
        int32_t ndim = t->ndim;
        fwrite(&ndim, sizeof(int32_t), 1, f);
        for (int d = 0; d < t->ndim; d++) {
            int32_t dim = t->shape[d];
            fwrite(&dim, sizeof(int32_t), 1, f);
        }

        size_t elem_size = cml_dtype_size(t->dtype);
        uint64_t data_size = (uint64_t)(t->numel * elem_size);
        fwrite(&data_size, sizeof(uint64_t), 1, f);
        void* data = tensor_data_ptr(t);
        if (data) fwrite(data, 1, (size_t)data_size, f);
    }
    int32_t has_optim = optimizer ? 1 : 0;
    fwrite(&has_optim, sizeof(int32_t), 1, f);

    if (optimizer) {
        int32_t num_groups = optimizer->num_param_groups;
        fwrite(&num_groups, sizeof(int32_t), 1, f);
        for (int i = 0; i < optimizer->num_param_groups; i++) {
            ParameterGroup* g = &optimizer->param_groups[i];
            fwrite(&g->lr, sizeof(float), 1, f);
            fwrite(&g->step_count, sizeof(int32_t), 1, f);
        }
    }

    fclose(f);
    return 0;
}

int model_load_checkpoint(Module* model, Optimizer* optimizer, int* epoch, float* loss,
                          const char* filepath) {
    if (!model || !filepath) return -1;

    FILE* f = fopen(filepath, "rb");
    if (!f) {
        LOG_ERROR("Failed to open checkpoint: %s", filepath);
        return -1;
    }

    char magic[4];
    fread(magic, 1, 4, f);
    if (memcmp(magic, "CKP", 3) != 0) {
        LOG_ERROR("Invalid checkpoint magic");
        fclose(f);
        return -1;
    }

    uint32_t version;
    fread(&version, sizeof(uint32_t), 1, f);

    int32_t ep;
    fread(&ep, sizeof(int32_t), 1, f);
    if (epoch) *epoch = ep;

    float l;
    fread(&l, sizeof(float), 1, f);
    if (loss) *loss = l;
    int32_t num_params;
    fread(&num_params, sizeof(int32_t), 1, f);

    for (int i = 0; i < num_params; i++) {
        int32_t name_len;
        fread(&name_len, sizeof(int32_t), 1, f);
        char* name = malloc((size_t)(name_len + 1));
        if (!name) { fclose(f); return -1; }
        fread(name, 1, (size_t)name_len, f);
        name[name_len] = '\0';

        int32_t dtype;
        fread(&dtype, sizeof(int32_t), 1, f);
        int32_t ndim;
        fread(&ndim, sizeof(int32_t), 1, f);
        for (int d = 0; d < ndim; d++) { int32_t dim; fread(&dim, sizeof(int32_t), 1, f); }

        uint64_t data_size;
        fread(&data_size, sizeof(uint64_t), 1, f);

        Parameter* param = module_get_parameter(model, name);
        if (param && param->tensor) {
            tensor_ensure_executed(param->tensor);
            void* data = tensor_data_ptr(param->tensor);
            size_t expected = param->tensor->numel * cml_dtype_size(param->tensor->dtype);
            if (data && expected == (size_t)data_size) {
                fread(data, 1, (size_t)data_size, f);
            } else {
                fseek(f, (long)data_size, SEEK_CUR);
            }
        } else {
            fseek(f, (long)data_size, SEEK_CUR);
        }
        free(name);
    }
    int32_t has_optim;
    fread(&has_optim, sizeof(int32_t), 1, f);

    if (has_optim && optimizer) {
        int32_t num_groups;
        fread(&num_groups, sizeof(int32_t), 1, f);
        int groups_to_read = num_groups < optimizer->num_param_groups ? num_groups : optimizer->num_param_groups;
        for (int i = 0; i < groups_to_read; i++) {
            float lr;
            int32_t step;
            fread(&lr, sizeof(float), 1, f);
            fread(&step, sizeof(int32_t), 1, f);
            optimizer->param_groups[i].lr = lr;
            optimizer->param_groups[i].step_count = step;
        }
        for (int i = groups_to_read; i < num_groups; i++) {
            float lr; int32_t step;
            fread(&lr, sizeof(float), 1, f);
            fread(&step, sizeof(int32_t), 1, f);
        }
    }

    fclose(f);
    return 0;
}
