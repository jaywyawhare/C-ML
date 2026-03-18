#include "core/model_architecture.h"
#include "nn.h"
#include "nn/layers/sequential.h"
#include "nn/layers/linear.h"
#include "nn/layers/activations.h"
#include "nn/layers/conv2d.h"
#include "nn/layers/pooling.h"
#include "core/logging.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

ModelArchitecture* model_architecture_create(void) {
    ModelArchitecture* arch = malloc(sizeof(ModelArchitecture));
    if (!arch)
        return NULL;

    arch->layers           = NULL;
    arch->num_layers       = 0;
    arch->capacity         = 0;
    arch->total_params     = 0;
    arch->trainable_params = 0;

    return arch;
}

static void write_json_string(FILE* f, const char* str) {
    fputc('"', f);
    for (const char* p = str; *p; p++) {
        if (*p == '"' || *p == '\\')
            fputc('\\', f);
        fputc(*p, f);
    }
    fputc('"', f);
}

static int extract_layer_info(Module* module, int layer_idx, LayerInfo* info) {
    if (!module || !info)
        return -1;

    memset(info, 0, sizeof(LayerInfo));
    info->type        = module->name;
    info->layer_index = layer_idx;

    if (strcmp(module->name, "Linear") == 0) {
        Linear* linear     = (Linear*)module;
        info->in_features  = linear->in_features;
        info->out_features = linear->out_features;
        info->has_bias     = linear->use_bias;

        if (linear->weight) {
            info->num_params += (int)linear->weight->tensor->numel;
        }
        if (linear->bias && linear->use_bias) {
            info->num_params += (int)linear->bias->tensor->numel;
        }
    } else if (strcmp(module->name, "Conv2d") == 0) {
        info->num_params =
            module->num_parameters > 0 ? (int)module->parameters[0]->tensor->numel : 0;
    } else if (strcmp(module->name, "ReLU") == 0 || strcmp(module->name, "Sigmoid") == 0 ||
               strcmp(module->name, "Tanh") == 0) {
        info->num_params = 0;
    } else {
        for (int i = 0; i < module->num_parameters; i++) {
            if (module->parameters[i] && module->parameters[i]->tensor) {
                info->num_params += (int)module->parameters[i]->tensor->numel;
            }
        }
    }

    return 0;
}

static int extract_from_sequential(Sequential* seq, ModelArchitecture* arch) {
    if (!seq || !arch)
        return -1;

    int num_modules = sequential_get_length(seq);

    if ((size_t)arch->num_layers + (size_t)num_modules > arch->capacity) {
        size_t new_capacity = arch->capacity == 0 ? 16 : arch->capacity * 2;
        while (new_capacity < (size_t)arch->num_layers + (size_t)num_modules) {
            new_capacity *= 2;
        }
        LayerInfo* new_layers = realloc(arch->layers, (size_t)new_capacity * sizeof(LayerInfo));
        if (!new_layers) {
            LOG_ERROR("Failed to allocate memory for layers");
            return -1;
        }
        arch->layers   = new_layers;
        arch->capacity = new_capacity;
    }

    for (int i = 0; i < num_modules; i++) {
        Module* child = sequential_get(seq, i);
        if (!child)
            continue;

        LayerInfo* info = &arch->layers[arch->num_layers];
        if (extract_layer_info(child, (int)arch->num_layers, info) == 0) {
            arch->num_layers++;
        }
    }

    return 0;
}

int model_architecture_extract(Module* module, ModelArchitecture* arch) {
    if (!module || !arch)
        return -1;

    if (strcmp(module->name, "Sequential") == 0) {
        Sequential* seq = (Sequential*)module;
        if (extract_from_sequential(seq, arch) != 0) {
            return -1;
        }
    } else {
        if (arch->num_layers >= arch->capacity) {
            size_t new_capacity   = arch->capacity == 0 ? 8 : arch->capacity * 2;
            LayerInfo* new_layers = realloc(arch->layers, (size_t)new_capacity * sizeof(LayerInfo));
            if (!new_layers)
                return -1;
            arch->layers   = new_layers;
            arch->capacity = new_capacity;
        }

        LayerInfo* info = &arch->layers[arch->num_layers];
        if (extract_layer_info(module, 0, info) == 0) {
            arch->num_layers++;
        }
    }

    Parameter** params = NULL;
    int num_params     = 0;
    if (module_collect_parameters(module, &params, &num_params, true) == 0) {
        arch->total_params     = 0;
        arch->trainable_params = 0;
        for (int i = 0; i < num_params; i++) {
            if (params[i] && params[i]->tensor) {
                arch->total_params += (int)params[i]->tensor->numel;
                if (params[i]->requires_grad) {
                    arch->trainable_params += (int)params[i]->tensor->numel;
                }
            }
        }
        if (params)
            free(params);
    }

    return 0;
}

int model_architecture_export_json(const ModelArchitecture* arch, const char* path) {
    if (!arch || !path)
        return -1;

    FILE* f = fopen(path, "wb");
    if (!f) {
        LOG_ERROR("Failed to open file for writing: %s", path);
        return -1;
    }

    fputs("{\n", f);
    fputs("  \"layers\": [\n", f);

    for (size_t i = 0; i < arch->num_layers; i++) {
        const LayerInfo* layer = &arch->layers[i];
        if (i > 0)
            fputs(",\n", f);

        fputs("    {\n", f);
        fprintf(f, "      \"index\": %d,\n", layer->layer_index);
        fprintf(f, "      \"type\": ");
        write_json_string(f, layer->type);
        fputs(",\n", f);

        if (layer->in_features > 0 && layer->out_features > 0) {
            fprintf(f, "      \"in_features\": %d,\n", layer->in_features);
            fprintf(f, "      \"out_features\": %d,\n", layer->out_features);
        }
        if (layer->in_channels > 0 && layer->out_channels > 0) {
            fprintf(f, "      \"in_channels\": %d,\n", layer->in_channels);
            fprintf(f, "      \"out_channels\": %d,\n", layer->out_channels);
        }
        if (layer->kernel_size > 0) {
            fprintf(f, "      \"kernel_size\": %d,\n", layer->kernel_size);
        }
        if (layer->stride > 0) {
            fprintf(f, "      \"stride\": %d,\n", layer->stride);
        }
        if (layer->padding >= 0) {
            fprintf(f, "      \"padding\": %d,\n", layer->padding);
        }

        fprintf(f, "      \"has_bias\": %s,\n", layer->has_bias ? "true" : "false");
        fprintf(f, "      \"num_params\": %d\n", layer->num_params);
        fputs("    }", f);
    }

    fputs("\n  ],\n", f);
    fprintf(f, "  \"total_params\": %d,\n", arch->total_params);
    fprintf(f, "  \"trainable_params\": %d\n", arch->trainable_params);
    fputs("}\n", f);

    fclose(f);
    return 0;
}

void model_architecture_free(ModelArchitecture* arch) {
    if (!arch)
        return;

    if (arch->layers) {
        for (size_t i = 0; i < arch->num_layers; i++) {
            if (arch->layers[i].details) {
                free(arch->layers[i].details);
            }
        }
        free(arch->layers);
    }

    free(arch);
}
