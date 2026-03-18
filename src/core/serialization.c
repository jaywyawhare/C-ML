#include "core/serialization.h"
#include "core/logging.h"
#include "backend/device.h"
#include "nn.h"
#include "nn/layers/linear.h"
#include "nn/layers/sequential.h"
#include "tensor/tensor.h"
#include "autograd/forward_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int module_named_parameters(Module* module, NamedParameter** named_params, int* num_params) {
    if (!module || !named_params || !num_params) {
        return -1;
    }

    Parameter** params = NULL;
    int total_params   = 0;
    if (module_collect_parameters(module, &params, &total_params, true) != 0) {
        return -1;
    }

    NamedParameter* np = malloc((size_t)total_params * sizeof(NamedParameter));
    if (!np) {
        free(params);
        return -1;
    }

    int idx = 0;
    char name_buf[256];

    if (strcmp(module->name, "Sequential") == 0) {
        Sequential* seq = (Sequential*)module;
        int num_modules = sequential_get_length(seq);
        for (int i = 0; i < num_modules; i++) {
            Module* child = sequential_get(seq, i);
            if (child) {
                Parameter** child_params = NULL;
                int child_num_params     = 0;
                if (module_collect_parameters(child, &child_params, &child_num_params, true) == 0) {
                    for (int j = 0; j < child_num_params; j++) {
                        snprintf(name_buf, sizeof(name_buf), "%d.%s.%d", i, child->name, j);
                        np[idx].name      = strdup(name_buf);
                        np[idx].parameter = child_params[j];
                        idx++;
                    }
                    free(child_params);
                }
            }
        }
    } else if (strcmp(module->name, "Linear") == 0) {
        Linear* linear = (Linear*)module;
        if (linear->weight) {
            snprintf(name_buf, sizeof(name_buf), "weight");
            np[idx].name      = strdup(name_buf);
            np[idx].parameter = linear->weight;
            idx++;
        }
        if (linear->bias) {
            snprintf(name_buf, sizeof(name_buf), "bias");
            np[idx].name      = strdup(name_buf);
            np[idx].parameter = linear->bias;
            idx++;
        }
    } else {
        for (int i = 0; i < total_params; i++) {
            snprintf(name_buf, sizeof(name_buf), "param_%d", i);
            np[idx].name      = strdup(name_buf);
            np[idx].parameter = params[i];
            idx++;
        }
    }

    *named_params = np;
    *num_params   = idx;
    return 0;
}

void module_named_parameters_free(NamedParameter* named_params, int num_params) {
    if (!named_params) {
        return;
    }

    for (int i = 0; i < num_params; i++) {
        if (named_params[i].name) {
            free((void*)named_params[i].name);
        }
    }
    free(named_params);
}

// Binary format for model serialization:
// - Magic number (4 bytes): "CMLM"
// - Version (1 byte): 1
// - Number of parameters (4 bytes): int
// - For each parameter:
//   - Name length (4 bytes): int
//   - Name (name_length bytes): char array
//   - Tensor data (using tensor serialization format)

#define MODEL_MAGIC "CMLM"
#define MODEL_VERSION 1

int module_save_stream(Module* module, FILE* file) {
    if (!module || !file) {
        LOG_ERROR("Invalid arguments to module_save_stream");
        return -1;
    }

    NamedParameter* named_params = NULL;
    int num_params               = 0;
    if (module_named_parameters(module, &named_params, &num_params) != 0) {
        LOG_ERROR("Failed to get named parameters");
        return -1;
    }
    if (fwrite(MODEL_MAGIC, 1, 4, file) != 4) {
        LOG_ERROR("Failed to write magic number");
        module_named_parameters_free(named_params, num_params);
        return -1;
    }
    uint8_t version = MODEL_VERSION;
    if (fwrite(&version, 1, 1, file) != 1) {
        LOG_ERROR("Failed to write version");
        module_named_parameters_free(named_params, num_params);
        return -1;
    }
    int32_t num_params_int = (int32_t)num_params;
    if (fwrite(&num_params_int, sizeof(int32_t), 1, file) != 1) {
        LOG_ERROR("Failed to write number of parameters");
        module_named_parameters_free(named_params, num_params);
        return -1;
    }
    for (int i = 0; i < num_params; i++) {
        NamedParameter* np = &named_params[i];
        Parameter* param   = np->parameter;

        if (!param || !param->tensor) {
            LOG_WARNING("Skipping parameter %d: invalid parameter", i);
            continue;
        }
        int name_len         = np->name ? (int)strlen(np->name) : 0;
        int32_t name_len_int = (int32_t)name_len;
        if (fwrite(&name_len_int, sizeof(int32_t), 1, file) != 1) {
            LOG_ERROR("Failed to write name length for parameter %d", i);
            module_named_parameters_free(named_params, num_params);
            return -1;
        }
        if (name_len > 0) {
            size_t name_len_size = (size_t)name_len;
            if (fwrite(np->name, 1, name_len_size, file) != name_len_size) {
                LOG_ERROR("Failed to write name for parameter %d", i);
                module_named_parameters_free(named_params, num_params);
                return -1;
            }
        }
        if (tensor_write_stream(param->tensor, file) != 0) {
            LOG_ERROR("Failed to write tensor for parameter %d", i);
            module_named_parameters_free(named_params, num_params);
            return -1;
        }
    }

    module_named_parameters_free(named_params, num_params);
    return 0;
}

int module_save(Module* module, const char* filepath) {
    if (!module || !filepath) {
        LOG_ERROR("Invalid arguments to module_save");
        return -1;
    }

    FILE* file = fopen(filepath, "wb");
    if (!file) {
        LOG_ERROR("Failed to open file for writing: %s", filepath);
        return -1;
    }

    int result = module_save_stream(module, file);
    fclose(file);

    if (result == 0) {
    }

    return result;
}

int module_load_stream(Module* module, FILE* file) {
    if (!module || !file) {
        LOG_ERROR("Invalid arguments to module_load_stream");
        return -1;
    }
    char magic[5] = {0};
    if (fread(magic, 1, 4, file) != 4) {
        LOG_ERROR("Failed to read magic number");
        return -1;
    }

    if (strncmp(magic, MODEL_MAGIC, 4) != 0) {
        LOG_ERROR("Invalid magic number: expected %s, got %.4s", MODEL_MAGIC, magic);
        return -1;
    }
    uint8_t version;
    if (fread(&version, 1, 1, file) != 1) {
        LOG_ERROR("Failed to read version");
        return -1;
    }

    if (version != MODEL_VERSION) {
        LOG_ERROR("Unsupported version: %d (expected %d)", version, MODEL_VERSION);
        return -1;
    }
    int32_t num_params_int;
    if (fread(&num_params_int, sizeof(int32_t), 1, file) != 1) {
        LOG_ERROR("Failed to read number of parameters");
        return -1;
    }
    int num_params = (int)num_params_int;
    NamedParameter* module_params = NULL;
    int module_num_params         = 0;
    if (module_named_parameters(module, &module_params, &module_num_params) != 0) {
        LOG_ERROR("Failed to get module parameters");
        return -1;
    }
    for (int i = 0; i < num_params && i < module_num_params; i++) {
        int32_t name_len_int;
        if (fread(&name_len_int, sizeof(int32_t), 1, file) != 1) {
            LOG_ERROR("Failed to read name length for parameter %d", i);
            module_named_parameters_free(module_params, module_num_params);
            return -1;
        }
        int name_len = (int)name_len_int;
        char* name = NULL;
        if (name_len > 0) {
            name = malloc((size_t)name_len + 1);
            if (!name) {
                LOG_ERROR("Failed to allocate name buffer");
                module_named_parameters_free(module_params, module_num_params);
                return -1;
            }

            size_t name_len_size = (size_t)name_len;
            if (fread(name, 1, name_len_size, file) != name_len_size) {
                LOG_ERROR("Failed to read name for parameter %d", i);
                free(name);
                module_named_parameters_free(module_params, module_num_params);
                return -1;
            }
            name[name_len] = '\0';
        }
        Tensor* loaded_tensor = tensor_read_stream(file);
        if (!loaded_tensor) {
            LOG_ERROR("Failed to read tensor for parameter %d", i);
            if (name) {
                free(name);
            }
            module_named_parameters_free(module_params, module_num_params);
            return -1;
        }
        Parameter* target_param = NULL;
        if (name) {
            for (int j = 0; j < module_num_params; j++) {
                if (module_params[j].name && strcmp(module_params[j].name, name) == 0) {
                    target_param = module_params[j].parameter;
                    break;
                }
            }
        } else if (i < module_num_params) {
            target_param = module_params[i].parameter;
        }

        if (target_param && target_param->tensor) {
            bool shapes_match = true;
            if (target_param->tensor->ndim != loaded_tensor->ndim) {
                shapes_match = false;
            } else {
                for (int d = 0; d < target_param->tensor->ndim; d++) {
                    if (target_param->tensor->shape[d] != loaded_tensor->shape[d]) {
                        shapes_match = false;
                        break;
                    }
                }
            }

            if (shapes_match && target_param->tensor->dtype == loaded_tensor->dtype) {
                size_t data_size =
                    target_param->tensor->numel * cml_dtype_size(target_param->tensor->dtype);
                if (loaded_tensor->device == DEVICE_CPU &&
                    target_param->tensor->device == DEVICE_CPU) {
                    memcpy(target_param->tensor->data, loaded_tensor->data, data_size);
                } else {
                    void* cpu_buffer = malloc(data_size);
                    if (cpu_buffer) {
                        if (loaded_tensor->device == DEVICE_CPU) {
                            memcpy(cpu_buffer, loaded_tensor->data, data_size);
                        } else {
                            device_copy_from_device(cpu_buffer, loaded_tensor->data, data_size,
                                                    loaded_tensor->device);
                        }

                        if (target_param->tensor->device == DEVICE_CPU) {
                            memcpy(target_param->tensor->data, cpu_buffer, data_size);
                        } else {
                            device_copy_to_device(target_param->tensor->data, cpu_buffer, data_size,
                                                  target_param->tensor->device);
                        }
                        free(cpu_buffer);
                    }
                }
            } else {
                LOG_WARNING("Shape or dtype mismatch for parameter %s, skipping",
                            name ? name : "unknown");
            }
        }

        if (name) {
            free(name);
        }
        tensor_free(loaded_tensor);
    }

    module_named_parameters_free(module_params, module_num_params);
    return 0;
}

int module_load(Module* module, const char* filepath) {
    if (!module || !filepath) {
        LOG_ERROR("Invalid arguments to module_load");
        return -1;
    }

    FILE* file = fopen(filepath, "rb");
    if (!file) {
        LOG_ERROR("Failed to open file for reading: %s", filepath);
        return -1;
    }

    int result = module_load_stream(module, file);
    fclose(file);

    if (result == 0) {
    }

    return result;
}

// Binary format for tensor serialization:
// - Magic number (4 bytes): "CMLT"
// - Version (1 byte): 1
// - DType (1 byte): enum value
// - DeviceType (1 byte): enum value
// - ndim (1 byte): number of dimensions
// - shape (ndim * 4 bytes): int array
// - numel (8 bytes): size_t
// - data (numel * dtype_size bytes): raw data

#define TENSOR_MAGIC "CMLT"
#define TENSOR_VERSION 1

int tensor_write_stream(Tensor* tensor, FILE* file) {
    if (!tensor || !file) {
        LOG_ERROR("Invalid arguments to tensor_write_stream");
        return -1;
    }
    if (fwrite(TENSOR_MAGIC, 1, 4, file) != 4) {
        LOG_ERROR("Failed to write magic number");
        return -1;
    }
    uint8_t version = TENSOR_VERSION;
    if (fwrite(&version, 1, 1, file) != 1) {
        LOG_ERROR("Failed to write version");
        return -1;
    }
    uint8_t dtype = (uint8_t)tensor->dtype;
    if (fwrite(&dtype, 1, 1, file) != 1) {
        LOG_ERROR("Failed to write dtype");
        return -1;
    }
    uint8_t device = (uint8_t)tensor->device;
    if (fwrite(&device, 1, 1, file) != 1) {
        LOG_ERROR("Failed to write device");
        return -1;
    }
    uint8_t ndim = (uint8_t)tensor->ndim;
    if (fwrite(&ndim, 1, 1, file) != 1) {
        LOG_ERROR("Failed to write ndim");
        return -1;
    }
    if (fwrite(tensor->shape, sizeof(int), (size_t)ndim, file) != (size_t)ndim) {
        LOG_ERROR("Failed to write shape");
        return -1;
    }
    if (fwrite(&tensor->numel, sizeof(size_t), 1, file) != 1) {
        LOG_ERROR("Failed to write numel");
        return -1;
    }
    Tensor* contiguous_tensor = tensor;
    if (!tensor->is_contiguous) {
        contiguous_tensor = tensor_contiguous(tensor);
        if (!contiguous_tensor) {
            LOG_ERROR("Failed to make tensor contiguous");
            return -1;
        }
    }
    size_t data_size = tensor->numel * cml_dtype_size(tensor->dtype);
    void* cpu_data   = NULL;
    if (tensor->device != DEVICE_CPU) {
        cpu_data = malloc(data_size);
        if (!cpu_data) {
            LOG_ERROR("Failed to allocate CPU buffer");
            if (contiguous_tensor != tensor) {
                tensor_free(contiguous_tensor);
            }
            return -1;
        }

        int result =
            device_copy_from_device(cpu_data, contiguous_tensor->data, data_size, tensor->device);
        if (result != 0) {
            LOG_ERROR("Failed to copy tensor data to CPU");
            free(cpu_data);
            if (contiguous_tensor != tensor) {
                tensor_free(contiguous_tensor);
            }
            return -1;
        }
    } else {
        cpu_data = contiguous_tensor->data;
    }
    if (fwrite(cpu_data, 1, data_size, file) != data_size) {
        LOG_ERROR("Failed to write tensor data");
        if (cpu_data != contiguous_tensor->data) {
            free(cpu_data);
        }
        if (contiguous_tensor != tensor) {
            tensor_free(contiguous_tensor);
        }
        return -1;
    }
    if (cpu_data != contiguous_tensor->data) {
        free(cpu_data);
    }
    if (contiguous_tensor != tensor) {
        tensor_free(contiguous_tensor);
    }

    return 0;
}

int tensor_write_file(Tensor* tensor, const char* filepath) {
    if (!tensor || !filepath) {
        LOG_ERROR("Invalid arguments to tensor_write_file");
        return -1;
    }

    FILE* file = fopen(filepath, "wb");
    if (!file) {
        LOG_ERROR("Failed to open file for writing: %s", filepath);
        return -1;
    }

    int result = tensor_write_stream(tensor, file);
    fclose(file);

    if (result == 0) {
    }

    return result;
}

Tensor* tensor_read_stream(FILE* file) {
    if (!file) {
        LOG_ERROR("Invalid arguments to tensor_read_stream");
        return NULL;
    }
    char magic[5] = {0};
    if (fread(magic, 1, 4, file) != 4) {
        LOG_ERROR("Failed to read magic number");
        return NULL;
    }

    if (strncmp(magic, TENSOR_MAGIC, 4) != 0) {
        LOG_ERROR("Invalid magic number: expected %s, got %.4s", TENSOR_MAGIC, magic);
        return NULL;
    }
    uint8_t version;
    if (fread(&version, 1, 1, file) != 1) {
        LOG_ERROR("Failed to read version");
        return NULL;
    }

    if (version != TENSOR_VERSION) {
        LOG_ERROR("Unsupported version: %d (expected %d)", version, TENSOR_VERSION);
        return NULL;
    }
    uint8_t dtype_val;
    if (fread(&dtype_val, 1, 1, file) != 1) {
        LOG_ERROR("Failed to read dtype");
        return NULL;
    }
    DType dtype = (DType)dtype_val;
    uint8_t device_val;
    if (fread(&device_val, 1, 1, file) != 1) {
        LOG_ERROR("Failed to read device");
        return NULL;
    }
    DeviceType device = (DeviceType)device_val;
    uint8_t ndim_val;
    if (fread(&ndim_val, 1, 1, file) != 1) {
        LOG_ERROR("Failed to read ndim");
        return NULL;
    }
    int ndim = (int)ndim_val;

    if (ndim <= 0 || ndim > 32) {
        LOG_ERROR("Invalid ndim: %d", ndim);
        return NULL;
    }
    int* shape = malloc((size_t)ndim * sizeof(int));
    if (!shape) {
        LOG_ERROR("Failed to allocate shape array");
        return NULL;
    }

    if (fread(shape, sizeof(int), (size_t)ndim, file) != (size_t)ndim) {
        LOG_ERROR("Failed to read shape");
        free(shape);
        return NULL;
    }
    size_t numel;
    if (fread(&numel, sizeof(size_t), 1, file) != 1) {
        LOG_ERROR("Failed to read numel");
        free(shape);
        return NULL;
    }
    TensorConfig config = {.dtype = dtype, .device = device, .has_dtype = true, .has_device = true};
    Tensor* tensor      = tensor_empty(shape, ndim, &config);
    free(shape);

    if (!tensor) {
        LOG_ERROR("Failed to create tensor");
        return NULL;
    }
    size_t data_size = numel * cml_dtype_size(dtype);
    void* cpu_data   = malloc(data_size);
    if (!cpu_data) {
        LOG_ERROR("Failed to allocate CPU buffer");
        tensor_free(tensor);
        return NULL;
    }

    if (fread(cpu_data, 1, data_size, file) != data_size) {
        LOG_ERROR("Failed to read tensor data");
        free(cpu_data);
        tensor_free(tensor);
        return NULL;
    }
    if (device == DEVICE_CPU) {
        memcpy(tensor->data, cpu_data, data_size);
    } else {
        int result = device_copy_to_device(tensor->data, cpu_data, data_size, device);
        if (result != 0) {
            LOG_ERROR("Failed to copy data to device");
            free(cpu_data);
            tensor_free(tensor);
            return NULL;
        }
    }

    free(cpu_data);

    return tensor;
}

Tensor* tensor_read_file(const char* filepath) {
    if (!filepath) {
        LOG_ERROR("Invalid arguments to tensor_read_file");
        return NULL;
    }

    FILE* file = fopen(filepath, "rb");
    if (!file) {
        LOG_ERROR("Failed to open file for reading: %s", filepath);
        return NULL;
    }

    Tensor* tensor = tensor_read_stream(file);
    fclose(file);

    if (tensor) {
    }

    return tensor;
}

typedef struct SGDMomentumState {
    Tensor* momentum_buffer;
} SGDMomentumState;

typedef struct AdamState {
    Tensor* exp_avg;
    Tensor* exp_avg_sq;
    Tensor* max_exp_avg_sq;
} AdamState;

typedef struct RMSpropState {
    Tensor* square_avg;
} RMSpropState;

typedef struct AdagradState {
    Tensor* sum_sq_grad;
} AdagradState;

// Binary format for optimizer serialization:
// - Magic number (4 bytes): "CMLO"
// - Version (1 byte): 1
// - Optimizer name length (4 bytes): int
// - Optimizer name (name_length bytes): char array
// - Number of parameter groups (4 bytes): int
// - For each parameter group:
//   - Learning rate (4 bytes): float
//   - Weight decay (4 bytes): float
//   - Momentum (4 bytes): float
//   - Beta1 (4 bytes): float
//   - Beta2 (4 bytes): float
//   - Epsilon (4 bytes): float
//   - Step count (4 bytes): int
//   - Number of parameters (4 bytes): int
//   - For each parameter:
//     - Parameter index (4 bytes): int
//     - State type (1 byte): uint8_t (0=none, 1=SGD, 2=Adam, 3=RMSprop, 4=Adagrad)
//     - For SGD: momentum_buffer tensor
//     - For Adam: exp_avg tensor, exp_avg_sq tensor, max_exp_avg_sq tensor (if amsgrad)
//     - For RMSprop: square_avg tensor
//     - For Adagrad: sum_sq_grad tensor

#define OPTIMIZER_MAGIC "CMLO"
#define OPTIMIZER_VERSION 1

typedef enum {
    OPTIMIZER_STATE_NONE    = 0,
    OPTIMIZER_STATE_SGD     = 1,
    OPTIMIZER_STATE_ADAM    = 2,
    OPTIMIZER_STATE_RMSPROP = 3,
    OPTIMIZER_STATE_ADAGRAD = 4
} OptimizerStateType;

int optimizer_save_stream(Optimizer* optimizer, FILE* file) {
    if (!optimizer || !file) {
        LOG_ERROR("Invalid arguments to optimizer_save_stream");
        return -1;
    }
    if (fwrite(OPTIMIZER_MAGIC, 1, 4, file) != 4) {
        LOG_ERROR("Failed to write magic number");
        return -1;
    }
    uint8_t version = OPTIMIZER_VERSION;
    if (fwrite(&version, 1, 1, file) != 1) {
        LOG_ERROR("Failed to write version");
        return -1;
    }
    int name_len         = optimizer->name ? (int)strlen(optimizer->name) : 0;
    int32_t name_len_int = (int32_t)name_len;
    if (fwrite(&name_len_int, sizeof(int32_t), 1, file) != 1) {
        LOG_ERROR("Failed to write name length");
        return -1;
    }

    if (name_len > 0) {
        if (fwrite(optimizer->name, 1, (size_t)name_len, file) != (size_t)name_len) {
            LOG_ERROR("Failed to write optimizer name");
            return -1;
        }
    }
    int32_t num_groups_int = (int32_t)optimizer->num_param_groups;
    if (fwrite(&num_groups_int, sizeof(int32_t), 1, file) != 1) {
        LOG_ERROR("Failed to write number of parameter groups");
        return -1;
    }
    for (int g = 0; g < optimizer->num_param_groups; g++) {
        ParameterGroup* group = &optimizer->param_groups[g];
        if (fwrite(&group->lr, sizeof(float), 1, file) != 1 ||
            fwrite(&group->weight_decay, sizeof(float), 1, file) != 1 ||
            fwrite(&group->momentum, sizeof(float), 1, file) != 1 ||
            fwrite(&group->beta1, sizeof(float), 1, file) != 1 ||
            fwrite(&group->beta2, sizeof(float), 1, file) != 1 ||
            fwrite(&group->epsilon, sizeof(float), 1, file) != 1) {
            LOG_ERROR("Failed to write hyperparameters for group %d", g);
            return -1;
        }
        int32_t step_count_int = (int32_t)group->step_count;
        if (fwrite(&step_count_int, sizeof(int32_t), 1, file) != 1) {
            LOG_ERROR("Failed to write step count for group %d", g);
            return -1;
        }
        int32_t num_params_int = (int32_t)group->num_parameters;
        if (fwrite(&num_params_int, sizeof(int32_t), 1, file) != 1) {
            LOG_ERROR("Failed to write number of parameters for group %d", g);
            return -1;
        }
        for (int i = 0; i < group->num_parameters; i++) {
            int32_t param_idx = (int32_t)i;
            if (fwrite(&param_idx, sizeof(int32_t), 1, file) != 1) {
                LOG_ERROR("Failed to write parameter index");
                return -1;
            }
            uint8_t state_type = OPTIMIZER_STATE_NONE;
            if (group->state) {
                if (strcmp(optimizer->name, "SGD") == 0) {
                    state_type                = OPTIMIZER_STATE_SGD;
                    SGDMomentumState** states = (SGDMomentumState**)group->state;
                    if (states[i] && states[i]->momentum_buffer) {
                        if (fwrite(&state_type, 1, 1, file) != 1 ||
                            tensor_write_stream(states[i]->momentum_buffer, file) != 0) {
                            LOG_ERROR("Failed to write SGD state for parameter %d", i);
                            return -1;
                        }
                    } else {
                        if (fwrite(&state_type, 1, 1, file) != 1) {
                            LOG_ERROR("Failed to write state type");
                            return -1;
                        }
                    }
                } else if (strcmp(optimizer->name, "Adam") == 0) {
                    state_type         = OPTIMIZER_STATE_ADAM;
                    AdamState** states = (AdamState**)group->state;
                    if (states[i]) {
                        if (fwrite(&state_type, 1, 1, file) != 1) {
                            LOG_ERROR("Failed to write state type");
                            return -1;
                        }
                        if (states[i]->exp_avg &&
                            tensor_write_stream(states[i]->exp_avg, file) != 0) {
                            LOG_ERROR("Failed to write Adam exp_avg for parameter %d", i);
                            return -1;
                        }
                        if (states[i]->exp_avg_sq &&
                            tensor_write_stream(states[i]->exp_avg_sq, file) != 0) {
                            LOG_ERROR("Failed to write Adam exp_avg_sq for parameter %d", i);
                            return -1;
                        }
                        if (states[i]->max_exp_avg_sq &&
                            tensor_write_stream(states[i]->max_exp_avg_sq, file) != 0) {
                            LOG_ERROR("Failed to write Adam max_exp_avg_sq for parameter %d", i);
                            return -1;
                        }
                    } else {
                        if (fwrite(&state_type, 1, 1, file) != 1) {
                            LOG_ERROR("Failed to write state type");
                            return -1;
                        }
                    }
                } else if (strcmp(optimizer->name, "RMSprop") == 0) {
                    state_type            = OPTIMIZER_STATE_RMSPROP;
                    RMSpropState** states = (RMSpropState**)group->state;
                    if (states[i] && states[i]->square_avg) {
                        if (fwrite(&state_type, 1, 1, file) != 1 ||
                            tensor_write_stream(states[i]->square_avg, file) != 0) {
                            LOG_ERROR("Failed to write RMSprop state for parameter %d", i);
                            return -1;
                        }
                    } else {
                        if (fwrite(&state_type, 1, 1, file) != 1) {
                            LOG_ERROR("Failed to write state type");
                            return -1;
                        }
                    }
                } else if (strcmp(optimizer->name, "Adagrad") == 0) {
                    state_type            = OPTIMIZER_STATE_ADAGRAD;
                    AdagradState** states = (AdagradState**)group->state;
                    if (states[i] && states[i]->sum_sq_grad) {
                        if (fwrite(&state_type, 1, 1, file) != 1 ||
                            tensor_write_stream(states[i]->sum_sq_grad, file) != 0) {
                            LOG_ERROR("Failed to write Adagrad state for parameter %d", i);
                            return -1;
                        }
                    } else {
                        if (fwrite(&state_type, 1, 1, file) != 1) {
                            LOG_ERROR("Failed to write state type");
                            return -1;
                        }
                    }
                }
            } else {
                if (fwrite(&state_type, 1, 1, file) != 1) {
                    LOG_ERROR("Failed to write state type");
                    return -1;
                }
            }
        }
    }

    return 0;
}

int optimizer_save(Optimizer* optimizer, const char* filepath) {
    if (!optimizer || !filepath) {
        LOG_ERROR("Invalid arguments to optimizer_save");
        return -1;
    }

    FILE* file = fopen(filepath, "wb");
    if (!file) {
        LOG_ERROR("Failed to open file for writing: %s", filepath);
        return -1;
    }

    int result = optimizer_save_stream(optimizer, file);
    fclose(file);

    if (result == 0) {
    }

    return result;
}

int optimizer_load_stream(Optimizer* optimizer, FILE* file) {
    if (!optimizer || !file) {
        LOG_ERROR("Invalid arguments to optimizer_load_stream");
        return -1;
    }
    char magic[5] = {0};
    if (fread(magic, 1, 4, file) != 4) {
        LOG_ERROR("Failed to read magic number");
        return -1;
    }

    if (strncmp(magic, OPTIMIZER_MAGIC, 4) != 0) {
        LOG_ERROR("Invalid magic number: expected %s, got %.4s", OPTIMIZER_MAGIC, magic);
        return -1;
    }
    uint8_t version;
    if (fread(&version, 1, 1, file) != 1) {
        LOG_ERROR("Failed to read version");
        return -1;
    }

    if (version != OPTIMIZER_VERSION) {
        LOG_ERROR("Unsupported version: %d (expected %d)", version, OPTIMIZER_VERSION);
        return -1;
    }
    int32_t name_len_int;
    if (fread(&name_len_int, sizeof(int32_t), 1, file) != 1) {
        LOG_ERROR("Failed to read name length");
        return -1;
    }
    int name_len = (int)name_len_int;
    if (name_len > 0) {
        char* saved_name = malloc((size_t)name_len + 1);
        if (!saved_name) {
            LOG_ERROR("Failed to allocate name buffer");
            return -1;
        }

        if (fread(saved_name, 1, (size_t)name_len, file) != (size_t)name_len) {
            LOG_ERROR("Failed to read optimizer name");
            free(saved_name);
            return -1;
        }
        saved_name[name_len] = '\0';

        if (optimizer->name && strcmp(optimizer->name, saved_name) != 0) {
            LOG_WARNING("Optimizer name mismatch: saved=%s, current=%s", saved_name,
                        optimizer->name);
        }

        free(saved_name);
    }
    int32_t num_groups_int;
    if (fread(&num_groups_int, sizeof(int32_t), 1, file) != 1) {
        LOG_ERROR("Failed to read number of parameter groups");
        return -1;
    }
    int num_groups = (int)num_groups_int;

    if (num_groups != optimizer->num_param_groups) {
        LOG_WARNING("Parameter group count mismatch: saved=%d, current=%d", num_groups,
                    optimizer->num_param_groups);
    }
    int groups_to_load =
        num_groups < optimizer->num_param_groups ? num_groups : optimizer->num_param_groups;
    for (int g = 0; g < groups_to_load; g++) {
        ParameterGroup* group = &optimizer->param_groups[g];
        float saved_lr, saved_weight_decay, saved_momentum, saved_beta1, saved_beta2, saved_epsilon;
        if (fread(&saved_lr, sizeof(float), 1, file) != 1 ||
            fread(&saved_weight_decay, sizeof(float), 1, file) != 1 ||
            fread(&saved_momentum, sizeof(float), 1, file) != 1 ||
            fread(&saved_beta1, sizeof(float), 1, file) != 1 ||
            fread(&saved_beta2, sizeof(float), 1, file) != 1 ||
            fread(&saved_epsilon, sizeof(float), 1, file) != 1) {
            LOG_ERROR("Failed to read hyperparameters for group %d", g);
            return -1;
        }
        group->lr           = saved_lr;
        group->weight_decay = saved_weight_decay;
        group->momentum     = saved_momentum;
        group->beta1        = saved_beta1;
        group->beta2        = saved_beta2;
        group->epsilon      = saved_epsilon;
        int32_t step_count_int;
        if (fread(&step_count_int, sizeof(int32_t), 1, file) != 1) {
            LOG_ERROR("Failed to read step count for group %d", g);
            return -1;
        }
        group->step_count = (int)step_count_int;
        int32_t num_params_int;
        if (fread(&num_params_int, sizeof(int32_t), 1, file) != 1) {
            LOG_ERROR("Failed to read number of parameters for group %d", g);
            return -1;
        }
        int num_params = (int)num_params_int;

        if (num_params != group->num_parameters) {
            LOG_WARNING("Parameter count mismatch for group %d: saved=%d, current=%d", g,
                        num_params, group->num_parameters);
        }
        for (int i = 0; i < num_params; i++) {
            int32_t param_idx;
            if (fread(&param_idx, sizeof(int32_t), 1, file) != 1) {
                LOG_ERROR("Failed to read parameter index");
                return -1;
            }
            uint8_t state_type;
            if (fread(&state_type, 1, 1, file) != 1) {
                LOG_ERROR("Failed to read state type");
                return -1;
            }
            if (param_idx >= 0 && param_idx < group->num_parameters) {
                if (state_type == OPTIMIZER_STATE_SGD && strcmp(optimizer->name, "SGD") == 0) {
                    if (!group->state) {
                        SGDMomentumState** states =
                            malloc((size_t)group->num_parameters * sizeof(SGDMomentumState*));
                        if (!states) {
                            LOG_ERROR("Failed to allocate SGD state");
                            return -1;
                        }
                        memset(states, 0,
                               (size_t)group->num_parameters * sizeof(SGDMomentumState*));
                        group->state = states;
                    }

                    SGDMomentumState** states = (SGDMomentumState**)group->state;
                    if (!states[param_idx]) {
                        states[param_idx] = malloc(sizeof(SGDMomentumState));
                        if (!states[param_idx]) {
                            LOG_ERROR("Failed to allocate SGD state for parameter %d", param_idx);
                            return -1;
                        }
                        states[param_idx]->momentum_buffer = NULL;
                    }

                    Tensor* momentum_buffer = tensor_read_stream(file);
                    if (momentum_buffer) {
                        if (states[param_idx]->momentum_buffer) {
                            tensor_free(states[param_idx]->momentum_buffer);
                        }
                        states[param_idx]->momentum_buffer = momentum_buffer;
                    }
                } else if (state_type == OPTIMIZER_STATE_ADAM &&
                           strcmp(optimizer->name, "Adam") == 0) {
                    if (!group->state) {
                        AdamState** states =
                            malloc((size_t)group->num_parameters * sizeof(AdamState*));
                        if (!states) {
                            LOG_ERROR("Failed to allocate Adam state");
                            return -1;
                        }
                        memset(states, 0, (size_t)group->num_parameters * sizeof(AdamState*));
                        group->state = states;
                    }

                    AdamState** states = (AdamState**)group->state;
                    if (!states[param_idx]) {
                        states[param_idx] = malloc(sizeof(AdamState));
                        if (!states[param_idx]) {
                            LOG_ERROR("Failed to allocate Adam state for parameter %d", param_idx);
                            return -1;
                        }
                        states[param_idx]->exp_avg        = NULL;
                        states[param_idx]->exp_avg_sq     = NULL;
                        states[param_idx]->max_exp_avg_sq = NULL;
                    }

                    Tensor* exp_avg        = tensor_read_stream(file);
                    Tensor* exp_avg_sq     = tensor_read_stream(file);
                    Tensor* max_exp_avg_sq = NULL;
                    if (optimizer->amsgrad) {
                        max_exp_avg_sq = tensor_read_stream(file);
                    }

                    if (exp_avg) {
                        if (states[param_idx]->exp_avg) {
                            tensor_free(states[param_idx]->exp_avg);
                        }
                        states[param_idx]->exp_avg = exp_avg;
                    }
                    if (exp_avg_sq) {
                        if (states[param_idx]->exp_avg_sq) {
                            tensor_free(states[param_idx]->exp_avg_sq);
                        }
                        states[param_idx]->exp_avg_sq = exp_avg_sq;
                    }
                    if (max_exp_avg_sq) {
                        if (states[param_idx]->max_exp_avg_sq) {
                            tensor_free(states[param_idx]->max_exp_avg_sq);
                        }
                        states[param_idx]->max_exp_avg_sq = max_exp_avg_sq;
                    }
                } else if (state_type == OPTIMIZER_STATE_RMSPROP &&
                           strcmp(optimizer->name, "RMSprop") == 0) {
                    if (!group->state) {
                        RMSpropState** states =
                            malloc((size_t)group->num_parameters * sizeof(RMSpropState*));
                        if (!states) {
                            LOG_ERROR("Failed to allocate RMSprop state");
                            return -1;
                        }
                        memset(states, 0, (size_t)group->num_parameters * sizeof(RMSpropState*));
                        group->state = states;
                    }

                    RMSpropState** states = (RMSpropState**)group->state;
                    if (!states[param_idx]) {
                        states[param_idx] = malloc(sizeof(RMSpropState));
                        if (!states[param_idx]) {
                            LOG_ERROR("Failed to allocate RMSprop state for parameter %d",
                                      param_idx);
                            return -1;
                        }
                        states[param_idx]->square_avg = NULL;
                    }

                    Tensor* square_avg = tensor_read_stream(file);
                    if (square_avg) {
                        if (states[param_idx]->square_avg) {
                            tensor_free(states[param_idx]->square_avg);
                        }
                        states[param_idx]->square_avg = square_avg;
                    }
                } else if (state_type == OPTIMIZER_STATE_ADAGRAD &&
                           strcmp(optimizer->name, "Adagrad") == 0) {
                    if (!group->state) {
                        AdagradState** states =
                            malloc((size_t)group->num_parameters * sizeof(AdagradState*));
                        if (!states) {
                            LOG_ERROR("Failed to allocate Adagrad state");
                            return -1;
                        }
                        memset(states, 0, (size_t)group->num_parameters * sizeof(AdagradState*));
                        group->state = states;
                    }

                    AdagradState** states = (AdagradState**)group->state;
                    if (!states[param_idx]) {
                        states[param_idx] = malloc(sizeof(AdagradState));
                        if (!states[param_idx]) {
                            LOG_ERROR("Failed to allocate Adagrad state for parameter %d",
                                      param_idx);
                            return -1;
                        }
                        states[param_idx]->sum_sq_grad = NULL;
                    }

                    Tensor* sum_sq_grad = tensor_read_stream(file);
                    if (sum_sq_grad) {
                        if (states[param_idx]->sum_sq_grad) {
                            tensor_free(states[param_idx]->sum_sq_grad);
                        }
                        states[param_idx]->sum_sq_grad = sum_sq_grad;
                    }
                } else {
                    LOG_WARNING("Unknown state type %d for parameter %d, skipping", state_type,
                                param_idx);
                }
            } else {
                LOG_WARNING("Invalid parameter index %d, skipping state", param_idx);
            }
        }
    }

    return 0;
}

int optimizer_load(Optimizer* optimizer, const char* filepath) {
    if (!optimizer || !filepath) {
        LOG_ERROR("Invalid arguments to optimizer_load");
        return -1;
    }

    FILE* file = fopen(filepath, "rb");
    if (!file) {
        LOG_ERROR("Failed to open file for reading: %s", filepath);
        return -1;
    }

    int result = optimizer_load_stream(optimizer, file);
    fclose(file);

    if (result == 0) {
    }

    return result;
}
