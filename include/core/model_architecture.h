#ifndef CML_MODEL_ARCHITECTURE_H
#define CML_MODEL_ARCHITECTURE_H

#include "nn.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char* type; // Layer type (e.g., "Linear", "ReLU", "Conv2d")
    int layer_index;  // Index in the model
    int in_features;  // Input features (for Linear/Conv layers)
    int out_features; // Output features (for Linear/Conv layers)
    int in_channels;  // Input channels (for Conv2d)
    int out_channels; // Output channels (for Conv2d)
    int kernel_size;  // Kernel size (for Conv2d/Pooling)
    int stride;       // Stride (for Conv2d/Pooling)
    int padding;      // Padding (for Conv2d)
    bool has_bias;    // Whether layer has bias
    int num_params;   // Number of parameters in this layer
    char* details;    // Additional layer details (JSON string)
} LayerInfo;

typedef struct {
    LayerInfo* layers;    // Array of layer information
    size_t num_layers;    // Number of layers
    size_t capacity;      // Capacity of layers array
    int total_params;     // Total parameters
    int trainable_params; // Trainable parameters
} ModelArchitecture;

ModelArchitecture* model_architecture_create(void);
int model_architecture_extract(Module* module, ModelArchitecture* arch);
int model_architecture_export_json(const ModelArchitecture* arch, const char* path);
void model_architecture_free(ModelArchitecture* arch);

#ifdef __cplusplus
}
#endif

#endif // CML_MODEL_ARCHITECTURE_H
