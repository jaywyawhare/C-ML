/**
 * @file model_architecture.h
 * @brief Model architecture visualization export for C-ML
 *
 * This module extracts model architecture information from C-ML models
 * and exports it in a structured format for visualization.
 */

#ifndef CML_MODEL_ARCHITECTURE_H
#define CML_MODEL_ARCHITECTURE_H

#include "nn/module.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Layer information structure
 */
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

/**
 * @brief Model architecture structure
 */
typedef struct {
    LayerInfo* layers;    // Array of layer information
    size_t num_layers;    // Number of layers
    size_t capacity;      // Capacity of layers array
    int total_params;     // Total parameters
    int trainable_params; // Trainable parameters
} ModelArchitecture;

/**
 * @brief Create a new model architecture structure
 */
ModelArchitecture* model_architecture_create(void);

/**
 * @brief Extract architecture from a Module
 *
 * Traverses the module structure (handles Sequential, Linear, etc.)
 * and extracts layer information.
 *
 * @param module The module to extract architecture from
 * @param arch Architecture structure to populate
 * @return 0 on success, negative value on failure
 */
int model_architecture_extract(Module* module, ModelArchitecture* arch);

/**
 * @brief Export model architecture to JSON
 *
 * Exports the architecture in a format suitable for visualization:
 * - Layer types and properties
 * - Connections between layers
 * - Parameter information
 *
 * @param arch Architecture to export
 * @param path Output file path
 * @return 0 on success, negative value on failure
 */
int model_architecture_export_json(const ModelArchitecture* arch, const char* path);

/**
 * @brief Free model architecture structure
 */
void model_architecture_free(ModelArchitecture* arch);

#ifdef __cplusplus
}
#endif

#endif // CML_MODEL_ARCHITECTURE_H
