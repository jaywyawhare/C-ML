/**
 * @file cml.c
 * @brief C-ML Library main implementation
 *
 * This file contains the main library initialization, cleanup,
 * and utility functions for the C-ML library.
 */

#include "cml.h"
#include "Core/logging.h"
#include "Core/memory_management.h"
#include "nn/module.h"
#include "nn/layers.h"
#include "nn/layers/sequential.h"
#include "nn/layers/linear.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Global library state
static bool g_cml_initialized = false;
static int g_cml_init_count   = 0;

// Build information
static const char* g_build_info = "C-ML Library v" CML_VERSION_STRING "\n"
                                  "Built with: GCC " __VERSION__ "\n"
                                  "Build date: " __DATE__ " " __TIME__ "\n"
                                  "Features: autograd, nn, optim, logging, memory_management";

/**
 * @brief Initialize the C-ML library
 *
 * This function should be called before using any C-ML functionality.
 * It initializes internal systems, sets up logging, and prepares
 * the library for use.
 *
 * @return 0 on success, negative value on failure
 */
int cml_init(void) {
    if (g_cml_initialized) {
        g_cml_init_count++;
        LOG_DEBUG("C-ML library already initialized, reference count: %d", g_cml_init_count);
        return 0;
    }

    LOG_INFO("Initializing C-ML Library v%s", CML_VERSION_STRING);

    int result = 0;

    set_log_level(LOG_LEVEL_ERROR);

    srand((unsigned int)time(NULL));

    extern void autograd_init(void);
    autograd_init();

    if (result == 0) {
        g_cml_initialized = true;
        g_cml_init_count  = 1;
        LOG_INFO("C-ML Library initialized successfully");
    } else {
        LOG_ERROR("Failed to initialize C-ML Library");
    }

    return result;
}

/**
 * @brief Cleanup the C-ML library
 *
 * This function should be called when the library is no longer needed.
 * It cleans up internal resources and ensures proper shutdown.
 *
 * @return 0 on success, negative value on failure
 */
int cml_cleanup(void) {
    if (!g_cml_initialized) {
        LOG_WARNING("C-ML library not initialized, nothing to cleanup");
        return 0;
    }

    g_cml_init_count--;

    if (g_cml_init_count > 0) {
        LOG_DEBUG("C-ML library cleanup requested, but reference count > 0: %d", g_cml_init_count);
        return 0;
    }

    LOG_INFO("Cleaning up C-ML Library");

    extern void autograd_shutdown(void);
    autograd_shutdown();

    g_cml_initialized = false;
    g_cml_init_count  = 0;

    LOG_INFO("C-ML Library cleanup completed");
    return 0;
}

/**
 * @brief Get library version information
 *
 * @param major Pointer to store major version
 * @param minor Pointer to store minor version
 * @param patch Pointer to store patch version
 * @param version_string Pointer to store version string
 */
void cml_get_version(int* major, int* minor, int* patch, const char** version_string) {
    if (major)
        *major = CML_VERSION_MAJOR;
    if (minor)
        *minor = CML_VERSION_MINOR;
    if (patch)
        *patch = CML_VERSION_PATCH;
    if (version_string)
        *version_string = CML_VERSION_STRING;
}

/**
 * @brief Get library build information
 *
 * @return String containing build information (compiler, flags, etc.)
 */
const char* cml_get_build_info(void) { return g_build_info; }

/**
 * @brief Check if C-ML library is initialized
 *
 * @return true if initialized, false otherwise
 */
bool cml_is_initialized(void) { return g_cml_initialized; }

/**
 * @brief Get C-ML library initialization count
 *
 * @return Current initialization reference count
 */
int cml_get_init_count(void) { return g_cml_init_count; }

/**
 * @brief Force cleanup of C-ML library (ignores reference count)
 *
 * This function should be used with caution as it forces cleanup
 * regardless of the reference count.
 *
 * @return 0 on success, negative value on failure
 */
int cml_force_cleanup(void) {
    if (!g_cml_initialized) {
        return 0;
    }

    LOG_WARNING("Forcing C-ML library cleanup (reference count was %d)", g_cml_init_count);

    g_cml_init_count = 0;
    return cml_cleanup();
}

/**
 * @brief Print model summary (TensorFlow/Keras style)
 *
 * Prints a summary of the model architecture, showing layers,
 * parameter counts, and total trainable parameters.
 *
 * @param module The module to summarize (can be Sequential, Linear, etc.)
 */
static void print_layer_summary(Module* module, int indent, int* layer_num) {
    if (!module)
        return;

    // If Sequential, skip printing it and just recurse into children
    if (strcmp(module->name, "Sequential") == 0) {
        Sequential* seq = (Sequential*)module;
        int num_modules = sequential_get_length(seq);
        for (int i = 0; i < num_modules; i++) {
            Module* child = sequential_get(seq, i);
            if (child) {
                print_layer_summary(child, indent, layer_num);
            }
        }
        return;
    }

    // Get layer type name
    const char* layer_type = module->name;

    // Count parameters for this layer
    int layer_params = 0;
    for (int i = 0; i < module->num_parameters; i++) {
        if (module->parameters[i] && module->parameters[i]->tensor) {
            layer_params += (int)module->parameters[i]->tensor->numel;
        }
    }

    // Build layer description
    char layer_desc[64] = {0};
    if (strcmp(module->name, "Linear") == 0) {
        // Get Linear-specific info
        Linear* linear   = (Linear*)module;
        int in_features  = linear_get_in_features(linear);
        int out_features = linear_get_out_features(linear);
        bool use_bias    = linear_get_use_bias(linear);
        snprintf(layer_desc, sizeof(layer_desc), "%s (%d->%d, bias=%s)", layer_type, in_features,
                 out_features, use_bias ? "True" : "False");
    } else {
        // Other layers (ReLU, Tanh, Sigmoid, etc.)
        snprintf(layer_desc, sizeof(layer_desc), "%s", layer_type);
    }

    // Print layer info - format: "LayerNum Type Description    Parameters"
    // Indentation for nested modules (if any)
    if (indent > 0) {
        for (int i = 0; i < indent; i++)
            printf("  ");
    }
    printf("%-5d %-35s %15d\n", (*layer_num)++, layer_desc, layer_params);
}

void summary(Module* module) {
    if (!module) {
        printf("Model Summary: (empty)\n");
        return;
    }

    printf("\n");
    for (int i = 0; i < 60; i++)
        printf("=");
    printf("\nModel Summary\n");
    for (int i = 0; i < 60; i++)
        printf("=");
    printf("\n");

    // Count total parameters
    Parameter** params  = NULL;
    int num_params      = 0;
    int total_trainable = 0;

    if (module_collect_parameters(module, &params, &num_params, true) == 0) {
        for (int i = 0; i < num_params; i++) {
            if (params[i] && params[i]->tensor && params[i]->requires_grad) {
                total_trainable += (int)params[i]->tensor->numel;
            }
        }
        if (params)
            CM_FREE(params);
    }

    // Print layers header (TensorFlow/Keras style)
    printf("%-5s %-35s %15s\n", "Layer", "Type", "Parameters");
    for (int i = 0; i < 60; i++)
        printf("-");
    printf("\n");

    int layer_num = 1;
    print_layer_summary(module, 0, &layer_num);

    for (int i = 0; i < 60; i++)
        printf("-");
    printf("\n");
    printf("Total params: %d\n", total_trainable);
    printf("Trainable params: %d\n", total_trainable);
    printf("Non-trainable params: 0\n");
    for (int i = 0; i < 60; i++)
        printf("=");
    printf("\n\n");
}
