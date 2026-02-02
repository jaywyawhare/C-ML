/**
 * @file serialization.h
 * @brief Serialization for tensors, models, and optimizers
 *
 * Provides:
 * - Model checkpointing (save/load weights)
 * - Tensor serialization
 * - Optimizer state serialization
 */

#ifndef CML_CORE_SERIALIZATION_H
#define CML_CORE_SERIALIZATION_H

#include "nn.h"
#include "tensor/tensor.h"
#include "optim.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Save model weights to file
 *
 * @param module Module to save
 * @param filepath Path to save file
 * @return 0 on success, negative value on failure
 */
int module_save(Module* module, const char* filepath);

/**
 * @brief Load model weights from file
 *
 * @param module Module to load weights into
 * @param filepath Path to load file
 * @return 0 on success, negative value on failure
 */
int module_load(Module* module, const char* filepath);

/**
 * @brief Save model to file stream
 *
 * @param module Module to save
 * @param file File stream to write to
 * @return 0 on success, negative value on failure
 */
int module_save_stream(Module* module, FILE* file);

/**
 * @brief Load model from file stream
 *
 * @param module Module to load weights into
 * @param file File stream to read from
 * @return 0 on success, negative value on failure
 */
int module_load_stream(Module* module, FILE* file);

/**
 * @brief Write tensor to file
 *
 * Saves dtype, shape, device, and raw data.
 *
 * @param tensor Tensor to save
 * @param filepath Path to save file
 * @return 0 on success, negative value on failure
 */
int tensor_write_file(Tensor* tensor, const char* filepath);

/**
 * @brief Read tensor from file
 *
 * @param filepath Path to load file
 * @return Tensor, or NULL on failure
 */
Tensor* tensor_read_file(const char* filepath);

/**
 * @brief Write tensor to file stream
 *
 * @param tensor Tensor to save
 * @param file File stream to write to
 * @return 0 on success, negative value on failure
 */
int tensor_write_stream(Tensor* tensor, FILE* file);

/**
 * @brief Read tensor from file stream
 *
 * @param file File stream to read from
 * @return Tensor, or NULL on failure
 */
Tensor* tensor_read_stream(FILE* file);

/**
 * @brief Save optimizer state to file
 *
 * @param optimizer Optimizer to save
 * @param filepath Path to save file
 * @return 0 on success, negative value on failure
 */
int optimizer_save(Optimizer* optimizer, const char* filepath);

/**
 * @brief Load optimizer state from file
 *
 * @param optimizer Optimizer to load state into
 * @param filepath Path to load file
 * @return 0 on success, negative value on failure
 */
int optimizer_load(Optimizer* optimizer, const char* filepath);

/**
 * @brief Save optimizer state to file stream
 *
 * @param optimizer Optimizer to save
 * @param file File stream to write to
 * @return 0 on success, negative value on failure
 */
int optimizer_save_stream(Optimizer* optimizer, FILE* file);

/**
 * @brief Load optimizer state from file stream
 *
 * @param optimizer Optimizer to load state into
 * @param file File stream to read from
 * @return 0 on success, negative value on failure
 */
int optimizer_load_stream(Optimizer* optimizer, FILE* file);

/**
 * @brief Named parameter structure
 */
typedef struct NamedParameter {
    char* name; // Non-const so we can allocate and free
    Parameter* parameter;
} NamedParameter;

/**
 * @brief Get named parameters from module
 *
 * @param module Module to get parameters from
 * @param named_params Output array of named parameters (caller must free)
 * @param num_params Output number of parameters
 * @return 0 on success, negative value on failure
 */
int module_named_parameters(Module* module, NamedParameter** named_params, int* num_params);

/**
 * @brief Free named parameters array
 *
 * @param named_params Array of named parameters
 * @param num_params Number of parameters
 */
void module_named_parameters_free(NamedParameter* named_params, int num_params);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_SERIALIZATION_H
