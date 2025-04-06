#ifndef C_ML_DATASET_H
#define C_ML_DATASET_H

#include <stddef.h>
#include "error_codes.h"

/**
 * @brief Structure representing a dataset.
 *
 * Contains input and output data arrays, along with metadata such as the number of samples
 * and the dimensions of input and output features.
 */
typedef struct Dataset
{
    float **X;       /**< Pointer to the input data array. */
    float **y;       /**< Pointer to the output data array. */
    int num_samples; /**< Number of samples in the dataset. */
    int input_dim;   /**< Dimension of input features. */
    int output_dim;  /**< Dimension of output features. */
} Dataset;

/**
 * @brief Create a new dataset.
 *
 * @return Dataset* Pointer to the newly created dataset.
 */
Dataset *dataset_create(void);

/**
 * @brief Free memory allocated for the dataset.
 *
 * @param dataset Pointer to the dataset to free.
 */
void dataset_free(Dataset *dataset);

/**
 * @brief Load dataset from arrays.
 *
 * @param dataset Pointer to the dataset structure.
 * @param X_array Pointer to the input data array.
 * @param y_array Pointer to the output data array.
 * @param num_samples Number of samples.
 * @param input_dim Dimension of input features.
 * @param output_dim Dimension of output features.
 * @return CM_Error Error code.
 */
CM_Error dataset_load_arrays(Dataset *dataset, float *X_array, float *y_array, int num_samples, int input_dim, int output_dim);

#endif
