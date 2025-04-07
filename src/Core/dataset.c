#include "../../include/Core/dataset.h"
#include "../../include/Core/memory_management.h"
#include "../../include/Core/error_codes.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Create a new dataset.
 *
 * Allocates memory for a new dataset structure and initializes its fields.
 *
 * @return Dataset* Pointer to the newly created dataset, or NULL on failure.
 */
Dataset *dataset_create(void)
{
    Dataset *dataset = (Dataset *)cm_safe_malloc(sizeof(Dataset), __FILE__, __LINE__);
    if (dataset)
    {
        dataset->X = NULL;
        dataset->y = NULL;
        dataset->num_samples = 0;
        dataset->input_dim = 0;
        dataset->output_dim = 0;
    }
    return dataset;
}

/**
 * @brief Load dataset from arrays.
 *
 * Copies input and output data arrays into the dataset structure.
 *
 * @param dataset Pointer to the dataset structure.
 * @param X_array Pointer to the input data array.
 * @param y_array Pointer to the output data array.
 * @param num_samples Number of samples.
 * @param input_dim Dimension of input features.
 * @param output_dim Dimension of output features.
 * @return CM_Error Error code indicating success or failure.
 */
CM_Error dataset_load_arrays(Dataset *dataset, float *X_array, float *y_array, int num_samples, int input_dim, int output_dim)
{
    if (!dataset || !X_array || !y_array)
    {
        fprintf(stderr, "[dataset_load_arrays] Error: Null pointer argument.\n");
        return CM_NULL_POINTER_ERROR;
    }

    dataset->num_samples = num_samples;
    dataset->input_dim = input_dim;
    dataset->output_dim = output_dim;

    dataset->X = (float **)cm_safe_malloc(num_samples * sizeof(float *), __FILE__, __LINE__);
    dataset->y = (float **)cm_safe_malloc(num_samples * sizeof(float *), __FILE__, __LINE__);

    if (!dataset->X || !dataset->y)
    {
        fprintf(stderr, "[dataset_load_arrays] Error: Memory allocation failed for X or y.\n");
        return CM_MEMORY_ALLOCATION_ERROR;
    }

    for (int i = 0; i < num_samples; i++)
    {
        dataset->X[i] = (float *)cm_safe_malloc(input_dim * sizeof(float), __FILE__, __LINE__);
        dataset->y[i] = (float *)cm_safe_malloc(output_dim * sizeof(float), __FILE__, __LINE__);

        if (!dataset->X[i] || !dataset->y[i])
        {
            LOG_ERROR("Memory allocation failed at index %d.", i);
            return CM_MEMORY_ALLOCATION_ERROR;
        }

        memcpy(dataset->X[i], X_array + i * input_dim, input_dim * sizeof(float));
        memcpy(dataset->y[i], y_array + i * output_dim, output_dim * sizeof(float));
    }
    return CM_SUCCESS;
}

/**
 * @brief Free memory allocated for the dataset.
 *
 * Releases all memory associated with the dataset, including input and output arrays.
 *
 * @param dataset Pointer to the dataset to free.
 */
void dataset_free(Dataset *dataset)
{
    if (!dataset)
        return;

    if (dataset->X)
    {
        for (int i = 0; i < dataset->num_samples; i++)
        {
            cm_safe_free((void **)&dataset->X[i]);
        }
        cm_safe_free((void **)&dataset->X);
    }

    if (dataset->y)
    {
        for (int i = 0; i < dataset->num_samples; i++)
        {
            cm_safe_free((void **)&dataset->y[i]);
        }
        cm_safe_free((void **)&dataset->y);
    }

    cm_safe_free((void **)&dataset);
}
