#include "../../include/Core/dataset.h"
#include "../../include/Core/memory_management.h"

Dataset *create_dataset(int num_samples, int input_size, int output_size)
{
    Dataset *dataset = cm_safe_malloc(sizeof(Dataset), __FILE__, __LINE__);

    dataset->num_samples = num_samples;
    dataset->input_size = input_size;
    dataset->output_size = output_size;

    // Allocate memory for input data
    dataset->X = cm_safe_malloc(num_samples * sizeof(float *), __FILE__, __LINE__);
    for (int i = 0; i < num_samples; i++)
    {
        dataset->X[i] = cm_safe_malloc(input_size * sizeof(float), __FILE__, __LINE__);
    }

    // Allocate memory for target data
    dataset->y = cm_safe_malloc(num_samples * sizeof(float *), __FILE__, __LINE__);
    for (int i = 0; i < num_samples; i++)
    {
        dataset->y[i] = cm_safe_malloc(output_size * sizeof(float), __FILE__, __LINE__);
    }

    return dataset;
}

void free_dataset(Dataset *dataset)
{
    if (!dataset)
        return;

    // Free input data
    for (int i = 0; i < dataset->num_samples; i++)
    {
        cm_safe_free((void **)&dataset->X[i]);
    }
    cm_safe_free((void **)&dataset->X);

    // Free target data
    for (int i = 0; i < dataset->num_samples; i++)
    {
        cm_safe_free((void **)&dataset->y[i]);
    }
    cm_safe_free((void **)&dataset->y);

    // Free dataset structure
    cm_safe_free((void **)&dataset);
}

// Convenience functions for main.c compatibility
Dataset *dataset_create(void)
{
    // Create an empty dataset that will be populated later
    Dataset *dataset = cm_safe_malloc(sizeof(Dataset), __FILE__, __LINE__);
    dataset->X = NULL;
    dataset->y = NULL;
    dataset->num_samples = 0;
    dataset->input_size = 0;
    dataset->output_size = 0;
    return dataset;
}

void dataset_load_arrays(Dataset *dataset, float *X_data, float *y_data, 
                        int num_samples, int input_size, int output_size)
{
    if (!dataset) return;
    
    dataset->num_samples = num_samples;
    dataset->input_size = input_size;
    dataset->output_size = output_size;

    // Allocate memory for input data
    dataset->X = cm_safe_malloc(num_samples * sizeof(float *), __FILE__, __LINE__);
    for (int i = 0; i < num_samples; i++)
    {
        dataset->X[i] = cm_safe_malloc(input_size * sizeof(float), __FILE__, __LINE__);
        // Copy data from the flat array
        for (int j = 0; j < input_size; j++)
        {
            dataset->X[i][j] = X_data[i * input_size + j];
        }
    }

    // Allocate memory for target data
    dataset->y = cm_safe_malloc(num_samples * sizeof(float *), __FILE__, __LINE__);
    for (int i = 0; i < num_samples; i++)
    {
        dataset->y[i] = cm_safe_malloc(output_size * sizeof(float), __FILE__, __LINE__);
        // Copy data from the flat array
        for (int j = 0; j < output_size; j++)
        {
            dataset->y[i][j] = y_data[i * output_size + j];
        }
    }
}

void dataset_free(Dataset *dataset)
{
    free_dataset(dataset);
}
