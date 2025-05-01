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
