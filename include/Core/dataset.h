#ifndef DATASET_H
#define DATASET_H

typedef struct
{
    float **X;       // Input data
    float **y;       // Target data
    int num_samples; // Number of samples
    int input_size;  // Size of input features
    int output_size; // Size of output/target
} Dataset;

// Dataset creation and management functions
Dataset *create_dataset(int num_samples, int input_size, int output_size);
void free_dataset(Dataset *dataset);

#endif
