# Layers


## Dense Layer
- **Description**: Fully connected layer where each input is connected to each output.
- **Function**: `dense(float *input, float *weights, float *bias, int input_size, int output_size)`
- **File**: [`dense.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Layers/dense.c)

## Dropout Layer
- **Description**: Randomly sets a fraction of input units to zero during training to prevent overfitting.
- **Function**: `dropout(float *input, float *output, float dropout_rate, int size)`
- **File**: [`dropout.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Layers/dropout.c)
