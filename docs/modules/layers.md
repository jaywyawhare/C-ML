# Layers

## Dense Layer
- **Description**: Fully connected layer where each input is connected to each output.
- **Function**: 
    - `initialize_dense(DenseLayer *layer, int input_size, int output_size)`
    - `forward_dense(DenseLayer *layer, float *input, float *output)`
    - `backward_dense(DenseLayer *layer, float *input, float *output, float *d_output, float *d_input, float *d_weights, float *d_biases)`
    - `update_dense(DenseLayer *layer, float *d_weights, float *d_biases, float learning_rate)`
    - `free_dense(DenseLayer *layer)`
- **File**: [`dense.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Layers/dense.c)

## Dropout Layer
- **Description**: Randomly sets a fraction of input units to zero during training to prevent overfitting.
- **Function**: 
    - `initialize_dropout(DropoutLayer *layer, float dropout_rate)`
    - `forward_dropout(DropoutLayer *layer, float *input, float *output, int size)`
    - `backward_dropout(DropoutLayer *layer, float *input, float *output, float *d_output, float *d_input, int size)`
- **File**: [`dropout.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Layers/dropout.c)

## Flatten Layer
- **Description**: Flattens the input without affecting the batch size.
- **Function**: 
    - `initializeFlatten(FlattenLayer *layer, int input_size)`
    - `forwardFlatten(FlattenLayer *layer, float *input, float *output)`
    - `backwardFlatten(FlattenLayer *layer, float *input, float *output, float *d_output, float *d_input)`
    - `freeFlatten(FlattenLayer *layer)`
- **File**: [`flatten.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Layers/flatten.c)

## Pooling Layer
- **Description**: Reduces the spatial size of the input volume.
- **Function**: 
    - `initialize_polling(PollingLayer *layer, int kernel_size, int stride)`
    - `compute_polling_output_size(int input_size, int kernel_size, int stride)`
    - `forward_polling(PollingLayer *layer, const float *input, float *output, int input_size)`
    - `free_polling(PollingLayer *layer)`
- **File**: [`pooling.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Layers/pooling.c)

## Max-Pooling Layer
- **Description**: Applies max pooling operation to the input.
- **Function**: 
    - `initialize_maxpooling(MaxPoolingLayer *layer, int kernel_size, int stride)`
    - `compute_maxpooling_output_size(int input_size, int kernel_size, int stride)`
    - `forward_maxpooling(MaxPoolingLayer *layer, const float *input, float *output, int input_size)`
    - `free_maxpooling(MaxPoolingLayer *layer)`
- **File**: [`maxpooling.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Layers/maxpooling.c)

