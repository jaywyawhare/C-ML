# Layers

## Input Layer
- **Description**: Represents the input layer of the neural network.
- **Functions**: 
    - `initialize_input(InputLayer *layer, int input_size)`
    - `forward_input(InputLayer *layer, float *input, float *output)`
    - `backward_input(InputLayer *layer, float *input, float *output, float *d_output, float *d_input)`
    - `free_input(InputLayer *layer)`
- **File**: [`input.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Layers/input.c)

## Dense Layer
- **Description**: Fully connected layer where each input is connected to each output.
- **Functions**: 
    - `initialize_dense(DenseLayer *layer, int input_size, int output_size)`
    - `forward_dense(DenseLayer *layer, float *input, float *output)`
    - `backward_dense(DenseLayer *layer, float *input, float *output, float *d_output, float *d_input, float *d_weights, float *d_biases)`
    - `update_dense(DenseLayer *layer, float *d_weights, float *d_biases, float learning_rate, const char *optimizer_type, float beta1, float beta2, float epsilon)`
    - `free_dense(DenseLayer *layer)`
- **File**: [`dense.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Layers/dense.c)

## Dropout Layer
- **Description**: Randomly sets a fraction of input units to zero during training to prevent overfitting.
- **Functions**: 
    - `initialize_dropout(DropoutLayer *layer, float dropout_rate)`
    - `forward_dropout(DropoutLayer *layer, float *input, float *output, int size)`
    - `backward_dropout(DropoutLayer *layer, float *input, float *output, float *d_output, float *d_input, int size)`
    - `free_dropout(DropoutLayer *layer)`
- **File**: [`dropout.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Layers/dropout.c)

## Flatten Layer
- **Description**: Flattens the input without affecting the batch size.
- **Functions**: 
    - `initialize_flatten(FlattenLayer *layer, int input_size)`
    - `forward_flatten(FlattenLayer *layer, float *input, float *output)`
    - `backward_flatten(FlattenLayer *layer, float *input, float *output, float *d_output, float *d_input)`
    - `free_flatten(FlattenLayer *layer)`
- **File**: [`flatten.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Layers/flatten.c)

## Reshape Layer
- **Description**: Reshapes the input tensor to a specified shape.
- **Functions**: 
    - `initialize_reshape(ReshapeLayer *layer, int input_size, int output_size)`
    - `forward_reshape(ReshapeLayer *layer, float *input, float *output)`
    - `backward_reshape(ReshapeLayer *layer, float *d_output, float *d_input)`
    - `free_reshape(ReshapeLayer *layer)`
- **File**: [`reshape.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Layers/reshape.c)

## Pooling Layer
- **Description**: Reduces the spatial size of the input volume.
- **Functions**: 
    - `initialize_pooling(PoolingLayer *layer, int kernel_size, int stride)`
    - `compute_pooling_output_size(int input_size, int kernel_size, int stride)`
    - `forward_pooling(PoolingLayer *layer, const float *input, float *output, int input_size)`
    - `backward_pooling(PoolingLayer *layer, const float *input, const float *output, const float *d_output, float *d_input, int input_size)`
    - `free_pooling(PoolingLayer *layer)`
- **File**: [`pooling.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Layers/pooling.c)

## Max-Pooling Layer
- **Description**: Applies max pooling operation to the input.
- **Functions**: 
    - `initialize_maxpooling(MaxPoolingLayer *layer, int kernel_size, int stride)`
    - `compute_maxpooling_output_size(int input_size, int kernel_size, int stride)`
    - `forward_maxpooling(MaxPoolingLayer *layer, const float *input, float *output, int input_size)`
    - `backward_maxpooling(MaxPoolingLayer *layer, const float *input, const float *output, const float *d_output, float *d_input, int input_size)`
    - `free_maxpooling(MaxPoolingLayer *layer)`
    - `validate_maxpooling_params(const int kernel_size, const int stride)`
- **File**: [`maxpooling.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Layers/maxpooling.c)

## Conv1D Layer
- **Description**: Implements 1D convolution operation.
- **Functions**: 
    - `initialize_conv1d(Conv1DLayer *layer, const int input_channels, const int output_channels, const int kernel_size, const int input_length, const int padding, const int stride, const int dilation)`
    - `forward_conv1d(const Conv1DLayer *layer, const float *input, float *output)`
    - `backward_conv1d(const Conv1DLayer *layer, const float *input, const float *output, const float *d_output, float *d_input)`
    - `update_conv1d(Conv1DLayer *layer, float *d_weights, float *d_biases, float learning_rate, const char *optimizer_type, float beta1, float beta2, float epsilon)`
    - `free_conv1d(Conv1DLayer *layer)`
- **File**: [`conv1d.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Layers/conv1d.c)

## Conv1D Transpose Layer
- **Description**: Implements 1D transposed convolution operation.
- **Functions**: 
    - `initialize_conv1d_transpose(Conv1DTransposeLayer *layer, int input_channels, int output_channels, int kernel_size, int input_length, int padding, int stride, int dilation)`
    - `forward_conv1d_transpose(Conv1DTransposeLayer *layer, float *input, float *output)`
    - `backward_conv1d_transpose(Conv1DTransposeLayer *layer, float *input, float *d_output, float *d_input)`
    - `update_conv1d_transpose(Conv1DTransposeLayer *layer, float *d_weights, float *d_biases, float learning_rate, const char *optimizer_type, float beta1, float beta2, float epsilon)`
    - `free_conv1d_transpose(Conv1DTransposeLayer *layer)`
- **File**: [`conv1d_transpose.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Layers/conv1d_transpose.c)

## Conv2D Layer
- **Description**: Implements 2D convolution operation.
- **Functions**: 
    - `initialize_conv2d(Conv2DLayer *layer, const int input_channels, const int output_channels, const int kernel_size, const int input_height, const int input_width, const int padding, const int stride, const int dilation)`
    - `forward_conv2d(Conv2DLayer *layer, const float *input, float *output)`
    - `backward_conv2d(Conv2DLayer *layer, const float *input, const float *output, const float *d_output, float *d_input)`
    - `update_conv2d(Conv2DLayer *layer, float *d_weights, float *d_biases, float learning_rate, const char *optimizer_type, float beta1, float beta2, float epsilon)`
    - `free_conv2d(Conv2DLayer *layer)`
- **File**: [`conv2d.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Layers/conv2d.c)

## Conv2D Transpose Layer
- **Description**: Implements 2D transposed convolution operation.
- **Functions**: 
    - `initialize_conv2d_transpose(Conv2DTransposeLayer *layer, int input_channels, int output_channels, int kernel_size, int input_height, int input_width, int padding, int stride, int dilation)`
    - `forward_conv2d_transpose(Conv2DTransposeLayer *layer, float *input, float *output)`
    - `backward_conv2d_transpose(Conv2DTransposeLayer *layer, float *input, float *output, float *d_output, float *d_input)`
    - `update_conv2d_transpose(Conv2DTransposeLayer *layer, float *d_weights, float *d_biases, float learning_rate)`
    - `free_conv2d_transpose(Conv2DTransposeLayer *layer)`
- **File**: [`conv2d_transpose.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Layers/conv2d_transpose.c)

## BatchNorm Layer
- **Description**: Normalizes the input to improve training stability.
- **Functions**: 
    - `initialize_batchnorm(BatchNormLayer *layer, int num_features)`
    - `forward_batchnorm(BatchNormLayer *layer, float *input, float *output, float *mean, float *variance)`
    - `backward_batchnorm(BatchNormLayer *layer, float *input, float *d_output, float *d_input, float *d_gamma, float *d_beta, float *mean, float *variance)`
    - `update_batchnorm(BatchNormLayer *layer, float *d_gamma, float *d_beta, float learning_rate)`
    - `free_batchnorm(BatchNormLayer *layer)`
    - `save_batchnorm_params(BatchNormLayer *layer, const char *filename)`
    - `load_batchnorm_params(BatchNormLayer *layer, const char *filename)`
- **File**: [`batchnorm.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Layers/batchnorm.c)

## Embedding Layer
- **Description**: Converts categorical data into dense vector representations.
- **Functions**: 
    - `initialize_embedding(EmbeddingLayer *layer, int vocab_size, int embedding_dim)`
    - `forward_embedding(EmbeddingLayer *layer, const int *input, float *output, int input_size)`
    - `backward_embedding(EmbeddingLayer *layer, const int *input, float *d_output, float *d_weights, int input_size)`
    - `update_embedding(EmbeddingLayer *layer, float *d_weights, float learning_rate)`
    - `free_embedding(EmbeddingLayer *layer)`
    - `save_embedding_weights(EmbeddingLayer *layer, const char *filename)`
    - `load_embedding_weights(EmbeddingLayer *layer, const char *filename)`
- **File**: [`embedding.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Layers/embedding.c)

## LSTM Layer
- **Description**: Long Short-Term Memory layer for sequential data.
- **Functions**: 
    - `initialize_lstm(LSTMLayer *layer, int input_size, int hidden_size)`
    - `forward_lstm(LSTMLayer *layer, float *input, float *output)`
    - `backward_lstm(LSTMLayer *layer, float *input, float *output, float *d_output, float *d_input)`
    - `update_lstm(LSTMLayer *layer, float *d_weights_input, float *d_weights_hidden, float *d_biases, float learning_rate)`
    - `reset_state_lstm(LSTMLayer *layer)`
    - `free_lstm(LSTMLayer *layer)`
- **File**: [`lstm.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Layers/lstm.c)

## GRU Layer
- **Description**: Gated Recurrent Unit layer for sequential data.
- **Functions**: 
    - `initialize_gru(GRULayer *layer, int input_size, int hidden_size)`
    - `forward_gru(GRULayer *layer, float *input, float *output)`
    - `backward_gru(GRULayer *layer, float *input, float *output, float *d_output, float *d_input)`
    - `update_gru(GRULayer *layer, float learning_rate)`
    - `reset_state_gru(GRULayer *layer)`
    - `free_gru(GRULayer *layer)`
- **File**: [`gru.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Layers/gru.c)

## Bidirectional LSTM Layer
- **Description**: Combines forward and backward LSTM layers to capture context from both directions in sequential data.
- **Functions**: 
    - `initialize_bidirectional_lstm(BidirectionalLSTMLayer *layer, int input_size, int hidden_size)`
    - `forward_bidirectional_lstm(BidirectionalLSTMLayer *layer, float *input, float *output, int input_size, int output_size)`
    - `backward_bidirectional_lstm(BidirectionalLSTMLayer *layer, float *input, float *d_output, float *d_input, int input_size, int output_size)`
    - `reset_state_bidirectional_lstm(BidirectionalLSTMLayer *layer)`
    - `free_bidirectional_lstm(BidirectionalLSTMLayer *layer)`
- **File**: [`bidirectional_lstm.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Layers/bidirectional_lstm.c)

## Attention Layer
- **Description**: Implements attention mechanism for sequence-to-sequence models.
- **Functions**: 
    - `initialize_attention(AttentionLayer *layer, int query_dim, int key_dim, int value_dim)`
    - `forward_attention(AttentionLayer *layer, float *query, float *key, float *value, float *output, const char *optimizer_type)`
    - `backward_attention(AttentionLayer *layer, float *query, float *key, float *value, float *d_output, float *d_input)`
    - `update_attention(AttentionLayer *layer, const float *d_weights_query, const float *d_weights_key, const float *d_weights_value, float learning_rate)`
    - `free_attention(AttentionLayer *layer)`
- **File**: [`attention.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Layers/attention.c)

## Additive Attention Layer
- **Description**: Implements additive attention mechanism.
- **Functions**: 
    - `initialize_additive_attention(AdditiveAttentionLayer *layer, int query_dim, int key_dim, int value_dim)`
    - `forward_additive_attention(AdditiveAttentionLayer *layer, float *query, float *key, float *value, float *output, const char *optimizer_type)`
    - `backward_additive_attention(AdditiveAttentionLayer *layer, float *query, float *key, float *value, float *d_output, float *d_input)`
    - `update_additive_attention(AdditiveAttentionLayer *layer, const float *d_weights_query, const float *d_weights_key, const float *d_weights_value, const float *d_bias, float learning_rate)`
    - `free_additive_attention(AdditiveAttentionLayer *layer)`
- **File**: [`additive_attention.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Layers/additive_attention.c)

## Multi-Head Attention Layer
- **Description**: Implements multi-head attention mechanism.
- **Functions**: 
    - `initialize_multi_head_attention(MultiHeadAttentionLayer *layer, int query_dim, int key_dim, int value_dim, int num_heads)`
    - `forward_multi_head_attention(MultiHeadAttentionLayer *layer, const float *query, const float *key, const float *value, float *output)`
    - `backward_multi_head_attention(MultiHeadAttentionLayer *layer, const float *query, const float *key, const float *value, const float *d_output, float *d_query, float *d_key, float *d_value)`
    - `update_multi_head_attention(MultiHeadAttentionLayer *layer, float *d_weights_query, float *d_weights_key, float *d_weights_value, float *d_weights_output, float learning_rate)`
    - `free_multi_head_attention(MultiHeadAttentionLayer *layer)`
- **File**: [`multi_head_attention.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Layers/multi_head_attention.c)

