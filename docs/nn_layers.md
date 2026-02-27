# Neural Network Layers API

CML provides a comprehensive set of neural network layers following a consistent Module-based API.
All layers inherit from `Module` and can be composed using container types like `Sequential`.

Include `"cml.h"` or `"nn/layers.h"` to access all layer types.

______________________________________________________________________

## Table of Contents

- [Module API](#module-api)
- [Containers](#containers)
  - [Sequential](#sequential)
  - [ModuleList](#modulelist)
  - [ModuleDict](#moduledict)
- [Linear](#linear)
- [Convolutions](#convolutions)
  - [Conv1d](#conv1d)
  - [Conv2d](#conv2d)
  - [Conv3d](#conv3d)
- [Recurrent Layers](#recurrent-layers)
  - [RNNCell](#rnncell)
  - [LSTMCell](#lstmcell)
  - [GRUCell](#grucell)
- [Transformer](#transformer)
  - [MultiHeadAttention](#multiheadattention)
  - [TransformerEncoderLayer](#transformerencoderlayer)
- [Embedding](#embedding)
- [Normalization](#normalization)
  - [BatchNorm2d](#batchnorm2d)
  - [LayerNorm](#layernorm)
  - [GroupNorm](#groupnorm)
- [Pooling](#pooling)
  - [MaxPool2d](#maxpool2d)
  - [AvgPool2d](#avgpool2d)
- [Activation Layers](#activation-layers)
  - [Module Form](#module-form)
  - [Functional Form](#functional-form)
- [Dropout](#dropout)
- [Complete Example](#complete-example)

______________________________________________________________________

## Module API

Every layer in CML inherits from the `Module` base type. The following operations work on any module:

```c
// Forward pass
Tensor* cml_nn_module_forward(Module* module, Tensor* input);

// Training/evaluation mode
void cml_nn_module_set_training(Module* module, bool training);
void cml_nn_module_train(Module* module);     // Set training mode
void cml_nn_module_eval(Module* module);      // Set evaluation mode
bool cml_nn_module_is_training(Module* module);

// Inspection
void cml_summary(Module* module);  // Print model architecture summary

// Cleanup
void module_free(Module* module);
```

Training mode affects the behavior of layers like `Dropout` (disabled during eval) and `BatchNorm2d` (uses running statistics during eval).

______________________________________________________________________

## Containers

### Sequential

A sequential container executes layers in the order they are added. This is the most common way to build feedforward networks.

```c
Sequential* nn_sequential(void);

// Add layers one at a time
void sequential_add(Sequential* seq, Module* module);

// Add multiple layers at once (NULL-terminated variadic)
Sequential* sequential_add_chain(Sequential* seq, Module* first, ...);

// Access layers
Module* sequential_get(Sequential* seq, int index);
int sequential_get_length(Sequential* seq);
```

**Example:**

```c
Sequential* model = nn_sequential();

// Fluent chaining
model = sequential_add_chain(model,
    (Module*)nn_linear(784, 128, dtype, device, true),
    (Module*)nn_relu(false),
    (Module*)nn_linear(128, 10, dtype, device, true),
    NULL
);

// Or add one at a time
sequential_add(model, (Module*)nn_relu(false));

Tensor* output = cml_nn_module_forward((Module*)model, input);
cml_summary((Module*)model);
module_free((Module*)model);
```

### ModuleList

A dynamic, ordered collection of modules. Unlike `Sequential`, a `ModuleList` does not define a forward pass -- you iterate over it manually.

```c
ModuleList* nn_module_list(void);

int  module_list_append(ModuleList* list, Module* module);
int  module_list_insert(ModuleList* list, int index, Module* module);
Module* module_list_get(ModuleList* list, int index);
int  module_list_remove(ModuleList* list, int index);
int  module_list_length(ModuleList* list);
```

**Example:**

```c
ModuleList* layers = nn_module_list();
module_list_append(layers, (Module*)nn_linear(64, 32, dtype, device, true));
module_list_append(layers, (Module*)nn_linear(32, 16, dtype, device, true));

// Manual forward through all layers
Tensor* x = input;
for (int i = 0; i < module_list_length(layers); i++) {
    x = cml_nn_module_forward(module_list_get(layers, i), x);
}
```

### ModuleDict

A dictionary mapping string keys to modules. Useful for named sub-networks or branching architectures.

```c
ModuleDict* nn_module_dict(void);

int    module_dict_add(ModuleDict* dict, const char* key, Module* module);
Module* module_dict_get(ModuleDict* dict, const char* key);
int    module_dict_remove(ModuleDict* dict, const char* key);
int    module_dict_size(ModuleDict* dict);
const char** module_dict_keys(ModuleDict* dict, int* num_keys);
```

**Example:**

```c
ModuleDict* branches = nn_module_dict();
module_dict_add(branches, "encoder", (Module*)nn_linear(784, 128, dtype, device, true));
module_dict_add(branches, "decoder", (Module*)nn_linear(128, 784, dtype, device, true));

Module* enc = module_dict_get(branches, "encoder");
Tensor* latent = cml_nn_module_forward(enc, input);
```

______________________________________________________________________

## Linear

Fully connected layer computing `output = input @ weight + bias`.

```c
Linear* nn_linear(int in_features, int out_features,
                  DType dtype, DeviceType device, bool use_bias);

// With custom initialization functions
Linear* nn_linear_with_init(int in_features, int out_features,
                            DType dtype, DeviceType device, bool use_bias,
                            void (*weight_init)(Tensor*, int, int),
                            void (*bias_init)(Tensor*, int));
```

**Parameters:**

| Parameter      | Description                           |
| -------------- | ------------------------------------- |
| `in_features`  | Size of each input sample             |
| `out_features` | Size of each output sample            |
| `dtype`        | Data type (e.g., `DTYPE_FLOAT32`)     |
| `device`       | Device placement (e.g., `DEVICE_CPU`) |
| `use_bias`     | If `true`, adds a learnable bias      |

**Weight shapes:**

- `weight`: `[out_features, in_features]`
- `bias`: `[out_features]` (if enabled)

**Accessor functions:**

```c
Parameter* linear_get_weight(Linear* linear);
Parameter* linear_get_bias(Linear* linear);
int linear_get_in_features(Linear* linear);
int linear_get_out_features(Linear* linear);
int linear_set_weight(Linear* linear, Tensor* weight);
int linear_set_bias(Linear* linear, Tensor* bias);
void linear_set_use_bias(Linear* linear, bool use_bias);
bool linear_get_use_bias(Linear* linear);
```

**Example:**

```c
Linear* fc = nn_linear(784, 256, DTYPE_FLOAT32, DEVICE_CPU, true);
Tensor* output = cml_nn_module_forward((Module*)fc, input);

Parameter* w = linear_get_weight(fc);
module_free((Module*)fc);
```

______________________________________________________________________

## Convolutions

### Conv1d

1D convolution over a temporal/sequential input.

```c
Conv1d* nn_conv1d(int in_channels, int out_channels, int kernel_size,
                  int stride, int padding, int dilation,
                  bool use_bias, DType dtype, DeviceType device);
```

**Weight shapes:**

- `weight`: `[out_channels, in_channels, kernel_size]`
- `bias`: `[out_channels]` (if enabled)

**Example:**

```c
Conv1d* conv = nn_conv1d(1, 16, 3, 1, 1, 1, true, DTYPE_FLOAT32, DEVICE_CPU);
Tensor* output = cml_nn_module_forward((Module*)conv, input);
// input shape: [batch, in_channels, length]
// output shape: [batch, out_channels, output_length]
```

### Conv2d

2D convolution over spatial input (images).

```c
Conv2d* nn_conv2d(int in_channels, int out_channels, int kernel_size,
                  int stride, int padding, int dilation,
                  bool bias, DType dtype, DeviceType device);
```

**Weight shapes:**

- `weight`: `[out_channels, in_channels, kernel_size, kernel_size]`
- `bias`: `[out_channels]` (if enabled)

**Example:**

```c
Conv2d* conv = nn_conv2d(3, 64, 3, 1, 1, 1, true, DTYPE_FLOAT32, DEVICE_CPU);
Tensor* output = cml_nn_module_forward((Module*)conv, input);
// input shape: [batch, in_channels, height, width]
// output shape: [batch, out_channels, out_height, out_width]
```

### Conv3d

3D convolution over volumetric input (video, 3D data).

```c
Conv3d* nn_conv3d(int in_channels, int out_channels, int kernel_size,
                  int stride, int padding, int dilation,
                  bool use_bias, DType dtype, DeviceType device);
```

**Weight shapes:**

- `weight`: `[out_channels, in_channels, kd, kh, kw]`
- `bias`: `[out_channels]` (if enabled)

When `kernel_size` is a single integer, it is applied uniformly across all three spatial dimensions (depth, height, width). Internally, `stride`, `padding`, and `dilation` are stored per-dimension as arrays of length 3.

**Example:**

```c
Conv3d* conv = nn_conv3d(3, 32, 3, 1, 1, 1, true, DTYPE_FLOAT32, DEVICE_CPU);
Tensor* output = cml_nn_module_forward((Module*)conv, input);
// input shape: [batch, in_channels, depth, height, width]
```

______________________________________________________________________

## Recurrent Layers

CML provides cell-level recurrent layers. For sequence processing, call the cell in a loop over time steps.

### RNNCell

A single vanilla RNN step: `h' = tanh(W_ih @ x + W_hh @ h + bias)`.

```c
RNNCell* nn_rnn_cell(int input_size, int hidden_size, bool use_bias,
                     DType dtype, DeviceType device);

Tensor* rnn_cell_forward(RNNCell* cell, Tensor* input, Tensor* hidden);
```

**Weight shapes:**

- `weight_ih`: `[hidden_size, input_size]`
- `weight_hh`: `[hidden_size, hidden_size]`
- `bias_ih`, `bias_hh`: `[hidden_size]` (if enabled)

**Example:**

```c
RNNCell* cell = nn_rnn_cell(10, 20, true, DTYPE_FLOAT32, DEVICE_CPU);

// Process a sequence of length T
Tensor* h = /* initial hidden state [batch, hidden_size] */;
for (int t = 0; t < seq_len; t++) {
    Tensor* x_t = /* input at time t [batch, input_size] */;
    h = rnn_cell_forward(cell, x_t, h);
}
```

### LSTMCell

A single LSTM step with four gates: input, forget, cell, output.

```c
LSTMCell* nn_lstm_cell(int input_size, int hidden_size, bool use_bias,
                       DType dtype, DeviceType device);

void lstm_cell_forward(LSTMCell* cell, Tensor* input,
                       Tensor* h_prev, Tensor* c_prev,
                       Tensor** h_out, Tensor** c_out);
```

**Weight shapes:**

- `weight_ih`: `[4*hidden_size, input_size]` (gates: input, forget, cell, output)
- `weight_hh`: `[4*hidden_size, hidden_size]`
- `bias_ih`, `bias_hh`: `[4*hidden_size]` (if enabled)

**Example:**

```c
LSTMCell* cell = nn_lstm_cell(10, 20, true, DTYPE_FLOAT32, DEVICE_CPU);

Tensor* h = /* [batch, hidden_size] */;
Tensor* c = /* [batch, hidden_size] */;

for (int t = 0; t < seq_len; t++) {
    Tensor* x_t = /* [batch, input_size] */;
    lstm_cell_forward(cell, x_t, h, c, &h, &c);
}
// h is the final hidden state, c is the final cell state
```

### GRUCell

A single GRU step with three gates: reset, update, new.

```c
GRUCell* nn_gru_cell(int input_size, int hidden_size, bool use_bias,
                     DType dtype, DeviceType device);

Tensor* gru_cell_forward(GRUCell* cell, Tensor* input, Tensor* hidden);
```

**Weight shapes:**

- `weight_ih`: `[3*hidden_size, input_size]` (gates: reset, update, new)
- `weight_hh`: `[3*hidden_size, hidden_size]`
- `bias_ih`, `bias_hh`: `[3*hidden_size]` (if enabled)

**Example:**

```c
GRUCell* cell = nn_gru_cell(10, 20, true, DTYPE_FLOAT32, DEVICE_CPU);

Tensor* h = /* [batch, hidden_size] */;
for (int t = 0; t < seq_len; t++) {
    Tensor* x_t = /* [batch, input_size] */;
    h = gru_cell_forward(cell, x_t, h);
}
```

______________________________________________________________________

## Transformer

### MultiHeadAttention

Scaled dot-product multi-head attention with learned projections for queries, keys, and values.

```c
MultiHeadAttention* nn_multihead_attention(int embed_dim, int num_heads,
                                            float dropout,
                                            DType dtype, DeviceType device);

Tensor* multihead_attention_forward(MultiHeadAttention* mha,
                                     Tensor* query, Tensor* key,
                                     Tensor* value, Tensor* mask);
```

**Parameters:**

| Parameter   | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| `embed_dim` | Total embedding dimension (must be divisible by `num_heads`) |
| `num_heads` | Number of parallel attention heads                           |
| `dropout`   | Dropout probability on attention weights                     |
| `mask`      | Optional attention mask (pass `NULL` for no mask)            |

**Weight shapes:**

- `W_q`, `W_k`, `W_v`, `W_o`: `[embed_dim, embed_dim]`
- `b_q`, `b_k`, `b_v`, `b_o`: `[embed_dim]`

Head dimension is computed as `embed_dim / num_heads`.

**Example:**

```c
MultiHeadAttention* mha = nn_multihead_attention(512, 8, 0.1f,
                                                  DTYPE_FLOAT32, DEVICE_CPU);

// Self-attention: query = key = value
Tensor* attn_out = multihead_attention_forward(mha, x, x, x, NULL);
```

### TransformerEncoderLayer

A single transformer encoder layer consisting of multi-head self-attention followed by a position-wise feedforward network, with residual connections and layer normalization.

```c
TransformerEncoderLayer* nn_transformer_encoder_layer(
    int d_model, int nhead, int dim_feedforward,
    float dropout, DType dtype, DeviceType device);
```

**Parameters:**

| Parameter         | Description                                 |
| ----------------- | ------------------------------------------- |
| `d_model`         | Model embedding dimension                   |
| `nhead`           | Number of attention heads                   |
| `dim_feedforward` | Hidden dimension of the feedforward network |
| `dropout`         | Dropout probability                         |

**Internal structure:**

- Self-attention (`MultiHeadAttention`)
- Feedforward: `Linear(d_model, dim_feedforward)` -> activation -> `Linear(dim_feedforward, d_model)`
- Two `LayerNorm` layers (norm1 and norm2)

**Example:**

```c
TransformerEncoderLayer* layer = nn_transformer_encoder_layer(
    512, 8, 2048, 0.1f, DTYPE_FLOAT32, DEVICE_CPU);

Tensor* output = cml_nn_module_forward((Module*)layer, input);
// input/output shape: [seq_len, batch, d_model]
```

______________________________________________________________________

## Embedding

A lookup table that maps integer indices to dense vectors. Commonly used for word embeddings.

```c
Embedding* nn_embedding(int num_embeddings, int embedding_dim,
                         int padding_idx,
                         DType dtype, DeviceType device);
```

**Parameters:**

| Parameter        | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| `num_embeddings` | Size of the vocabulary (number of embeddings)                |
| `embedding_dim`  | Dimension of each embedding vector                           |
| `padding_idx`    | Index whose embedding is zeroed out; use `-1` for no padding |

**Weight shape:**

- `weight`: `[num_embeddings, embedding_dim]`

**Example:**

```c
// Vocabulary of 10000 words, 128-dimensional embeddings
Embedding* emb = nn_embedding(10000, 128, 0, DTYPE_FLOAT32, DEVICE_CPU);
// padding_idx=0 means the embedding at index 0 stays zero

Tensor* output = cml_nn_module_forward((Module*)emb, token_indices);
// token_indices shape: [batch, seq_len] (integer tensor)
// output shape: [batch, seq_len, 128]
```

______________________________________________________________________

## Normalization

### BatchNorm2d

Batch normalization over 4D input (batch, channels, height, width). Normalizes across the batch dimension per channel.

```c
BatchNorm2d* nn_batchnorm2d(int num_features, float eps, float momentum,
                             bool affine, bool track_running_stats,
                             DType dtype, DeviceType device);
```

**Parameters:**

| Parameter             | Description                                              |
| --------------------- | -------------------------------------------------------- |
| `num_features`        | Number of channels (C)                                   |
| `eps`                 | Small constant for numerical stability (typical: `1e-5`) |
| `momentum`            | Running mean/variance update factor (typical: `0.1`)     |
| `affine`              | If `true`, learnable scale (gamma) and shift (beta)      |
| `track_running_stats` | If `true`, tracks running mean/variance for eval mode    |

**Example:**

```c
BatchNorm2d* bn = nn_batchnorm2d(64, 1e-5f, 0.1f, true, true,
                                  DTYPE_FLOAT32, DEVICE_CPU);
Tensor* output = cml_nn_module_forward((Module*)bn, input);
// input shape: [batch, 64, height, width]
```

### LayerNorm

Layer normalization over the last dimension. Normalizes each sample independently.

```c
LayerNorm* nn_layernorm(int normalized_shape, float eps, bool affine,
                         DType dtype, DeviceType device);
```

**Parameters:**

| Parameter          | Description                                              |
| ------------------ | -------------------------------------------------------- |
| `normalized_shape` | Size of the normalized dimension                         |
| `eps`              | Small constant for numerical stability (typical: `1e-5`) |
| `affine`           | If `true`, learnable scale and shift parameters          |

**Example:**

```c
LayerNorm* ln = nn_layernorm(512, 1e-5f, true, DTYPE_FLOAT32, DEVICE_CPU);
Tensor* output = cml_nn_module_forward((Module*)ln, input);
// input shape: [batch, seq_len, 512]
```

### GroupNorm

Group normalization that divides channels into groups and normalizes within each group. A middle ground between LayerNorm and InstanceNorm.

```c
GroupNorm* nn_groupnorm(int num_groups, int num_channels, float eps,
                         bool affine, DType dtype, DeviceType device);
```

**Parameters:**

| Parameter      | Description                                                  |
| -------------- | ------------------------------------------------------------ |
| `num_groups`   | Number of groups to divide channels into                     |
| `num_channels` | Total number of channels (must be divisible by `num_groups`) |
| `eps`          | Small constant for numerical stability (typical: `1e-5`)     |
| `affine`       | If `true`, learnable scale and shift per channel             |

**Weight shapes** (if affine):

- `weight`: `[num_channels]`
- `bias`: `[num_channels]`

**Example:**

```c
// 32 groups over 256 channels (8 channels per group)
GroupNorm* gn = nn_groupnorm(32, 256, 1e-5f, true, DTYPE_FLOAT32, DEVICE_CPU);
Tensor* output = cml_nn_module_forward((Module*)gn, input);
```

______________________________________________________________________

## Pooling

### MaxPool2d

2D max pooling over spatial input. Selects the maximum value within each pooling window.

```c
MaxPool2d* nn_maxpool2d(int kernel_size, int stride, int padding,
                         int dilation, bool ceil_mode);
```

**Parameters:**

| Parameter     | Description                                          |
| ------------- | ---------------------------------------------------- |
| `kernel_size` | Size of the pooling window                           |
| `stride`      | Stride of the pooling window                         |
| `padding`     | Zero-padding added to both sides                     |
| `dilation`    | Spacing between kernel elements                      |
| `ceil_mode`   | If `true`, use ceil instead of floor for output size |

### AvgPool2d

2D average pooling over spatial input. Computes the average value within each pooling window.

```c
AvgPool2d* nn_avgpool2d(int kernel_size, int stride, int padding,
                         bool ceil_mode, bool count_include_pad);
```

**Parameters:**

| Parameter           | Description                                             |
| ------------------- | ------------------------------------------------------- |
| `kernel_size`       | Size of the pooling window                              |
| `stride`            | Stride of the pooling window                            |
| `padding`           | Zero-padding added to both sides                        |
| `ceil_mode`         | If `true`, use ceil instead of floor for output size    |
| `count_include_pad` | If `true`, include padding in the averaging calculation |

**Example:**

```c
MaxPool2d* pool = nn_maxpool2d(2, 2, 0, 1, false);
Tensor* output = cml_nn_module_forward((Module*)pool, input);
// input shape: [batch, channels, height, width]
// output shape: [batch, channels, height/2, width/2]
```

______________________________________________________________________

## Activation Layers

Activations are available in two forms: as module objects (for use in `Sequential`) and as standalone functions (zero-allocation, for manual forward passes).

### Module Form

Create activation modules that can be added to `Sequential` or used standalone.

| Layer      | Constructor                                         | Parameters                            |
| ---------- | --------------------------------------------------- | ------------------------------------- |
| ReLU       | `nn_relu(bool inplace)`                             | `inplace`: modify input in-place      |
| LeakyReLU  | `nn_leaky_relu(float negative_slope, bool inplace)` | `negative_slope`: slope for x \< 0    |
| Sigmoid    | `nn_sigmoid()`                                      | --                                    |
| Tanh       | `nn_tanh()`                                         | --                                    |
| GELU       | `nn_gelu(bool approximate)`                         | `approximate`: use tanh approximation |
| ELU        | `nn_elu(float alpha, bool inplace)`                 | `alpha`: scale for negative values    |
| SELU       | `nn_selu()`                                         | --                                    |
| SiLU       | `nn_silu()`                                         | -- (also known as Swish)              |
| Mish       | `nn_mish()`                                         | --                                    |
| HardSwish  | `nn_hardswish()`                                    | --                                    |
| Softmax    | `nn_softmax(int dim)`                               | `dim`: dimension to apply softmax     |
| LogSoftmax | `nn_log_softmax(int dim)`                           | `dim`: dimension to apply log-softmax |

**Example:**

```c
ReLU* relu = nn_relu(false);
Tensor* output = cml_nn_module_forward((Module*)relu, input);

// In a Sequential
sequential_add(model, (Module*)nn_gelu(false));
sequential_add(model, (Module*)nn_softmax(-1));
```

### Functional Form

Functional activations do not allocate a module. They take a tensor and return a tensor directly.

```c
Tensor* f_relu(Tensor* input);
Tensor* f_sigmoid(Tensor* input);
Tensor* f_tanh(Tensor* input);
Tensor* f_gelu(Tensor* input);
Tensor* f_elu(Tensor* input, float alpha);
Tensor* f_selu(Tensor* input);
Tensor* f_silu(Tensor* input);
Tensor* f_mish(Tensor* input);
Tensor* f_hardswish(Tensor* input);
```

**Example:**

```c
Tensor* h = f_relu(cml_nn_module_forward((Module*)linear, input));
Tensor* probs = f_sigmoid(h);
```

______________________________________________________________________

## Dropout

Randomly zeroes elements during training with probability `p`. Automatically disabled during evaluation.

```c
Dropout* nn_dropout(float p, bool inplace);
```

**Parameters:**

| Parameter | Description                                         |
| --------- | --------------------------------------------------- |
| `p`       | Probability of an element being zeroed (0.0 to 1.0) |
| `inplace` | If `true`, modify the input tensor in-place         |

During training, outputs are scaled by `1/(1-p)` to maintain expected values. During evaluation (`cml_nn_module_eval`), dropout is a no-op.

**Example:**

```c
Dropout* drop = nn_dropout(0.5f, false);

cml_nn_module_train((Module*)drop);  // Dropout active
Tensor* train_out = cml_nn_module_forward((Module*)drop, input);

cml_nn_module_eval((Module*)drop);   // Dropout disabled
Tensor* eval_out = cml_nn_module_forward((Module*)drop, input);
```

______________________________________________________________________

## Complete Example

A full example building and running a multi-layer perceptron classifier:

```c
#include "cml.h"

int main() {
    cml_init();

    DeviceType device = cml_get_default_device();
    DType dtype = cml_get_default_dtype();

    // Build model
    Sequential* model = nn_sequential();
    sequential_add(model, (Module*)nn_linear(784, 256, dtype, device, true));
    sequential_add(model, (Module*)nn_relu(false));
    sequential_add(model, (Module*)nn_dropout(0.3f, false));
    sequential_add(model, (Module*)nn_linear(256, 128, dtype, device, true));
    sequential_add(model, (Module*)nn_relu(false));
    sequential_add(model, (Module*)nn_dropout(0.3f, false));
    sequential_add(model, (Module*)nn_linear(128, 10, dtype, device, true));

    // Print architecture
    cml_summary((Module*)model);

    // Training mode
    cml_nn_module_train((Module*)model);
    Tensor* train_output = cml_nn_module_forward((Module*)model, train_input);

    // Evaluation mode
    cml_nn_module_eval((Module*)model);
    Tensor* eval_output = cml_nn_module_forward((Module*)model, test_input);

    // Cleanup
    module_free((Module*)model);
    cml_cleanup();
    return 0;
}
```

A convolutional network example:

```c
#include "cml.h"

int main() {
    cml_init();

    DType dtype = DTYPE_FLOAT32;
    DeviceType device = DEVICE_CPU;

    Sequential* model = nn_sequential();

    // Convolutional feature extractor
    sequential_add(model, (Module*)nn_conv2d(1, 32, 3, 1, 1, 1, true, dtype, device));
    sequential_add(model, (Module*)nn_batchnorm2d(32, 1e-5f, 0.1f, true, true, dtype, device));
    sequential_add(model, (Module*)nn_relu(false));
    sequential_add(model, (Module*)nn_maxpool2d(2, 2, 0, 1, false));

    sequential_add(model, (Module*)nn_conv2d(32, 64, 3, 1, 1, 1, true, dtype, device));
    sequential_add(model, (Module*)nn_batchnorm2d(64, 1e-5f, 0.1f, true, true, dtype, device));
    sequential_add(model, (Module*)nn_relu(false));
    sequential_add(model, (Module*)nn_maxpool2d(2, 2, 0, 1, false));

    // Classifier head (after flattening)
    sequential_add(model, (Module*)nn_linear(64 * 7 * 7, 128, dtype, device, true));
    sequential_add(model, (Module*)nn_relu(false));
    sequential_add(model, (Module*)nn_dropout(0.5f, false));
    sequential_add(model, (Module*)nn_linear(128, 10, dtype, device, true));

    cml_summary((Module*)model);

    module_free((Module*)model);
    cml_cleanup();
    return 0;
}
```
