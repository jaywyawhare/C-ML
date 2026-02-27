# C-ML Examples

## Building

All examples are built automatically when using CMake with `-DBUILD_EXAMPLES=ON`:

```bash
mkdir -p build && cd build
cmake -DBUILD_EXAMPLES=ON ..
make -j$(nproc)
```

Binaries are placed in `build/`.

## Tutorials

Step-by-step examples that progressively introduce CML features.

### hello_cml -- Minimal Example

Minimal forward pass through a linear layer. Good starting point.

```bash
./build/hello_cml
```

### simple_xor -- XOR Training

Classic XOR problem with a full training loop. Demonstrates the complete train-evaluate cycle.

```bash
./build/simple_xor
```

### ex01 -- Tensor Operations

Basic tensor creation, element-wise arithmetic, reductions, and shape manipulation. No dataset required.

```bash
./build/ex01_tensor_ops
```

### ex02 -- Linear Regression

Linear regression trained with SGD on the Boston Housing dataset. Demonstrates `cml_dataset_load()`, `dataset_normalize()`, and `dataset_split()`.

```bash
./build/ex02_linear_regression
```

### ex03 -- Logistic Regression

Binary classification using sigmoid + BCE loss on the Breast Cancer dataset.

```bash
./build/ex03_logistic_regression
```

### ex04 -- MLP Classifier

Multi-class classification with a 2-layer MLP on the Iris dataset. Shows Sequential model building and MSE training.

```bash
./build/ex04_mlp_classifier
```

### ex05 -- Autoencoder

Autoencoder with an encoder-decoder architecture and bottleneck layer. Trained on Digits 8x8 images.

```bash
./build/ex05_autoencoder
```

### ex06 -- Image Classification

Image classification MLP that processes 8x8 digit images using raw pixel features. Uses Digits dataset with BCE loss.

```bash
./build/ex06_conv_net
```

### ex07 -- RNN Sequence Prediction

RNN-based time series prediction on the Airline passengers dataset. Demonstrates `cml_nn_rnn_cell()` with sliding window input.

```bash
./build/ex07_rnn_sequence
```

### ex08 -- LSTM Time Series

LSTM-based time series forecasting on Airline data. Shows `cml_nn_lstm_cell()` with hidden and cell state management.

```bash
./build/ex08_lstm_timeseries
```

### ex09 -- GRU Classifier

GRU-based sequence classifier on the Iris dataset. Demonstrates `cml_nn_gru_cell()` processing features as a sequence.

```bash
./build/ex09_gru_classifier
```

### ex10 -- Embedding

Embedding lookup table demo. Shows `cml_nn_embedding()` for mapping integer indices to dense vectors.

```bash
./build/ex10_embedding
```

### ex11 -- GAN

Generative Adversarial Network with separate generator and discriminator networks. Trained on Digits 8x8.

```bash
./build/ex11_gan
```

### ex12 -- Multi-Task Learning

Multi-task learning with shared layers and task-specific heads. Uses the Wine dataset with two prediction targets.

```bash
./build/ex12_multi_task
```

### ex13 -- Transformer Encoder

Demonstrates `cml_nn_multihead_attention()` and `cml_nn_transformer_encoder_layer()` with self-attention on synthetic sequences.

```bash
./build/ex13_transformer
```

### ex14 -- LR Scheduler Comparison

Compares learning rate schedulers (Step, Exponential, Cosine, ReduceOnPlateau) side-by-side on Boston Housing regression.

```bash
./build/ex14_lr_scheduler
```

### ex15 -- Activation Functions Showcase

Trains separate networks using all 12 activation functions (ReLU, Sigmoid, Tanh, LeakyReLU, GELU, ELU, SELU, SiLU, Mish, HardSwish, Softmax, LogSoftmax) and compares convergence.

```bash
./build/ex15_activations_showcase
```

## Benchmarks

Performance measurement tools.

### bench_forward -- Forward Pass Benchmark

Benchmarks forward pass throughput for various layer types and sizes.

```bash
./build/bench_forward
```

### bench_gemm -- GEMM Benchmark

Benchmarks general matrix multiply (GEMM) performance across different matrix sizes.

```bash
./build/bench_gemm
```

## Demos

Feature demonstrations and advanced usage patterns.

| Demo | Description |
|------|-------------|
| `autograd_example` | Automatic differentiation and gradient computation |
| `auto_capture_example` | Automatic graph capture for optimization |
| `comprehensive_fusion_example` | IR fusion pass demonstration |
| `dead_code_example` | Dead code elimination optimization |
| `early_stopping_lr_scheduler` | Early stopping with learning rate scheduling |
| `export_graph` | Export computation graph to DOT format |
| `mnist_example` | MNIST digit classification |
| `print_kernels` | Print generated IR kernels |
| `training_loop_example` | Complete training loop with metrics |

Run any demo:

```bash
./build/<demo_name>
```

## Summary Table

| Example | Task | Dataset | Key APIs |
|---------|------|---------|----------|
| ex01 | Tensor ops | -- | `cml_add`, `cml_matmul`, `cml_sum` |
| ex02 | Regression | Boston Housing | `cml_dataset_load`, `cml_optim_sgd` |
| ex03 | Binary classification | Breast Cancer | `cml_nn_bce_loss`, `cml_nn_sigmoid` |
| ex04 | Multi-class classification | Iris | `cml_nn_sequential`, `cml_nn_linear` |
| ex05 | Autoencoder | Digits 8x8 | `cml_nn_sequential`, bottleneck |
| ex06 | Image classification | Digits 8x8 | `cml_nn_linear`, `cml_nn_relu` |
| ex07 | Time series | Airline | `cml_nn_rnn_cell` |
| ex08 | Time series | Airline | `cml_nn_lstm_cell` |
| ex09 | Sequence classification | Iris | `cml_nn_gru_cell` |
| ex10 | Embedding | -- | `cml_nn_embedding` |
| ex11 | Generative | Digits 8x8 | GAN training loop |
| ex12 | Multi-task | Wine | Shared layers |
| ex13 | Self-attention | -- | `cml_nn_transformer_encoder_layer` |
| ex14 | Scheduler comparison | Boston Housing | `lr_scheduler_*` |
| ex15 | Activation comparison | Breast Cancer | All activation layers |
| hello_cml | Forward pass | -- | `cml_nn_linear` |
| simple_xor | Full training | XOR | Complete training loop |
