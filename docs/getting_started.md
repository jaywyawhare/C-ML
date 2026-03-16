# Getting Started with C-ML

A practical guide to building, linking, and writing your first programs with C-ML.

## Prerequisites

- **C11 compatible compiler**: GCC 4.9+, Clang 3.5+, or MSVC 2015+
- **CMake 3.16+** (or GNU Make)
- **Math library** (libm, included on most systems)
- **Optional**: OpenBLAS or Intel MKL for BLAS-accelerated matrix operations
- **Optional**: SLEEF for vectorized transcendental math (exp, log, tanh)

## Building

### CMake (recommended)

```bash
git clone https://github.com/jaywyawhare/C-ML.git
cd C-ML
mkdir -p build && cd build
cmake -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON ..
make -j$(nproc)
```

This produces `build/lib/libcml_static.a`, all example binaries in `build/bin/`, and test binaries in the build tree.

### Makefile

```bash
make          # Standard build
make release  # Optimized (-O3, LTO)
make debug    # Debug build with sanitizers
make test     # Build + run all tests
```

## Your First Program

Create a file called `hello.c`:

```c
#include "cml.h"
#include <stdio.h>

int main(void) {
    cml_init();

    Sequential* model = cml_nn_sequential();
    cml_nn_sequential_add(model, (Module*)cml_nn_linear(10, 5, DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(model, (Module*)cml_nn_relu(false));
    cml_nn_sequential_add(model, (Module*)cml_nn_linear(5, 2, DTYPE_FLOAT32, DEVICE_CPU, true));

    cml_summary((Module*)model);

    float input_data[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int shape[] = {1, 10};
    Tensor* input = cml_tensor(input_data, shape, 2, NULL);
    Tensor* output = cml_nn_sequential_forward(model, input);

    // Lazy execution -- force evaluation before reading data
    tensor_ensure_executed(output);
    printf("Output: [%.4f, %.4f]\n",
           tensor_get_float(output, 0),
           tensor_get_float(output, 1));

    cml_cleanup();
    return 0;
}
```

Compile and run:

```bash
gcc -std=c11 -O2 hello.c -I./include -L./build/lib -lcml_static -lm -ldl -o hello
./hello
```

## Training a Model

This example trains a 3-class MLP classifier on the built-in Iris dataset:

```c
#include "cml.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    cml_init();

    Dataset* ds = cml_dataset_load("iris");
    dataset_normalize(ds, "minmax");
    Dataset *train, *test;
    dataset_split(ds, 0.8f, &train, &test);

    int nc = ds->num_classes;  // 3

    // Build one-hot targets (BCE training avoids cross-entropy gather issues)
    int n_train = train->num_samples;
    float* train_y = (float*)tensor_data_ptr(train->y);
    float* onehot = calloc(n_train * nc, sizeof(float));
    for (int i = 0; i < n_train; i++) {
        int cls = (int)train_y[i];
        if (cls >= 0 && cls < nc)
            onehot[i * nc + cls] = 1.0f;
    }
    int y_shape[] = {n_train, nc};
    Tensor* y_oh = cml_tensor(onehot, y_shape, 2, NULL);

    Sequential* model = cml_nn_sequential();
    cml_nn_sequential_add(model, (Module*)cml_nn_linear(train->input_size, 16,
                          DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(model, (Module*)cml_nn_relu(false));
    cml_nn_sequential_add(model, (Module*)cml_nn_linear(16, 8, DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(model, (Module*)cml_nn_relu(false));
    cml_nn_sequential_add(model, (Module*)cml_nn_linear(8, nc, DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(model, (Module*)cml_nn_sigmoid());

    Optimizer* opt = cml_optim_adam_for_model((Module*)model, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);

    for (int epoch = 1; epoch <= 100; epoch++) {
        Tensor* pred = cml_nn_sequential_forward(model, train->X);
        Tensor* loss = cml_nn_bce_loss(pred, y_oh);

        cml_optim_zero_grad(opt);
        cml_backward(loss, NULL, false, false);
        cml_optim_step(opt);

        if (epoch % 20 == 0)
            printf("Epoch %3d  Loss: %.6f\n", epoch, tensor_get_float(loss, 0));
    }

    Tensor* test_pred = cml_nn_sequential_forward(model, test->X);
    tensor_ensure_executed(test_pred);
    float* test_y = (float*)tensor_data_ptr(test->y);
    int correct = 0;
    for (int i = 0; i < test->num_samples; i++) {
        float best = -1e9f;
        int pred_cls = 0;
        for (int c = 0; c < nc; c++) {
            float v = tensor_get_float(test_pred, i * nc + c);
            if (v > best) { best = v; pred_cls = c; }
        }
        correct += (pred_cls == (int)test_y[i]);
    }
    printf("Test accuracy: %d/%d (%.1f%%)\n",
           correct, test->num_samples,
           correct / (float)test->num_samples * 100);

    free(onehot);
    cml_cleanup();
    return 0;
}
```

## Running Examples

After building with `-DBUILD_EXAMPLES=ON`, all example binaries are in `build/bin/`:

```bash
./build/bin/ex01_tensor_ops      # Basic tensor operations
./build/bin/ex04_mlp_classifier  # Iris classification
./build/bin/hello_cml            # Smoke test
```

See [examples/README.md](../examples/README.md) for the full list of 15+ examples covering linear regression, autoencoders, CNNs, RNNs, LSTMs, transformers, GANs, and more.

## Python Bindings

C-ML includes Python bindings via CFFI. After building the C library:

```bash
cd python
pip install cffi
python cml/build_cffi.py
python -c "import cml; cml.init(); print('OK'); cml.cleanup()"
```

See [python/INSTALLATION.md](../python/INSTALLATION.md) for full setup instructions and usage examples.

## Next Steps

- [API Reference](api_reference.md) -- complete function and type documentation
- [Neural Network Layers](nn_layers.md) -- available layers (Linear, Conv, RNN, Transformer, etc.)
- [Training Guide](training.md) -- optimizers, loss functions, learning rate schedulers
- [Datasets](datasets.md) -- built-in datasets and data loading utilities
