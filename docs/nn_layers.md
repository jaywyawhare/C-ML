# Neural Network Layers API

## Sequential Container

```c
// Create sequential model
Sequential* model = cml_nn_sequential();  // or nn_sequential()

// Add layers (fluent API)
model = sequential_add_chain(model,
    (Module*)cml_nn_linear(784, 128, dtype, device, true),
    (Module*)cml_nn_relu(false),
    (Module*)cml_nn_linear(128, 64, dtype, device, true),
    NULL
);

// Or using sequential_add
sequential_add(model, (Module*)cml_nn_linear(10, 20, dtype, device, true));
sequential_add(model, (Module*)cml_nn_relu(false));

// Forward pass
Tensor* output = cml_nn_module_forward((Module*)model, input);

// Access layers
Module* layer = sequential_get(model, 0);
int length = sequential_get_length(model);

// Training mode
cml_nn_module_set_training((Module*)model, true);   // Training mode
cml_nn_module_eval((Module*)model);                // Evaluation mode
bool is_training = cml_nn_module_is_training((Module*)model);
```

## Linear Layer

```c
Linear* linear = cml_nn_linear(in_features, out_features, dtype, device, bias);
// or nn_linear(in_features, out_features, dtype, device, bias)

Tensor* output = cml_nn_module_forward((Module*)linear, input);

// Access parameters
Parameter* weight = linear_get_weight(linear);
Parameter* bias = linear_get_bias(linear);
```

## Activation Layers

```c
ReLU* relu = cml_nn_relu(inplace);              // or nn_relu(inplace)
Sigmoid* sigmoid = cml_nn_sigmoid();            // or nn_sigmoid()
Tanh* tanh = cml_nn_tanh();                     // or nn_tanh()
LeakyReLU* leaky = cml_nn_leaky_relu(0.01f, false);  // or nn_leaky_relu(negative_slope, inplace)
GELU* gelu = nn_gelu(inplace);

Tensor* output = cml_nn_module_forward((Module*)relu, input);
```

## Dropout

```c
Dropout* dropout = cml_nn_dropout(0.5f, false);  // or nn_dropout(probability, inplace)
Tensor* output = cml_nn_module_forward((Module*)dropout, input);
// Only active in training mode
```

## Conv2d

```c
Conv2d* conv = cml_nn_conv2d(in_channels, out_channels, kernel_size,
                             stride, padding, dilation, bias, dtype, device);
// or nn_conv2d(in_channels, out_channels, kernel_size, stride, padding,
//              dilation, groups, bias, dtype, device)

Tensor* output = cml_nn_module_forward((Module*)conv, input);
```

## BatchNorm2d

```c
BatchNorm2d* bn = cml_nn_batchnorm2d(num_features, eps, momentum,
                                     affine, track_running_stats, dtype, device);
// or nn_batchnorm2d(num_features, eps, momentum, affine, track_running_stats, dtype, device)

Tensor* output = cml_nn_module_forward((Module*)bn, input);
// Different behavior in training vs evaluation mode
```

## LayerNorm

```c
LayerNorm* ln = cml_nn_layernorm(normalized_shape, eps, affine, dtype, device);
// or nn_layernorm(normalized_shape, eps, affine, dtype, device)

Tensor* output = cml_nn_module_forward((Module*)ln, input);
```

## Pooling

```c
MaxPool2d* maxpool = cml_nn_maxpool2d(kernel_size, stride, padding, dilation, ceil_mode);
// or nn_maxpool2d(kernel_size, stride, padding, dilation, ceil_mode)

AvgPool2d* avgpool = cml_nn_avgpool2d(kernel_size, stride, padding, ceil_mode, count_include_pad);
// or nn_avgpool2d(kernel_size, stride, padding, ceil_mode, count_include_pad)

Tensor* output = cml_nn_module_forward((Module*)maxpool, input);
```

## Complete Example

```c
#include "cml.h"

int main() {
    cml_init();

    DeviceType device = cml_get_default_device();
    DType dtype = cml_get_default_dtype();

    Sequential* model = cml_nn_sequential();
    sequential_add(model, (Module*)cml_nn_linear(784, 128, dtype, device, true));
    sequential_add(model, (Module*)cml_nn_relu(false));
    sequential_add(model, (Module*)cml_nn_dropout(0.5f, false));
    sequential_add(model, (Module*)cml_nn_linear(128, 64, dtype, device, true));
    sequential_add(model, (Module*)cml_nn_relu(false));
    sequential_add(model, (Module*)cml_nn_linear(64, 10, dtype, device, true));

    cml_summary((Module*)model);
    cml_nn_module_set_training((Module*)model, true);

    Tensor* output = cml_nn_module_forward((Module*)model, input);

    module_free((Module*)model);
    cml_cleanup();
    return 0;
}
```
