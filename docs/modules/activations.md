# Activations

## ReLU
- **Description**: Rectified Linear Unit activation function.
- **Function**: `relu(float x)`
- **File**: [`relu.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Activations/relu.c)

## Sigmoid
- **Description**: Sigmoid activation function.
- **Function**: `sigmoid(float x)`
- **File**: [`sigmoid.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Activations/sigmoid.c)

## Tanh
- **Description**: Hyperbolic tangent activation function.
- **Function**: `tanH(float x)`
- **File**: [`tanh.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Activations/tanh.c)

## Softmax
- **Description**: Converts logits into probabilities.
- **Function**: `softmax(float *z, int n)`
- **File**: [`softmax.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Activations/softmax.c)

## ELU
- **Description**: Exponential Linear Unit activation function.
- **Function**: `elu(float x, float alpha)`
- **File**: [`elu.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Activations/elu.c)

## Leaky ReLU
- **Description**: Leaky version of ReLU to allow small gradients for negative inputs.
- **Function**: `leaky_relu(float x)`
- **File**: [`leaky_relu.c`](https://github.com/jaywyawhare/C-ML/blob/master/src/Activations/leaky_relu.c)

## Linear
- **Description**: Linear activation function (identity function).
- **Function**: `linear(float x)`
- **File**: [`linear.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Activations/linear.c)

## GELU
- **Description**: Gaussian Error Linear Unit activation function.
- **Function**: `gelu(float x)`
- **File**: [`gelu.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Activations/gelu.c)
