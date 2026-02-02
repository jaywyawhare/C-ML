"""
Neural network layers and modules.
"""

from cml._cml_lib import ffi, lib
from cml.core import Tensor, DTYPE_FLOAT32, DEVICE_CPU


class Module:
    """Base class for neural network modules."""

    def __init__(self, c_module):
        self._module = c_module

    def __call__(self, input_tensor):
        """Forward pass."""
        return self.forward(input_tensor)

    def forward(self, input_tensor):
        """Forward pass through module.

        Args:
            input_tensor: Input tensor

        Returns:
            Output tensor
        """
        result = lib.cml_nn_module_forward(self._module, input_tensor._tensor)
        return Tensor(result)

    def set_training(self, training=True):
        """Set training mode.

        Args:
            training: True for training mode, False for inference

        Example:
            >>> model.set_training(True)   # Training
            >>> model.set_training(False)  # Inference
        """
        lib.cml_nn_module_set_training(self._module, training)

    def __del__(self):
        """Clean up module."""
        if self._module != ffi.NULL:
            lib.module_free(self._module)


class Sequential(Module):
    """Sequential container of modules.

    Applies modules in sequence: y = module_n(...module_2(module_1(x))...)

    Example:
        >>> model = Sequential()
        >>> model.add(Linear(10, 20))
        >>> model.add(ReLU())
        >>> model.add(Linear(20, 5))
        >>>
        >>> output = model(input_tensor)
    """

    def __init__(self):
        """Initialize empty sequential container."""
        seq = lib.cml_nn_sequential()
        super().__init__(seq)
        self.layers = []

    def add(self, layer):
        """Add a module to the sequence.

        Args:
            layer: Module to add

        Returns:
            Self for chaining

        Example:
            >>> model = Sequential()
            >>> model.add(Linear(10, 20)).add(ReLU()).add(Linear(20, 5))
        """
        self._module = lib.cml_nn_sequential_add(self._module, layer._module)
        self.layers.append(layer)
        return self

    def __len__(self):
        """Get number of layers."""
        return len(self.layers)

    def __getitem__(self, index):
        """Get layer by index."""
        return self.layers[index]


class Linear(Module):
    """Fully connected (dense) layer.

    y = xW + b

    Example:
        >>> layer = Linear(in_features=784, out_features=128)
        >>> output = layer(input)  # [batch, 128]
    """

    def __init__(
        self,
        in_features,
        out_features,
        dtype=DTYPE_FLOAT32,
        device=DEVICE_CPU,
        bias=True,
    ):
        """Initialize linear layer.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            dtype: Data type
            device: Device type
            bias: Whether to include bias term
        """
        linear = lib.cml_nn_linear(in_features, out_features, dtype, device, bias)
        super().__init__(linear)
        self.in_features = in_features
        self.out_features = out_features


class ReLU(Module):
    """ReLU (Rectified Linear Unit) activation.

    y = max(x, 0)

    Example:
        >>> relu = ReLU()
        >>> output = relu(input)
    """

    def __init__(self, inplace=False):
        """Initialize ReLU.

        Args:
            inplace: Whether to apply in-place
        """
        relu = lib.cml_nn_relu(inplace)
        super().__init__(relu)


class Sigmoid(Module):
    """Sigmoid activation function.

    y = 1 / (1 + exp(-x))

    Example:
        >>> sigmoid = Sigmoid()
        >>> output = sigmoid(input)
    """

    def __init__(self, inplace=False):
        """Initialize Sigmoid.

        Args:
            inplace: Whether to apply in-place
        """
        sigmoid = lib.cml_nn_sigmoid(inplace)
        super().__init__(sigmoid)


class Tanh(Module):
    """Hyperbolic tangent activation.

    y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    Example:
        >>> tanh = Tanh()
        >>> output = tanh(input)
    """

    def __init__(self, inplace=False):
        """Initialize Tanh.

        Args:
            inplace: Whether to apply in-place
        """
        tanh = lib.cml_nn_tanh(inplace)
        super().__init__(tanh)


class Softmax(Module):
    """Softmax activation.

    Converts logits to probability distribution.

    Example:
        >>> softmax = Softmax(dim=1)
        >>> probs = softmax(logits)
    """

    def __init__(self, dim=1, inplace=False):
        """Initialize Softmax.

        Args:
            dim: Dimension to apply softmax
            inplace: Whether to apply in-place
        """
        softmax = lib.cml_nn_softmax(dim, inplace)
        super().__init__(softmax)


class Dropout(Module):
    """Dropout regularization.

    Randomly zeros out elements during training.

    Example:
        >>> dropout = Dropout(p=0.5)
        >>> output = dropout(input)
    """

    def __init__(self, p=0.5, inplace=False):
        """Initialize Dropout.

        Args:
            p: Probability of dropping (0.0 to 1.0)
            inplace: Whether to apply in-place
        """
        dropout = lib.cml_nn_dropout(p, inplace)
        super().__init__(dropout)


class BatchNorm2d(Module):
    """2D Batch Normalization.

    Normalizes input across batch and spatial dimensions.

    Example:
        >>> bn = BatchNorm2d(num_features=64)
        >>> output = bn(input)
    """

    def __init__(
        self,
        num_features,
        dtype=DTYPE_FLOAT32,
        device=DEVICE_CPU,
        momentum=0.1,
        epsilon=1e-5,
    ):
        """Initialize BatchNorm2d.

        Args:
            num_features: Number of channels
            dtype: Data type
            device: Device type
            momentum: Momentum for running statistics
            epsilon: Small constant for numerical stability
        """
        bn = lib.cml_nn_batchnorm2d(num_features, dtype, device, momentum, epsilon)
        super().__init__(bn)


class LayerNorm(Module):
    """Layer Normalization.

    Normalizes across features for each sample independently.

    Example:
        >>> ln = LayerNorm(normalized_shape=[256])
        >>> output = ln(input)
    """

    def __init__(self, normalized_shape, dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
        """Initialize LayerNorm.

        Args:
            normalized_shape: Shape to normalize (list/tuple)
            dtype: Data type
            device: Device type
        """
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        shape_array = ffi.new("int[]", normalized_shape)
        ln = lib.cml_nn_layernorm(shape_array, len(normalized_shape), dtype, device)
        super().__init__(ln)


class Conv2d(Module):
    """2D Convolution layer.

    Applies 2D convolution over input.

    Example:
        >>> conv = Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        >>> output = conv(input)  # [batch, 64, height, width]
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dtype=DTYPE_FLOAT32,
        device=DEVICE_CPU,
    ):
        """Initialize Conv2d.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolution kernel (int or tuple)
            stride: Stride of convolution
            padding: Padding to add
            dtype: Data type
            device: Device type
        """
        if isinstance(kernel_size, (list, tuple)):
            kernel_size = kernel_size[0]

        conv = lib.cml_nn_conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dtype, device
        )
        super().__init__(conv)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size


class MaxPool2d(Module):
    """2D Max Pooling layer.

    Takes maximum over a pooling window.

    Example:
        >>> pool = MaxPool2d(kernel_size=2, stride=2)
        >>> output = pool(input)
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        """Initialize MaxPool2d.

        Args:
            kernel_size: Size of pooling window
            stride: Stride of pooling (defaults to kernel_size)
            padding: Padding to add
        """
        if stride is None:
            stride = kernel_size

        pool = lib.cml_nn_maxpool2d(kernel_size, stride, padding)
        super().__init__(pool)


class AvgPool2d(Module):
    """2D Average Pooling layer.

    Takes average over a pooling window.

    Example:
        >>> pool = AvgPool2d(kernel_size=2, stride=2)
        >>> output = pool(input)
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        """Initialize AvgPool2d.

        Args:
            kernel_size: Size of pooling window
            stride: Stride of pooling (defaults to kernel_size)
            padding: Padding to add
        """
        if stride is None:
            stride = kernel_size

        pool = lib.cml_nn_avgpool2d(kernel_size, stride, padding)
        super().__init__(pool)
