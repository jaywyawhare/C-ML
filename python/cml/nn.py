"""Neural network layers and modules."""

from cml._cml_lib import ffi, lib
from cml.core import Tensor, DTYPE_FLOAT32, DEVICE_CPU


class Module:
    """Base class for neural network modules."""

    def __init__(self, c_module):
        self._module = c_module

    def __call__(self, input_tensor):
        return self.forward(input_tensor)

    def forward(self, input_tensor):
        result = lib.cml_nn_module_forward(self._module, input_tensor._tensor)
        return Tensor(result)

    def set_training(self, training=True):
        lib.cml_nn_module_set_training(self._module, training)

    def __del__(self):
        if self._module != ffi.NULL:
            lib.module_free(self._module)


class Sequential(Module):
    """Applies modules in sequence: y = module_n(...module_2(module_1(x)))."""

    def __init__(self):
        seq = lib.cml_nn_sequential()
        super().__init__(seq)
        self.layers = []

    def add(self, layer):
        self._module = lib.cml_nn_sequential_add(self._module, layer._module)
        self.layers.append(layer)
        return self

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, index):
        return self.layers[index]


class Linear(Module):
    """Fully connected layer: y = xW + b."""

    def __init__(
        self,
        in_features,
        out_features,
        dtype=DTYPE_FLOAT32,
        device=DEVICE_CPU,
        bias=True,
    ):
        linear = lib.cml_nn_linear(in_features, out_features, dtype, device, bias)
        super().__init__(linear)
        self.in_features = in_features
        self.out_features = out_features


class ReLU(Module):
    def __init__(self, inplace=False):
        super().__init__(lib.cml_nn_relu(inplace))


class Sigmoid(Module):
    def __init__(self, inplace=False):
        super().__init__(lib.cml_nn_sigmoid(inplace))


class Tanh(Module):
    def __init__(self, inplace=False):
        super().__init__(lib.cml_nn_tanh(inplace))


class Softmax(Module):
    def __init__(self, dim=1, inplace=False):
        super().__init__(lib.cml_nn_softmax(dim, inplace))


class Dropout(Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__(lib.cml_nn_dropout(p, inplace))


class BatchNorm2d(Module):
    def __init__(
        self,
        num_features,
        dtype=DTYPE_FLOAT32,
        device=DEVICE_CPU,
        momentum=0.1,
        epsilon=1e-5,
    ):
        bn = lib.cml_nn_batchnorm2d(num_features, dtype, device, momentum, epsilon)
        super().__init__(bn)


class LayerNorm(Module):
    def __init__(self, normalized_shape, dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        shape_array = ffi.new("int[]", normalized_shape)
        ln = lib.cml_nn_layernorm(shape_array, len(normalized_shape), dtype, device)
        super().__init__(ln)


class Conv2d(Module):
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
    def __init__(self, kernel_size, stride=None, padding=0):
        if stride is None:
            stride = kernel_size
        super().__init__(lib.cml_nn_maxpool2d(kernel_size, stride, padding))


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        if stride is None:
            stride = kernel_size
        super().__init__(lib.cml_nn_avgpool2d(kernel_size, stride, padding))
