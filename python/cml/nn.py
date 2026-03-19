"""Neural network layers and modules."""

from cml._cml_lib import ffi, lib
from cml.core import Tensor, DTYPE_FLOAT32, DEVICE_CPU

UPSAMPLE_NEAREST = 0
UPSAMPLE_BILINEAR = 1
UPSAMPLE_BICUBIC = 2


class Module:
    def __init__(self, c_module):
        self._module = c_module

    def __call__(self, input_tensor):
        return self.forward(input_tensor)

    def forward(self, input_tensor):
        return Tensor(lib.cml_nn_module_forward(self._module, input_tensor._tensor))

    def set_training(self, training=True):
        lib.cml_nn_module_set_training(self._module, training)

    def is_training(self):
        return lib.cml_nn_module_is_training(self._module)

    def eval(self):
        lib.cml_nn_module_eval(self._module)
        return self

    def train(self, mode=True):
        if mode:
            lib.cml_nn_module_train(self._module)
        else:
            lib.cml_nn_module_eval(self._module)
        return self

    def __del__(self):
        if self._module != ffi.NULL:
            lib.module_free(self._module)


class Sequential(Module):
    def __init__(self):
        super().__init__(lib.cml_nn_sequential())
        self.layers = []

    def add(self, layer):
        self._module = lib.cml_nn_sequential_add(self._module, layer._module)
        self.layers.append(layer)
        return self

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, index):
        return self.layers[index]


class ModuleList(Module):
    def __init__(self):
        super().__init__(lib.cml_nn_module_list())
        self._children = []

    def append(self, module):
        lib.module_list_append(self._module, module._module)
        self._children.append(module)
        return self

    def insert(self, index, module):
        lib.module_list_insert(self._module, index, module._module)
        self._children.insert(index, module)

    def __getitem__(self, index):
        return self._children[index]

    def __len__(self):
        return lib.module_list_length(self._module)

    def __iter__(self):
        return iter(self._children)


class ModuleDict(Module):
    def __init__(self):
        super().__init__(lib.cml_nn_module_dict())
        self._children = {}

    def __setitem__(self, key, module):
        lib.module_dict_add(self._module, key.encode("utf-8"), module._module)
        self._children[key] = module

    def __getitem__(self, key):
        return self._children[key]

    def __contains__(self, key):
        return key in self._children

    def __len__(self):
        return lib.module_dict_size(self._module)

    def keys(self):
        return self._children.keys()

    def values(self):
        return self._children.values()

    def items(self):
        return self._children.items()


class Linear(Module):
    def __init__(self, in_features, out_features, dtype=DTYPE_FLOAT32, device=DEVICE_CPU, bias=True):
        super().__init__(lib.cml_nn_linear(in_features, out_features, dtype, device, bias))
        self.in_features = in_features
        self.out_features = out_features


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=-1,
                 dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
        super().__init__(lib.cml_nn_embedding(num_embeddings, embedding_dim, padding_idx, dtype, device))
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim


class ReLU(Module):
    def __init__(self, inplace=False):
        super().__init__(lib.cml_nn_relu(inplace))


class Sigmoid(Module):
    def __init__(self):
        super().__init__(lib.cml_nn_sigmoid())


class Tanh(Module):
    def __init__(self):
        super().__init__(lib.cml_nn_tanh())


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01, inplace=False):
        super().__init__(lib.cml_nn_leaky_relu(negative_slope, inplace))


class PReLU(Module):
    def __init__(self, num_parameters=1, init=0.25, dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
        super().__init__(lib.cml_nn_prelu(num_parameters, init, dtype, device))


class Dropout(Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__(lib.cml_nn_dropout(p, inplace))


class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__(lib.cml_nn_flatten(start_dim, end_dim))


class Identity(Module):
    def __init__(self):
        super().__init__(lib.cml_nn_identity())


def _make_conv(c_fn_name, cls_name):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, bias=True, dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
        if isinstance(kernel_size, (list, tuple)):
            kernel_size = kernel_size[0]
        c_fn = getattr(lib, c_fn_name)
        super(type(self), self).__init__(
            c_fn(in_channels, out_channels, kernel_size, stride, padding, dilation, bias, dtype, device))
        self.in_channels, self.out_channels, self.kernel_size = in_channels, out_channels, kernel_size
    return type(cls_name, (Module,), {'__init__': __init__})


Conv1d = _make_conv('cml_nn_conv1d', 'Conv1d')
Conv2d = _make_conv('cml_nn_conv2d', 'Conv2d')
Conv3d = _make_conv('cml_nn_conv3d', 'Conv3d')


def _make_conv_transpose(c_fn_name, cls_name):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, bias=True, dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
        if isinstance(kernel_size, (list, tuple)):
            kernel_size = kernel_size[0]
        c_fn = getattr(lib, c_fn_name)
        super(type(self), self).__init__(
            c_fn(in_channels, out_channels, kernel_size, stride, padding, output_padding,
                 bias, dtype, device))
        self.in_channels, self.out_channels, self.kernel_size = in_channels, out_channels, kernel_size
    return type(cls_name, (Module,), {'__init__': __init__})


ConvTranspose1d = _make_conv_transpose('cml_nn_conv_transpose1d', 'ConvTranspose1d')
ConvTranspose3d = _make_conv_transpose('cml_nn_conv_transpose3d', 'ConvTranspose3d')


def _make_batchnorm(c_fn_name, cls_name):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
        c_fn = getattr(lib, c_fn_name)
        super(type(self), self).__init__(
            c_fn(num_features, eps, momentum, affine, track_running_stats, dtype, device))
    return type(cls_name, (Module,), {'__init__': __init__})


BatchNorm1d = _make_batchnorm('cml_nn_batchnorm1d', 'BatchNorm1d')
BatchNorm2d = _make_batchnorm('cml_nn_batchnorm2d', 'BatchNorm2d')
BatchNorm3d = _make_batchnorm('cml_nn_batchnorm3d', 'BatchNorm3d')


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, affine=True,
                 dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
        if isinstance(normalized_shape, (list, tuple)):
            normalized_shape = normalized_shape[0]
        super().__init__(lib.cml_nn_layernorm(normalized_shape, eps, affine, dtype, device))


class LayerNorm2d(Module):
    def __init__(self, num_channels, eps=1e-5, affine=True,
                 dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
        super().__init__(lib.cml_nn_layernorm2d(num_channels, eps, affine, dtype, device))


class InstanceNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, affine=False,
                 dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
        super().__init__(lib.cml_nn_instancenorm2d(num_features, eps, affine, dtype, device))


class GroupNorm(Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True,
                 dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
        super().__init__(lib.cml_nn_groupnorm(num_groups, num_channels, eps, affine, dtype, device))


def _make_pool(c_fn_name, cls_name, has_dilation=False, has_count_include_pad=False):
    if has_dilation:
        def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
            if stride is None:
                stride = kernel_size
            c_fn = getattr(lib, c_fn_name)
            super(type(self), self).__init__(c_fn(kernel_size, stride, padding, dilation, ceil_mode))
    elif has_count_include_pad:
        def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
            if stride is None:
                stride = kernel_size
            c_fn = getattr(lib, c_fn_name)
            super(type(self), self).__init__(c_fn(kernel_size, stride, padding, ceil_mode, count_include_pad))
    else:
        raise ValueError("Pool factory requires has_dilation or has_count_include_pad")
    return type(cls_name, (Module,), {'__init__': __init__})


MaxPool1d = _make_pool('cml_nn_maxpool1d', 'MaxPool1d', has_dilation=True)
MaxPool2d = _make_pool('cml_nn_maxpool2d', 'MaxPool2d', has_dilation=True)
MaxPool3d = _make_pool('cml_nn_maxpool3d', 'MaxPool3d', has_dilation=True)

AvgPool1d = _make_pool('cml_nn_avgpool1d', 'AvgPool1d', has_count_include_pad=True)
AvgPool2d = _make_pool('cml_nn_avgpool2d', 'AvgPool2d', has_count_include_pad=True)
AvgPool3d = _make_pool('cml_nn_avgpool3d', 'AvgPool3d', has_count_include_pad=True)


class AdaptiveAvgPool1d(Module):
    def __init__(self, output_size):
        super().__init__(lib.cml_nn_adaptive_avgpool1d(output_size))


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size):
        if isinstance(output_size, int):
            output_h = output_w = output_size
        else:
            output_h, output_w = output_size
        super().__init__(lib.cml_nn_adaptive_avgpool2d(output_h, output_w))


class AdaptiveMaxPool1d(Module):
    def __init__(self, output_size):
        super().__init__(lib.cml_nn_adaptive_maxpool1d(output_size))


class AdaptiveMaxPool2d(Module):
    def __init__(self, output_size):
        if isinstance(output_size, int):
            output_h = output_w = output_size
        else:
            output_h, output_w = output_size
        super().__init__(lib.cml_nn_adaptive_maxpool2d(output_h, output_w))


def _make_rnn_cell(c_fn_name, cls_name):
    def __init__(self, input_size, hidden_size, bias=True,
                 dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
        c_fn = getattr(lib, c_fn_name)
        super(type(self), self).__init__(c_fn(input_size, hidden_size, bias, dtype, device))
        self.input_size, self.hidden_size = input_size, hidden_size
    return type(cls_name, (Module,), {'__init__': __init__})


RNNCell = _make_rnn_cell('cml_nn_rnn_cell', 'RNNCell')
LSTMCell = _make_rnn_cell('cml_nn_lstm_cell', 'LSTMCell')
GRUCell = _make_rnn_cell('cml_nn_gru_cell', 'GRUCell')


def _make_rnn(c_fn_name, cls_name):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False,
                 batch_first=True, dropout=0.0, bias=True,
                 dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
        c_fn = getattr(lib, c_fn_name)
        super(type(self), self).__init__(
            c_fn(input_size, hidden_size, num_layers, bidirectional, batch_first,
                 dropout, bias, dtype, device))
        self.input_size, self.hidden_size, self.num_layers = input_size, hidden_size, num_layers
    return type(cls_name, (Module,), {'__init__': __init__})


RNN = _make_rnn('cml_nn_rnn', 'RNN')
LSTM = _make_rnn('cml_nn_lstm', 'LSTM')
GRU = _make_rnn('cml_nn_gru', 'GRU')


class MultiHeadAttention(Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0,
                 dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
        super().__init__(lib.cml_nn_multihead_attention(embed_dim, num_heads, dropout, dtype, device))
        self.embed_dim = embed_dim
        self.num_heads = num_heads


def _make_transformer_layer(c_fn_name, cls_name):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
        c_fn = getattr(lib, c_fn_name)
        super(type(self), self).__init__(
            c_fn(d_model, nhead, dim_feedforward, dropout, dtype, device))
    return type(cls_name, (Module,), {'__init__': __init__})


TransformerEncoderLayer = _make_transformer_layer(
    'cml_nn_transformer_encoder_layer', 'TransformerEncoderLayer')
TransformerDecoderLayer = _make_transformer_layer(
    'cml_nn_transformer_decoder_layer', 'TransformerDecoderLayer')


def _make_transformer(c_fn_name, cls_name):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 num_layers=6, dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
        c_fn = getattr(lib, c_fn_name)
        super(type(self), self).__init__(
            c_fn(d_model, nhead, dim_feedforward, dropout, num_layers, dtype, device))
    return type(cls_name, (Module,), {'__init__': __init__})


TransformerEncoder = _make_transformer('cml_nn_transformer_encoder', 'TransformerEncoder')
TransformerDecoder = _make_transformer('cml_nn_transformer_decoder', 'TransformerDecoder')


class Upsample(Module):
    def __init__(self, scale_factor=None, output_size=None,
                 mode=UPSAMPLE_NEAREST, align_corners=False):
        if output_size is not None:
            if isinstance(output_size, int):
                output_size = [output_size]
            c_output_size = ffi.new("int[]", output_size)
            num_output_dims = len(output_size)
        else:
            c_output_size = ffi.NULL
            num_output_dims = 0
        if scale_factor is None:
            scale_factor = 0.0
        super().__init__(lib.cml_nn_upsample(scale_factor, c_output_size, num_output_dims, mode, align_corners))


class PixelShuffle(Module):
    def __init__(self, upscale_factor):
        super().__init__(lib.cml_nn_pixel_shuffle(upscale_factor))


class PixelUnshuffle(Module):
    def __init__(self, downscale_factor):
        super().__init__(lib.cml_nn_pixel_unshuffle(downscale_factor))
