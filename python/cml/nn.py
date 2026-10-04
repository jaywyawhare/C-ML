"""Neural network layers and modules."""

from cml._cml_lib import ffi, lib
from cml.core import Tensor, DTYPE_FLOAT32, DEVICE_CPU

UPSAMPLE_NEAREST = 0
UPSAMPLE_BILINEAR = 1
UPSAMPLE_BICUBIC = 2


class Parameter:
    """View over a C Parameter. The underlying tensor and struct stay owned by
    their module; this wrapper only reads them."""

    def __init__(self, c_param):
        """Wrap a C ``Parameter`` struct, which stays owned by its module."""
        self._param = c_param

    @property
    def name(self):
        """The parameter's name, or ``""`` if unnamed."""
        if self._param.name == ffi.NULL:
            return ""
        return ffi.string(self._param.name).decode()

    @property
    def requires_grad(self):
        """Whether autograd tracks gradients for this parameter."""
        return bool(self._param.requires_grad)

    @property
    def tensor(self):
        """The underlying tensor as a borrowed view, or ``None``."""
        t = self._param.tensor
        return Tensor.borrow(t) if t != ffi.NULL else None

    data = tensor

    @property
    def grad(self):
        """The parameter's gradient tensor, or ``None`` if not yet computed."""
        t = self._param.tensor
        if t == ffi.NULL or t.grad == ffi.NULL:
            return None
        return Tensor.borrow(t.grad)


class Module:
    def __init__(self, c_module):
        """Wrap a C module handle; this object owns and frees it unless reparented."""
        self._module = c_module
        # A module frees its own C handle unless it has been handed to a parent
        # container (e.g. Sequential) that owns and frees it - see add()/append().
        self._owned = True

    def _as_module(self):
        """Concrete layer handles are typed (Linear*, Sequential*, ...); the
        generic module_* C functions take the base Module*, so cast."""
        return ffi.cast("Module*", self._module)

    def __call__(self, input_tensor):
        """Call the module on ``input_tensor`` (delegates to ``forward``)."""
        return self.forward(input_tensor)

    def forward(self, input_tensor):
        """Run the module's forward pass and return the output tensor."""
        return Tensor(lib.cml_nn_module_forward(self._as_module(), input_tensor._tensor))

    def set_training(self, training=True):
        """Set the module's training flag (affects Dropout, BatchNorm, ...)."""
        lib.cml_nn_module_set_training(self._as_module(), training)

    def is_training(self):
        """Return whether the module is in training mode."""
        return lib.cml_nn_module_is_training(self._as_module())

    def parameters(self, recursive=True):
        """The module's parameters as a list of Parameter views (optimizers use
        the raw C array via optim._collect_parameters instead)."""
        params_out = ffi.new("Parameter***")
        num_out = ffi.new("int*")
        ret = lib.module_collect_parameters(self._as_module(), params_out, num_out, recursive)
        if ret != 0:
            raise RuntimeError("Failed to collect parameters from module")
        arr, n = params_out[0], num_out[0]
        result = [Parameter(arr[i]) for i in range(n)]
        if arr != ffi.NULL:
            lib.cml_free(arr)  # entries stay owned by their modules; only the array is ours
        return result

    def eval(self):
        """Put the module in evaluation mode and return it (torch.nn.Module.eval)."""
        lib.cml_nn_module_eval(self._as_module())
        return self

    def train(self, mode=True):
        """Put the module in training (or eval) mode and return it (torch.nn.Module.train)."""
        if mode:
            lib.cml_nn_module_train(self._as_module())
        else:
            lib.cml_nn_module_eval(self._as_module())
        return self

    def __del__(self):
        """Free the owned C module handle, unless a parent container owns it."""
        m = getattr(self, "_module", None)
        if m is not None and m != ffi.NULL and getattr(self, "_owned", False):
            # Layer handles are typed as their concrete struct (Linear*, ReLU*,
            # ...); module_free takes the base Module*, so cast. Modules owned by
            # a parent container are freed by that parent - don't double-free.
            lib.module_free(ffi.cast("Module*", m))
            self._module = ffi.NULL


class Sequential(Module):
    def __init__(self, *modules):
        """Sequential() or Sequential(layer1, layer2, ...) (PyTorch-style)."""
        super().__init__(lib.cml_nn_sequential())
        self.layers = []
        for m in modules:
            self.add(m)

    def add(self, layer):
        """Append a layer; the C Sequential takes ownership of it (torch.nn.Sequential.append)."""
        self._module = lib.cml_nn_sequential_add(
            ffi.cast("Sequential*", self._module),
            ffi.cast("Module*", layer._module),
        )
        # The Sequential (C side) now owns and frees this child.
        layer._owned = False
        self.layers.append(layer)
        return self

    def __len__(self):
        """Number of layers in the container."""
        return len(self.layers)

    def __getitem__(self, index):
        """Return the layer at ``index``."""
        return self.layers[index]


class ModuleList(Module):
    def __init__(self):
        """Create an empty module list (torch.nn.ModuleList)."""
        super().__init__(lib.cml_nn_module_list())
        self._children = []

    def append(self, module):
        """Append a module; the ModuleList takes ownership of it."""
        lib.module_list_append(
            ffi.cast("ModuleList*", self._module), ffi.cast("Module*", module._module))
        module._owned = False   # the ModuleList now owns and frees this child
        self._children.append(module)
        return self

    def insert(self, index, module):
        """Insert a module before ``index``; the ModuleList takes ownership of it."""
        lib.module_list_insert(
            ffi.cast("ModuleList*", self._module), index, ffi.cast("Module*", module._module))
        module._owned = False
        self._children.insert(index, module)

    def __getitem__(self, index):
        """Return the module at ``index``."""
        return self._children[index]

    def __len__(self):
        """Number of modules in the list."""
        return lib.module_list_length(self._module)

    def __iter__(self):
        """Iterate over the contained modules."""
        return iter(self._children)


class ModuleDict(Module):
    def __init__(self):
        """Create an empty module dict (torch.nn.ModuleDict)."""
        super().__init__(lib.cml_nn_module_dict())
        self._children = {}

    def __setitem__(self, key, module):
        """Register ``module`` under ``key``; the ModuleDict takes ownership of it."""
        lib.module_dict_add(
            ffi.cast("ModuleDict*", self._module),
            key.encode("utf-8"),
            ffi.cast("Module*", module._module))
        module._owned = False   # the ModuleDict now owns and frees this child
        self._children[key] = module

    def __getitem__(self, key):
        """Return the module registered under ``key``."""
        return self._children[key]

    def __contains__(self, key):
        """Return whether ``key`` is registered."""
        return key in self._children

    def __len__(self):
        """Number of modules in the dict."""
        return lib.module_dict_size(self._module)

    def keys(self):
        """The registered keys."""
        return self._children.keys()

    def values(self):
        """The registered modules."""
        return self._children.values()

    def items(self):
        """The ``(key, module)`` pairs."""
        return self._children.items()


class Linear(Module):
    def __init__(self, in_features, out_features, dtype=DTYPE_FLOAT32, device=DEVICE_CPU, bias=True):
        """Fully-connected layer ``y = x A^T + b`` (torch.nn.Linear)."""
        super().__init__(lib.cml_nn_linear(in_features, out_features, dtype, device, bias))
        self.in_features = in_features
        self.out_features = out_features


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=-1,
                 dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
        """Lookup table mapping indices to dense vectors (torch.nn.Embedding)."""
        super().__init__(lib.cml_nn_embedding(num_embeddings, embedding_dim, padding_idx, dtype, device))
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim


class ReLU(Module):
    def __init__(self, inplace=False):
        """Rectified linear unit activation (torch.nn.ReLU)."""
        super().__init__(lib.cml_nn_relu(inplace))


class Sigmoid(Module):
    def __init__(self):
        """Logistic sigmoid activation (torch.nn.Sigmoid)."""
        super().__init__(lib.cml_nn_sigmoid())


class Tanh(Module):
    def __init__(self):
        """Hyperbolic tangent activation (torch.nn.Tanh)."""
        super().__init__(lib.cml_nn_tanh())


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01, inplace=False):
        """Leaky rectified linear unit activation (torch.nn.LeakyReLU)."""
        super().__init__(lib.cml_nn_leaky_relu(negative_slope, inplace))


class PReLU(Module):
    def __init__(self, num_parameters=1, init=0.25, dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
        """Parametric ReLU with a learnable negative slope (torch.nn.PReLU)."""
        super().__init__(lib.cml_nn_prelu(num_parameters, init, dtype, device))


class Dropout(Module):
    def __init__(self, p=0.5, inplace=False):
        """Randomly zero elements during training (torch.nn.Dropout)."""
        super().__init__(lib.cml_nn_dropout(p, inplace))


class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        """Flatten a contiguous range of dims into one (torch.nn.Flatten)."""
        super().__init__(lib.cml_nn_flatten(start_dim, end_dim))


class Identity(Module):
    def __init__(self):
        """Pass the input through unchanged (torch.nn.Identity)."""
        super().__init__(lib.cml_nn_identity())


def _make_conv(c_fn_name, cls_name):
    """Build a convolution module class bound to the C constructor ``c_fn_name``."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, bias=True, dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
        """Convolution layer with scalar ``kernel_size`` (torch.nn.ConvNd)."""
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
    """Build a transposed-convolution module class bound to ``c_fn_name``."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, bias=True, dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
        """Transposed convolution layer with scalar ``kernel_size`` (torch.nn.ConvTransposeNd)."""
        if isinstance(kernel_size, (list, tuple)):
            kernel_size = kernel_size[0]
        c_fn = getattr(lib, c_fn_name)
        super(type(self), self).__init__(
            c_fn(in_channels, out_channels, kernel_size, stride, padding, output_padding,
                 bias, dtype, device))
        self.in_channels, self.out_channels, self.kernel_size = in_channels, out_channels, kernel_size
    return type(cls_name, (Module,), {'__init__': __init__})


ConvTranspose1d = _make_conv_transpose('cml_nn_conv_transpose1d', 'ConvTranspose1d')
ConvTranspose2d = _make_conv_transpose('cml_nn_conv_transpose2d', 'ConvTranspose2d')
ConvTranspose3d = _make_conv_transpose('cml_nn_conv_transpose3d', 'ConvTranspose3d')


def _make_batchnorm(c_fn_name, cls_name):
    """Build a batch-normalization module class bound to ``c_fn_name``."""
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
        """Batch normalization over ``num_features`` channels (torch.nn.BatchNormNd)."""
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
        """Layer normalization over ``normalized_shape`` (torch.nn.LayerNorm)."""
        if isinstance(normalized_shape, (list, tuple)):
            normalized_shape = normalized_shape[0]
        super().__init__(lib.cml_nn_layernorm(normalized_shape, eps, affine, dtype, device))


class LayerNorm2d(Module):
    def __init__(self, num_channels, eps=1e-5, affine=True,
                 dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
        """Channel-wise layer normalization for NCHW inputs."""
        super().__init__(lib.cml_nn_layernorm2d(num_channels, eps, affine, dtype, device))


class InstanceNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, affine=False,
                 dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
        """Per-sample, per-channel normalization (torch.nn.InstanceNorm2d)."""
        super().__init__(lib.cml_nn_instancenorm2d(num_features, eps, affine, dtype, device))


class GroupNorm(Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True,
                 dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
        """Group normalization over channel groups (torch.nn.GroupNorm)."""
        super().__init__(lib.cml_nn_groupnorm(num_groups, num_channels, eps, affine, dtype, device))


def _make_pool(c_fn_name, cls_name, has_dilation=False, has_count_include_pad=False):
    """Build a pooling module class bound to ``c_fn_name``, with the chosen options."""
    if has_dilation:
        def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
            """Max-pooling layer; ``stride`` defaults to ``kernel_size`` (torch.nn.MaxPoolNd)."""
            if stride is None:
                stride = kernel_size
            c_fn = getattr(lib, c_fn_name)
            super(type(self), self).__init__(c_fn(kernel_size, stride, padding, dilation, ceil_mode))
    elif has_count_include_pad:
        def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
            """Average-pooling layer; ``stride`` defaults to ``kernel_size`` (torch.nn.AvgPoolNd)."""
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
        """Adaptive average pooling to a fixed output size (torch.nn.AdaptiveAvgPool1d)."""
        super().__init__(lib.cml_nn_adaptive_avgpool1d(output_size))


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size):
        """Adaptive average pooling to a fixed output size (torch.nn.AdaptiveAvgPool2d)."""
        if isinstance(output_size, int):
            output_h = output_w = output_size
        else:
            output_h, output_w = output_size
        super().__init__(lib.cml_nn_adaptive_avgpool2d(output_h, output_w))


class AdaptiveMaxPool1d(Module):
    def __init__(self, output_size):
        """Adaptive max pooling to a fixed output size (torch.nn.AdaptiveMaxPool1d)."""
        super().__init__(lib.cml_nn_adaptive_maxpool1d(output_size))


class AdaptiveMaxPool2d(Module):
    def __init__(self, output_size):
        """Adaptive max pooling to a fixed output size (torch.nn.AdaptiveMaxPool2d)."""
        if isinstance(output_size, int):
            output_h = output_w = output_size
        else:
            output_h, output_w = output_size
        super().__init__(lib.cml_nn_adaptive_maxpool2d(output_h, output_w))


def _make_rnn_cell(c_fn_name, cls_name):
    """Build a recurrent-cell module class bound to ``c_fn_name``."""
    def __init__(self, input_size, hidden_size, bias=True,
                 dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
        """Single-step recurrent cell (torch.nn.RNNCell/LSTMCell/GRUCell)."""
        c_fn = getattr(lib, c_fn_name)
        super(type(self), self).__init__(c_fn(input_size, hidden_size, bias, dtype, device))
        self.input_size, self.hidden_size = input_size, hidden_size
    return type(cls_name, (Module,), {'__init__': __init__})


RNNCell = _make_rnn_cell('cml_nn_rnn_cell', 'RNNCell')
LSTMCell = _make_rnn_cell('cml_nn_lstm_cell', 'LSTMCell')
GRUCell = _make_rnn_cell('cml_nn_gru_cell', 'GRUCell')


def _make_rnn(c_fn_name, cls_name):
    """Build a recurrent-network module class bound to ``c_fn_name``."""
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False,
                 batch_first=True, dropout=0.0, bias=True,
                 dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
        """Multi-layer recurrent network (torch.nn.RNN/LSTM/GRU)."""
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
        """Multi-head scaled dot-product attention (torch.nn.MultiheadAttention)."""
        super().__init__(lib.cml_nn_multihead_attention(embed_dim, num_heads, dropout, dtype, device))
        self.embed_dim = embed_dim
        self.num_heads = num_heads


def _make_transformer_layer(c_fn_name, cls_name):
    """Build a transformer-layer module class bound to ``c_fn_name``."""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
        """Single transformer encoder/decoder layer (torch.nn.TransformerEncoderLayer)."""
        c_fn = getattr(lib, c_fn_name)
        super(type(self), self).__init__(
            c_fn(d_model, nhead, dim_feedforward, dropout, dtype, device))
    return type(cls_name, (Module,), {'__init__': __init__})


TransformerEncoderLayer = _make_transformer_layer(
    'cml_nn_transformer_encoder_layer', 'TransformerEncoderLayer')
TransformerDecoderLayer = _make_transformer_layer(
    'cml_nn_transformer_decoder_layer', 'TransformerDecoderLayer')


def _make_transformer(c_fn_name, cls_name):
    """Build a transformer-stack module class bound to ``c_fn_name``."""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 num_layers=6, dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
        """Stack of transformer layers (torch.nn.TransformerEncoder/TransformerDecoder)."""
        c_fn = getattr(lib, c_fn_name)
        super(type(self), self).__init__(
            c_fn(d_model, nhead, dim_feedforward, dropout, num_layers, dtype, device))
    return type(cls_name, (Module,), {'__init__': __init__})


TransformerEncoder = _make_transformer('cml_nn_transformer_encoder', 'TransformerEncoder')
TransformerDecoder = _make_transformer('cml_nn_transformer_decoder', 'TransformerDecoder')


class Upsample(Module):
    def __init__(self, scale_factor=None, output_size=None,
                 mode=UPSAMPLE_NEAREST, align_corners=False):
        """Upsample by ``scale_factor`` or to ``output_size`` (torch.nn.Upsample)."""
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
        """Rearrange channels into spatial resolution (torch.nn.PixelShuffle)."""
        super().__init__(lib.cml_nn_pixel_shuffle(upscale_factor))


class PixelUnshuffle(Module):
    def __init__(self, downscale_factor):
        """Rearrange spatial resolution into channels (torch.nn.PixelUnshuffle)."""
        super().__init__(lib.cml_nn_pixel_unshuffle(downscale_factor))
