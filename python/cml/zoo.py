"""Pre-built model architectures."""

from cml.core import _get_lib, DEVICE_CPU, DTYPE_FLOAT32
from cml.nn import Sequential, Linear, ReLU, Sigmoid, BatchNorm2d, LayerNorm, Conv2d, MaxPool2d, AvgPool2d, Dropout

MLP_MNIST = 0
MLP_CIFAR10 = 1
RESNET18 = 2
RESNET34 = 3
RESNET50 = 4
VGG11 = 5
VGG16 = 6
GPT2_SMALL = 7
BERT_TINY = 8


def mlp_mnist(num_classes=10, pretrained=False, dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
    """Three-layer ReLU MLP sized for flattened 28x28 MNIST inputs (784 -> 256 -> 128 -> classes)."""
    model = Sequential()
    model.add(Linear(784, 256, dtype=dtype, device=device))
    model.add(ReLU())
    model.add(Linear(256, 128, dtype=dtype, device=device))
    model.add(ReLU())
    model.add(Linear(128, num_classes, dtype=dtype, device=device))

    if pretrained:
        _try_load_pretrained(model, "mlp_mnist")

    return model


def mlp_cifar10(num_classes=10, pretrained=False, dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
    """Three-layer ReLU MLP sized for flattened 3x32x32 CIFAR-10 inputs (3072 -> 512 -> 256 -> classes)."""
    model = Sequential()
    model.add(Linear(3072, 512, dtype=dtype, device=device))
    model.add(ReLU())
    model.add(Linear(512, 256, dtype=dtype, device=device))
    model.add(ReLU())
    model.add(Linear(256, num_classes, dtype=dtype, device=device))

    if pretrained:
        _try_load_pretrained(model, "mlp_cifar10")

    return model


def resnet18(num_classes=1000, pretrained=False, dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
    """ResNet-18 (torchvision.models.resnet18); ``[2, 2, 2, 2]`` block layout.

    Blocks are plain Conv-BN-ReLU pairs here; the identity skip connections are not wired up.
    """
    return _build_resnet([2, 2, 2, 2], num_classes, pretrained, "resnet18", dtype, device)


def resnet34(num_classes=1000, pretrained=False, dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
    """ResNet-34 (torchvision.models.resnet34); ``[3, 4, 6, 3]`` block layout."""
    return _build_resnet([3, 4, 6, 3], num_classes, pretrained, "resnet34", dtype, device)


def resnet50(num_classes=1000, pretrained=False, dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
    """ResNet-50 (torchvision.models.resnet50); built from the same ``[3, 4, 6, 3]`` basic blocks as ResNet-34 (no bottleneck)."""
    return _build_resnet([3, 4, 6, 3], num_classes, pretrained, "resnet50", dtype, device)


def vgg11(num_classes=1000, pretrained=False, dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
    """VGG-11 (torchvision.models.vgg11); ``'M'`` entries in the config denote max-pool stages."""
    cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    return _build_vgg(cfg, num_classes, pretrained, "vgg11", dtype, device)


def vgg16(num_classes=1000, pretrained=False, dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
    """VGG-16 (torchvision.models.vgg16); ``'M'`` entries in the config denote max-pool stages."""
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    return _build_vgg(cfg, num_classes, pretrained, "vgg16", dtype, device)


def gpt2_small(vocab_size=50257, pretrained=False, dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
    """GPT-2 small-sized stack: 12 LayerNorm/MLP blocks at width 768, tied to ``vocab_size`` ends.

    A simplified feed-forward stand-in; it has no attention or positional embeddings.
    """
    model = Sequential()
    model.add(Linear(vocab_size, 768, dtype=dtype, device=device, bias=False))

    for _ in range(12):
        model.add(LayerNorm(768, dtype=dtype, device=device))
        model.add(Linear(768, 768, dtype=dtype, device=device))
        model.add(ReLU())
        model.add(Linear(768, 768, dtype=dtype, device=device))

    model.add(LayerNorm(768, dtype=dtype, device=device))
    model.add(Linear(768, vocab_size, dtype=dtype, device=device, bias=False))

    if pretrained:
        _try_load_pretrained(model, "gpt2_small")

    return model


def bert_tiny(vocab_size=30522, pretrained=False, dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
    """BERT-tiny-sized stack: 2 LayerNorm/MLP blocks at width 128, tied to ``vocab_size`` ends.

    A simplified feed-forward stand-in; it has no attention or positional embeddings.
    """
    model = Sequential()
    model.add(Linear(vocab_size, 128, dtype=dtype, device=device, bias=False))

    for _ in range(2):
        model.add(LayerNorm(128, dtype=dtype, device=device))
        model.add(Linear(128, 128, dtype=dtype, device=device))
        model.add(ReLU())
        model.add(Linear(128, 128, dtype=dtype, device=device))

    model.add(LayerNorm(128, dtype=dtype, device=device))
    model.add(Linear(128, vocab_size, dtype=dtype, device=device, bias=False))

    if pretrained:
        _try_load_pretrained(model, "bert_tiny")

    return model


def _build_resnet(layers, num_classes, pretrained, name, dtype, device):
    """Assemble a ResNet stem + ``layers`` stages of Conv-BN-ReLU blocks into a ``Sequential``.

    ``layers`` gives the block count per stage; the first block of each later stage downsamples.
    """
    model = Sequential()

    model.add(Conv2d(3, 64, kernel_size=7, stride=2, padding=3, dtype=dtype, device=device))
    model.add(BatchNorm2d(64, dtype=dtype, device=device))
    model.add(ReLU())
    model.add(MaxPool2d(kernel_size=3, stride=2, padding=1))

    channels = [64, 128, 256, 512]
    for layer_idx, num_blocks in enumerate(layers):
        in_ch = 64 if layer_idx == 0 else channels[layer_idx - 1]
        out_ch = channels[layer_idx]

        for block in range(num_blocks):
            stride = 2 if block == 0 and layer_idx > 0 else 1
            block_in = in_ch if block == 0 else out_ch

            model.add(Conv2d(block_in, out_ch, kernel_size=3, stride=stride, padding=1, dtype=dtype, device=device))
            model.add(BatchNorm2d(out_ch, dtype=dtype, device=device))
            model.add(ReLU())
            model.add(Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device))
            model.add(BatchNorm2d(out_ch, dtype=dtype, device=device))
            model.add(ReLU())

    model.add(AvgPool2d(kernel_size=7))
    model.add(Linear(512, num_classes, dtype=dtype, device=device))

    if pretrained:
        _try_load_pretrained(model, name)

    return model


def _build_vgg(cfg, num_classes, pretrained, name, dtype, device):
    """Assemble a VGG feature extractor + 3-layer classifier from ``cfg`` into a ``Sequential``.

    Integer entries in ``cfg`` are conv channel counts; ``'M'`` inserts a 2x2 max-pool.
    """
    model = Sequential()

    in_channels = 3
    for v in cfg:
        if v == 'M':
            model.add(MaxPool2d(kernel_size=2, stride=2))
        else:
            model.add(Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device))
            model.add(ReLU())
            in_channels = v

    model.add(Linear(512 * 7 * 7, 4096, dtype=dtype, device=device))
    model.add(ReLU())
    model.add(Dropout(0.5))
    model.add(Linear(4096, 4096, dtype=dtype, device=device))
    model.add(ReLU())
    model.add(Dropout(0.5))
    model.add(Linear(4096, num_classes, dtype=dtype, device=device))

    if pretrained:
        _try_load_pretrained(model, name)

    return model


def _try_load_pretrained(model, name):
    """Load cached pretrained weights into `model`.

    Weights live at ``$CML_WEIGHTS_DIR/<name>.safetensors`` (default
    ``~/.cml/weights``) and are loaded through the real safetensors loader, which
    matches parameters by their dotted module names and checks shapes. The old
    path looked for a ``.bin`` and called a ``model_load`` C entry point that does
    not exist, so `pretrained=True` silently loaded nothing.
    """
    import os
    from . import safetensors as _st

    weights_dir = os.environ.get("CML_WEIGHTS_DIR",
                                 os.path.expanduser("~/.cml/weights"))
    path = os.path.join(weights_dir, f"{name}.safetensors")

    if os.path.exists(path):
        try:
            loaded = _st.load_pretrained(model, path)
            print(f"Loaded {loaded} pretrained tensors for '{name}' from {path}")
        except Exception as e:  # noqa: BLE001 - surface any load failure, don't crash construction
            print(f"Warning: failed to load pretrained weights for '{name}': {e}")
    else:
        print(f"Note: pretrained weights not found at {path}. "
              f"Download with: cml.zoo.download_weights('{name}')")


def download_weights(model_name, weights_dir=None):
    """Fetch ``<model_name>.safetensors`` into the weights cache, returning its path (``None`` on failure).

    Caches under ``$CML_WEIGHTS_DIR`` (default ``~/.cml/weights``) from ``$CML_WEIGHTS_URL``;
    skips the download if already present.
    """
    import os
    import urllib.request

    if weights_dir is None:
        weights_dir = os.environ.get("CML_WEIGHTS_DIR",
                                      os.path.expanduser("~/.cml/weights"))

    os.makedirs(weights_dir, exist_ok=True)
    path = os.path.join(weights_dir, f"{model_name}.safetensors")

    if os.path.exists(path):
        print(f"Weights already cached: {path}")
        return path

    base_url = os.environ.get("CML_WEIGHTS_URL", "https://weights.cml-lib.org/v1")
    url = f"{base_url}/{model_name}.safetensors"

    print(f"Downloading weights: {url} -> {path}")
    try:
        urllib.request.urlretrieve(url, path)
        print(f"Weights downloaded: {path}")
        return path
    except Exception as e:
        print(f"Failed to download weights: {e}")
        return None


__all__ = [
    "mlp_mnist",
    "mlp_cifar10",
    "resnet18",
    "resnet34",
    "resnet50",
    "vgg11",
    "vgg16",
    "gpt2_small",
    "bert_tiny",
    "download_weights",
]
