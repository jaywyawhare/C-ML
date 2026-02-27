"""
CML Model Zoo - Python bindings for zoo/zoo.h

Pre-built model architectures with optional pretrained weights.
"""

from cml.core import _get_lib, DEVICE_CPU, DTYPE_FLOAT32
from cml.nn import Sequential, Linear, ReLU, Sigmoid, BatchNorm2d, LayerNorm, Conv2d, MaxPool2d, AvgPool2d, Dropout

# Model identifiers
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
    """Create MLP for MNIST classification.

    Architecture: 784 -> 256 -> 128 -> num_classes

    Args:
        num_classes: Number of output classes
        pretrained: Load pretrained weights
        dtype: Data type
        device: Device type

    Returns:
        Sequential model
    """
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
    """Create MLP for CIFAR-10 classification.

    Architecture: 3072 -> 512 -> 256 -> num_classes
    """
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
    """Create ResNet-18.

    Args:
        num_classes: Number of output classes
        pretrained: Load pretrained ImageNet weights
        dtype: Data type
        device: Device type
    """
    return _build_resnet([2, 2, 2, 2], num_classes, pretrained, "resnet18", dtype, device)


def resnet34(num_classes=1000, pretrained=False, dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
    """Create ResNet-34."""
    return _build_resnet([3, 4, 6, 3], num_classes, pretrained, "resnet34", dtype, device)


def resnet50(num_classes=1000, pretrained=False, dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
    """Create ResNet-50."""
    return _build_resnet([3, 4, 6, 3], num_classes, pretrained, "resnet50", dtype, device)


def vgg11(num_classes=1000, pretrained=False, dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
    """Create VGG-11."""
    cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    return _build_vgg(cfg, num_classes, pretrained, "vgg11", dtype, device)


def vgg16(num_classes=1000, pretrained=False, dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
    """Create VGG-16."""
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    return _build_vgg(cfg, num_classes, pretrained, "vgg16", dtype, device)


def gpt2_small(vocab_size=50257, pretrained=False, dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
    """Create GPT-2 small (124M params, 12 layers, 768 hidden)."""
    model = Sequential()

    # Embedding (simplified)
    model.add(Linear(vocab_size, 768, dtype=dtype, device=device, bias=False))

    # Transformer layers
    for _ in range(12):
        model.add(LayerNorm(768, dtype=dtype, device=device))
        model.add(Linear(768, 768, dtype=dtype, device=device))
        model.add(ReLU())
        model.add(Linear(768, 768, dtype=dtype, device=device))

    # LM head
    model.add(LayerNorm(768, dtype=dtype, device=device))
    model.add(Linear(768, vocab_size, dtype=dtype, device=device, bias=False))

    if pretrained:
        _try_load_pretrained(model, "gpt2_small")

    return model


def bert_tiny(vocab_size=30522, pretrained=False, dtype=DTYPE_FLOAT32, device=DEVICE_CPU):
    """Create BERT-tiny (4M params, 2 layers, 128 hidden)."""
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


# ===== Internal Helpers =====

def _build_resnet(layers, num_classes, pretrained, name, dtype, device):
    """Build a ResNet model."""
    model = Sequential()

    # Initial conv
    model.add(Conv2d(3, 64, kernel_size=7, stride=2, padding=3, dtype=dtype, device=device))
    model.add(BatchNorm2d(64, dtype=dtype, device=device))
    model.add(ReLU())
    model.add(MaxPool2d(kernel_size=3, stride=2, padding=1))

    # Residual layers
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

    # Classifier
    model.add(AvgPool2d(kernel_size=7))
    model.add(Linear(512, num_classes, dtype=dtype, device=device))

    if pretrained:
        _try_load_pretrained(model, name)

    return model


def _build_vgg(cfg, num_classes, pretrained, name, dtype, device):
    """Build a VGG model."""
    model = Sequential()

    in_channels = 3
    for v in cfg:
        if v == 'M':
            model.add(MaxPool2d(kernel_size=2, stride=2))
        else:
            model.add(Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device))
            model.add(ReLU())
            in_channels = v

    # Classifier
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
    """Try to load pretrained weights."""
    import os
    weights_dir = os.environ.get("CML_WEIGHTS_DIR",
                                  os.path.expanduser("~/.cml/weights"))
    path = os.path.join(weights_dir, f"{name}.bin")

    if os.path.exists(path):
        try:
            lib = _get_lib()
            if hasattr(lib, 'model_load'):
                lib.model_load(model._handle, path.encode())
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights for {name}: {e}")
    else:
        print(f"Note: Pretrained weights not found at {path}. "
              f"Download with: cml.zoo.download_weights('{name}')")


def download_weights(model_name, weights_dir=None):
    """Download pretrained weights for a model.

    Args:
        model_name: Model name (e.g., "resnet18", "mlp_mnist")
        weights_dir: Directory to store weights (default: ~/.cml/weights/)
    """
    import os
    import urllib.request

    if weights_dir is None:
        weights_dir = os.environ.get("CML_WEIGHTS_DIR",
                                      os.path.expanduser("~/.cml/weights"))

    os.makedirs(weights_dir, exist_ok=True)
    path = os.path.join(weights_dir, f"{model_name}.bin")

    if os.path.exists(path):
        print(f"Weights already cached: {path}")
        return path

    base_url = os.environ.get("CML_WEIGHTS_URL", "https://weights.cml-lib.org/v1")
    url = f"{base_url}/{model_name}.bin"

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
