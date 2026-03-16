"""Data loading and batch processing utilities."""

from typing import Tuple, List, Optional, Union
import cml
from cml.core import Tensor


class Dataset:
    """Dataset wrapper pairing features with optional targets."""

    def __init__(self, X: Tensor, y: Optional[Tensor] = None):
        self.X = X
        self.y = y
        self.size = X.size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: Union[int, slice]) -> Tuple[Tensor, Optional[Tensor]]:
        if isinstance(idx, slice):
            start = idx.start or 0
            stop = idx.stop or self.size
            X_batch = self.X.slice(start, stop)
            y_batch = self.y.slice(start, stop) if self.y else None
            return X_batch, y_batch
        else:
            # Single item
            X_item = self.X.slice(idx, idx + 1)
            y_item = self.y.slice(idx, idx + 1) if self.y else None
            return X_item, y_item


class DataLoader:
    """Iterates through a dataset in batches."""

    def __init__(
        self,
        dataset: Union[Dataset, Tensor],
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 0,
    ):
        if isinstance(dataset, Tensor):
            self.dataset = Dataset(dataset)
        else:
            self.dataset = dataset

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.indices = list(range(len(self.dataset)))

        if shuffle:
            import random
            random.shuffle(self.indices)

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch_size = min(self.batch_size, len(self.dataset) - i)
            X_batch, y_batch = self.dataset[i : i + batch_size]
            yield X_batch, y_batch

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def create_dataset(X: Tensor, y: Optional[Tensor] = None) -> Dataset:
    """Create a dataset from tensors."""
    return Dataset(X, y)


def create_dataloader(
    dataset: Union[Dataset, Tensor], batch_size: int = 32, shuffle: bool = False
) -> DataLoader:
    """Create a data loader."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_test_split(
    X: Tensor,
    y: Optional[Tensor] = None,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple:
    """Split data into train and test sets, returning (X_train, X_test[, y_train, y_test])."""
    if random_state is not None:
        cml.seed(random_state)

    num_samples = X.size  # Simplified
    num_test = int(num_samples * test_size)
    num_train = num_samples - num_test

    X_train = X.slice(0, num_train)
    X_test = X.slice(num_train, num_samples)

    if y is not None:
        y_train = y.slice(0, num_train)
        y_test = y.slice(num_train, num_samples)
        return X_train, X_test, y_train, y_test
    else:
        return X_train, X_test


def normalize(
    X: Tensor, mean: Optional[Tensor] = None, std: Optional[Tensor] = None
) -> Tensor:
    """Normalize features to zero mean and unit variance."""
    import numpy as np
    arr = X.numpy()
    if mean is None:
        mean_val = np.mean(arr, axis=0)
    else:
        mean_val = mean.numpy() if hasattr(mean, 'numpy') else mean
    if std is None:
        std_val = np.std(arr, axis=0)
        std_val = np.where(std_val == 0, 1.0, std_val)  # avoid division by zero
    else:
        std_val = std.numpy() if hasattr(std, 'numpy') else std
    result = (arr - mean_val) / std_val
    return Tensor.from_numpy(result.astype(np.float32))


def standardize(X: Tensor) -> Tensor:
    """Standardize features (alias for normalize)."""
    return normalize(X)


def minmax_scale(X: Tensor, min_val: float = 0.0, max_val: float = 1.0) -> Tensor:
    """Scale features to [min_val, max_val] range."""
    import numpy as np
    arr = X.numpy()
    x_min = np.min(arr, axis=0)
    x_max = np.max(arr, axis=0)
    denom = x_max - x_min
    denom = np.where(denom == 0, 1.0, denom)
    scaled = (arr - x_min) / denom
    result = scaled * (max_val - min_val) + min_val
    return Tensor.from_numpy(result.astype(np.float32))


def one_hot_encode(labels: Tensor, num_classes: int) -> Tensor:
    """Convert class labels to one-hot encoding."""
    import numpy as np
    arr = labels.numpy().flatten().astype(int)
    one_hot = np.zeros((len(arr), num_classes), dtype=np.float32)
    one_hot[np.arange(len(arr)), arr] = 1.0
    return Tensor.from_numpy(one_hot)


def split_into_batches(
    X: Tensor, y: Optional[Tensor] = None, batch_size: int = 32
) -> List[Tuple]:
    """Split data into a list of (X_batch, y_batch) tuples."""
    batches = []
    num_samples = X.size

    for i in range(0, num_samples, batch_size):
        end = min(i + batch_size, num_samples)
        X_batch = X.slice(i, end)
        y_batch = y.slice(i, end) if y is not None else None
        batches.append((X_batch, y_batch))

    return batches


def _try_sklearn_load(name: str):
    try:
        import sklearn.datasets
        loader = getattr(sklearn.datasets, f"load_{name}", None)
        if loader:
            data = loader()
            return data.data, data.target
    except ImportError:
        pass
    return None, None


def load_iris() -> Tuple[Tensor, Tensor]:
    """Load Iris dataset as (X, y) tensors. Falls back to synthetic data."""
    import numpy as np

    X_np, y_np = _try_sklearn_load("iris")
    if X_np is not None:
        X = Tensor.from_numpy(np.array(X_np, dtype=np.float32))
        y = Tensor.from_numpy(np.array(y_np, dtype=np.float32))
        return X, y

    # Fallback: generate synthetic iris-like data
    np.random.seed(42)
    X_np = np.random.randn(150, 4).astype(np.float32) * 0.5
    X_np[:50] += np.array([5.0, 3.4, 1.4, 0.2])
    X_np[50:100] += np.array([5.9, 2.8, 4.3, 1.3])
    X_np[100:] += np.array([6.6, 3.0, 5.6, 2.0])
    y_np = np.array([0]*50 + [1]*50 + [2]*50, dtype=np.float32)

    X = Tensor.from_numpy(X_np)
    y = Tensor.from_numpy(y_np)
    return X, y


def load_mnist() -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
    """Load MNIST dataset as ((X_train, y_train), (X_test, y_test)). Falls back to synthetic data."""
    import numpy as np

    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X = np.array(mnist.data, dtype=np.float32) / 255.0
        y = np.array(mnist.target, dtype=np.float32)
        X_train = Tensor.from_numpy(X[:60000])
        y_train = Tensor.from_numpy(y[:60000])
        X_test = Tensor.from_numpy(X[60000:])
        y_test = Tensor.from_numpy(y[60000:])
        return (X_train, y_train), (X_test, y_test)
    except (ImportError, Exception):
        pass

    # Fallback: small synthetic data
    np.random.seed(42)
    X_train = Tensor.from_numpy(np.random.randn(1000, 784).astype(np.float32) * 0.1)
    y_train = Tensor.from_numpy(np.random.randint(0, 10, 1000).astype(np.float32))
    X_test = Tensor.from_numpy(np.random.randn(200, 784).astype(np.float32) * 0.1)
    y_test = Tensor.from_numpy(np.random.randint(0, 10, 200).astype(np.float32))
    return (X_train, y_train), (X_test, y_test)


def load_cifar10() -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
    """Load CIFAR-10 dataset as ((X_train, y_train), (X_test, y_test)). Falls back to synthetic data."""
    import numpy as np

    try:
        import pickle, os, gzip
        cache_dir = os.path.expanduser("~/.cml/datasets/cifar10")
        if os.path.exists(os.path.join(cache_dir, "data_batch_1")):
            X_batches, y_batches = [], []
            for i in range(1, 6):
                with open(os.path.join(cache_dir, f"data_batch_{i}"), 'rb') as f:
                    batch = pickle.load(f, encoding='bytes')
                X_batches.append(batch[b'data'])
                y_batches.append(batch[b'labels'])
            X_train_np = np.concatenate(X_batches).astype(np.float32) / 255.0
            y_train_np = np.concatenate(y_batches).astype(np.float32)

            with open(os.path.join(cache_dir, "test_batch"), 'rb') as f:
                test = pickle.load(f, encoding='bytes')
            X_test_np = np.array(test[b'data'], dtype=np.float32) / 255.0
            y_test_np = np.array(test[b'labels'], dtype=np.float32)

            X_train = Tensor.from_numpy(X_train_np.reshape(-1, 3, 32, 32))
            y_train = Tensor.from_numpy(y_train_np)
            X_test = Tensor.from_numpy(X_test_np.reshape(-1, 3, 32, 32))
            y_test = Tensor.from_numpy(y_test_np)
            return (X_train, y_train), (X_test, y_test)
    except Exception:
        pass

    # Fallback: small synthetic data
    np.random.seed(42)
    X_train = Tensor.from_numpy(np.random.randn(1000, 3, 32, 32).astype(np.float32) * 0.1)
    y_train = Tensor.from_numpy(np.random.randint(0, 10, 1000).astype(np.float32))
    X_test = Tensor.from_numpy(np.random.randn(200, 3, 32, 32).astype(np.float32) * 0.1)
    y_test = Tensor.from_numpy(np.random.randint(0, 10, 200).astype(np.float32))
    return (X_train, y_train), (X_test, y_test)
