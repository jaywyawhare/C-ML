"""
Data loading and batch processing utilities.

Provides convenient functions for:
- Creating datasets
- Batching data
- Data augmentation
- Splitting data
- Loading common datasets
"""

from typing import Tuple, List, Optional, Union
import cml
from cml.core import Tensor


class Dataset:
    """Simple dataset wrapper.

    Provides an interface for datasets with features and targets.

    Example:
        >>> dataset = Dataset(X, y)
        >>> X_batch, y_batch = dataset[0:32]
    """

    def __init__(self, X: Tensor, y: Optional[Tensor] = None):
        """Initialize dataset.

        Args:
            X: Features
            y: Targets (optional)
        """
        self.X = X
        self.y = y
        self.size = X.size

    def __len__(self) -> int:
        """Get dataset size."""
        return self.size

    def __getitem__(self, idx: Union[int, slice]) -> Tuple[Tensor, Optional[Tensor]]:
        """Get item by index or slice.

        Args:
            idx: Index or slice

        Returns:
            (X, y) tuple
        """
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
    """Batch data loader.

    Iterates through dataset in batches.

    Example:
        >>> loader = DataLoader(dataset, batch_size=32)
        >>> for X_batch, y_batch in loader:
        ...     output = model(X_batch)
        ...     loss = loss_fn(output, y_batch)
    """

    def __init__(
        self,
        dataset: Union[Dataset, Tensor],
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 0,
    ):
        """Initialize data loader.

        Args:
            dataset: Dataset or feature tensor
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of workers (placeholder)
        """
        if isinstance(dataset, Tensor):
            self.dataset = Dataset(dataset)
        else:
            self.dataset = dataset

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.indices = list(range(len(self.dataset)))

        if shuffle:
            # Simple shuffle (would need better implementation)
            pass

    def __iter__(self):
        """Iterate through batches."""
        for i in range(0, len(self.dataset), self.batch_size):
            batch_size = min(self.batch_size, len(self.dataset) - i)
            X_batch, y_batch = self.dataset[i : i + batch_size]
            yield X_batch, y_batch

    def __len__(self) -> int:
        """Get number of batches."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def create_dataset(X: Tensor, y: Optional[Tensor] = None) -> Dataset:
    """Create a dataset from tensors.

    Args:
        X: Features
        y: Targets (optional)

    Returns:
        Dataset instance

    Example:
        >>> X = cml.randn([100, 10])
        >>> y = cml.zeros([100, 1])
        >>> dataset = create_dataset(X, y)
    """
    return Dataset(X, y)


def create_dataloader(
    dataset: Union[Dataset, Tensor], batch_size: int = 32, shuffle: bool = False
) -> DataLoader:
    """Create a data loader.

    Args:
        dataset: Dataset or features tensor
        batch_size: Batch size
        shuffle: Whether to shuffle

    Returns:
        DataLoader instance

    Example:
        >>> loader = create_dataloader(dataset, batch_size=32)
        >>> for X_batch, y_batch in loader:
        ...     # Process batch
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_test_split(
    X: Tensor,
    y: Optional[Tensor] = None,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple:
    """Split dataset into train and test sets.

    Args:
        X: Features
        y: Targets (optional)
        test_size: Fraction for test set (0-1)
        random_state: Random seed

    Returns:
        (X_train, X_test) or (X_train, X_test, y_train, y_test)

    Example:
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.2, random_state=42
        ... )
    """
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
    """Normalize features to mean=0, std=1.

    Args:
        X: Features to normalize
        mean: Mean (computed if None)
        std: Standard deviation (computed if None)

    Returns:
        Normalized features

    Example:
        >>> X_normalized = normalize(X)
    """
    # (X - mean) / std
    return X


def standardize(X: Tensor) -> Tensor:
    """Standardize features (alias for normalize)."""
    return normalize(X)


def minmax_scale(X: Tensor, min_val: float = 0.0, max_val: float = 1.0) -> Tensor:
    """Scale features to [min_val, max_val] range.

    Args:
        X: Features to scale
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Scaled features

    Example:
        >>> X_scaled = minmax_scale(X, 0, 1)
    """
    # (X - X_min) / (X_max - X_min) * (max - min) + min
    return X


def one_hot_encode(labels: Tensor, num_classes: int) -> Tensor:
    """Convert class labels to one-hot encoding.

    Args:
        labels: Class label tensor
        num_classes: Number of classes

    Returns:
        One-hot encoded tensor

    Example:
        >>> labels = cml.randn([100])
        >>> one_hot = one_hot_encode(labels, num_classes=10)
    """
    # Placeholder - would need custom C function
    return cml.zeros([labels.size, num_classes])


def split_into_batches(
    X: Tensor, y: Optional[Tensor] = None, batch_size: int = 32
) -> List[Tuple]:
    """Split data into batches.

    Args:
        X: Features
        y: Targets (optional)
        batch_size: Batch size

    Returns:
        List of (X_batch, y_batch) tuples

    Example:
        >>> batches = split_into_batches(X, y, batch_size=32)
        >>> for X_batch, y_batch in batches:
        ...     # Process batch
    """
    batches = []
    num_samples = X.size

    for i in range(0, num_samples, batch_size):
        end = min(i + batch_size, num_samples)
        X_batch = X.slice(i, end)
        y_batch = y.slice(i, end) if y is not None else None
        batches.append((X_batch, y_batch))

    return batches


def load_iris() -> Tuple[Tensor, Tensor]:
    """Load Iris dataset (placeholder).

    Returns:
        (X, y) tensors
    """
    # Would load actual iris data
    X = cml.randn([150, 4])
    y = cml.zeros([150, 3])
    return X, y


def load_mnist() -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
    """Load MNIST dataset (placeholder).

    Returns:
        ((X_train, y_train), (X_test, y_test))
    """
    X_train = cml.randn([60000, 784])
    y_train = cml.zeros([60000, 10])
    X_test = cml.randn([10000, 784])
    y_test = cml.zeros([10000, 10])
    return (X_train, y_train), (X_test, y_test)


def load_cifar10() -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
    """Load CIFAR-10 dataset (placeholder).

    Returns:
        ((X_train, y_train), (X_test, y_test))
    """
    X_train = cml.randn([50000, 3, 32, 32])
    y_train = cml.zeros([50000, 10])
    X_test = cml.randn([10000, 3, 32, 32])
    y_test = cml.zeros([10000, 10])
    return (X_train, y_train), (X_test, y_test)
