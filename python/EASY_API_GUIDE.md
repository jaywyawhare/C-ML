# CML Easy API

High-level convenience functions for rapid prototyping.

```python
import cml
from cml.simple_api import build_model, train_model, evaluate_model, predict

cml.init()

model = build_model([784, 128, 64, 10], dropout=0.2, activation="relu")
losses = train_model(model, X_train, y_train, epochs=20, loss_fn="cross_entropy", optimizer="adam")
test_loss = evaluate_model(model, X_test, y_test)
predictions = predict(model, X_new)

cml.cleanup()
```

## Functions

### `build_model(layer_sizes, dropout=0, activation="relu")`

Build a feedforward network. Returns a `Sequential` model.

### `train_model(model, X, y, epochs=10, **opts)`

Train with sensible defaults. Options: `batch_size`, `learning_rate`, `loss_fn` ("mse", "mae", "cross_entropy", "bce"), `optimizer` ("adam", "sgd", "rmsprop", "adamw", "nadam", "lamb", "lars"), `verbose`.

### `evaluate_model(model, X_test, y_test, loss_fn="mse")`

Returns scalar loss on test data. Sets model to eval mode automatically.

### `predict(model, X)`

Returns output tensor. Sets model to eval mode automatically.

## Training Utilities

```python
from cml.functional import EarlyStopping, MetricsTracker
from cml.data import create_dataloader, train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
loader = create_dataloader(dataset, batch_size=32, shuffle=True)

metrics = MetricsTracker()
early_stop = EarlyStopping(patience=5)

for epoch in range(100):
    for X_batch, y_batch in loader:
        # ... training step ...
        pass
    metrics.log("loss", loss_val)
    if early_stop(val_loss):
        break
```

## When to Drop to Standard API

Use `cml.nn`, `cml.optim`, `cml.losses` directly when you need custom architectures, LR schedulers, mixed precision, serialization, or distributed training.

```python
import cml.nn as nn
import cml.optim as optim

model = nn.Sequential()
model.add(nn.Linear(10, 64))
model.add(nn.ReLU())
model.add(nn.LSTM(64, 32))
```
