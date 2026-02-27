# CML Easy API Guide

Complete guide to CML's convenient, high-level APIs.

## Overview

CML provides a layered API design:

```
Easy API (Quickest)
    v
Convenience API (Common tasks)
    v
Standard API (Full control)
    v
Low-level API (Advanced)
```

This guide focuses on the Easy and Convenience APIs.

## Quick Start (30 seconds)

```python
import cml

cml.init()

# 1. Build a model (one line!)
model = cml.build_model([784, 128, 64, 10])

# 2. Create data
X = cml.randn([1000, 784])
y = cml.zeros([1000, 10])

# 3. Train (one line!)
losses = cml.train_model(model, X, y, epochs=10)

# 4. Evaluate
test_loss = cml.evaluate_model(model, X_test, y_test)

# 5. Predict
predictions = cml.predict(model, X_test)

cml.cleanup()
```

Done!

## Easy API Functions

### Model Building

#### `build_model(layer_sizes, dropout=0, activation="relu")`

Quickly build a feedforward neural network.

```python
# Simple 3-layer network
model = cml.build_model([784, 256, 128, 10])

# With dropout regularization
model = cml.build_model([784, 256, 128, 10], dropout=0.5)

# Different activation
model = cml.build_model([784, 256, 10], activation="sigmoid")
```

**Parameters:**
- `layer_sizes` (list): Network dimensions
- `dropout` (float): Dropout probability (0 = disabled)
- `activation` (str): "relu", "sigmoid", or "tanh"

**Returns:** Sequential model ready to use

---

### Training

#### `train_model(model, X, y, epochs=10, **options)`

Train a model with sensible defaults.

```python
# Basic training
losses = cml.train_model(model, X_train, y_train)

# With custom options
losses = cml.train_model(
    model,
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    learning_rate=0.01,
    loss_fn="cross_entropy",
    optimizer="sgd",
    verbose=True
)
```

**Parameters:**
- `model`: Model to train
- `X_train`: Training features
- `y_train`: Training targets
- `epochs`: Number of epochs (default: 10)
- `batch_size`: Batch size (default: full batch)
- `learning_rate`: Learning rate (default: 0.001)
- `loss_fn`: Loss function (default: "mse")
  - Options: "mse", "mae", "cross_entropy", "bce"
- `optimizer`: Optimizer (default: "adam")
  - Options: "adam", "sgd", "rmsprop", "adagrad"
- `verbose`: Print progress (default: True)

**Returns:** List of loss values per epoch

---

### Evaluation

#### `evaluate_model(model, X_test, y_test, loss_fn="mse")`

Evaluate model on test data.

```python
test_loss = cml.evaluate_model(model, X_test, y_test)
print(f"Test loss: {test_loss:.4f}")
```

---

### Inference

#### `predict(model, X)`

Make predictions on new data.

```python
predictions = cml.predict(model, X_new)
```

Sets model to inference mode automatically.

---

### Optimizer Creation

#### `create_optimizer(model, optimizer="adam", learning_rate=0.001, **kwargs)`

Create optimizer with sensible defaults.

```python
# Adam optimizer
opt = cml.create_optimizer(model, "adam", learning_rate=0.01)

# SGD with momentum
opt = cml.create_optimizer(model, "sgd", learning_rate=0.1, momentum=0.9)

# RMSprop
opt = cml.create_optimizer(model, "rmsprop", learning_rate=0.001)
```

---

### Loss Functions

#### `get_loss_function(loss_fn)`

Get a loss function by name.

```python
loss_fn = cml.get_loss_function("cross_entropy")
loss = loss_fn(output, target)
```

---

## Data Utilities

### Dataset Management

#### `create_dataset(X, y=None)`

Create a dataset wrapper.

```python
X = cml.randn([1000, 10])
y = cml.zeros([1000, 5])

dataset = cml.create_dataset(X, y)
```

#### `create_dataloader(dataset, batch_size=32, shuffle=False)`

Create a batch iterator.

```python
loader = cml.create_dataloader(dataset, batch_size=32, shuffle=True)

for X_batch, y_batch in loader:
    output = model(X_batch)
    loss = loss_fn(output, y_batch)
```

#### `train_test_split(X, y, test_size=0.2, random_state=None)`

Split data into train/test sets.

```python
X_train, X_test, y_train, y_test = cml.train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

#### `normalize(X)`

Normalize data to mean=0, std=1.

```python
X_normalized = cml.normalize(X)
```

#### `minmax_scale(X, min_val=0, max_val=1)`

Scale data to [min, max] range.

```python
X_scaled = cml.minmax_scale(X, 0, 1)
```

---

## Training Utilities

### Metrics Tracking

```python
metrics = cml.MetricsTracker()

for epoch in range(10):
    loss = train_epoch(model, data)
    metrics.log("loss", loss)
    print(metrics)  # Automatically formatted output

# Access metrics
avg_loss = metrics.average("loss")
latest_loss = metrics.latest("loss")
```

### Early Stopping

Stop training if validation loss doesn't improve.

```python
early_stop = cml.EarlyStopping(patience=5)

for epoch in range(100):
    train(model)
    val_loss = evaluate(model, val_data)

    if early_stop(val_loss):
        print("Stopping early!")
        break
```

### Learning Rate Scheduling

Automatically decay learning rate.

```python
scheduler = cml.LearningRateScheduler(
    optimizer,
    schedule="step",
    decay=0.95,
    step_size=10
)

for epoch in range(100):
    train(model)
    scheduler.step()  # Decay LR every step_size epochs
```

---

## Context Managers

### Device/Dtype Control

```python
# Use CPU with float32
with cml.TrainingContext(device="cpu", dtype="float32"):
    model = cml.build_model([10, 20, 5])
    X = cml.randn([100, 10])
    output = model(X)

# Use GPU if available
with cml.TrainingContext(device="cuda"):
    # Training on GPU
```

### Training Mode

```python
model = cml.build_model([10, 20, 5])

# Enable gradients and dropout
with cml.training_mode(model, True):
    output = model(X)
    loss = cml.mse_loss(output, y)
    cml.backward(loss)

# Disable gradients
with cml.disable_grad():
    output = model(X)  # No gradients computed
```

---

## Complete Training Pipeline

```python
import cml

cml.init()
cml.seed(42)

# 1. Load/create data
X_train = cml.randn([1000, 20])
y_train = cml.zeros([1000, 5])
X_test = cml.randn([200, 20])
y_test = cml.zeros([200, 5])

# 2. Build model
model = cml.build_model([20, 64, 32, 5], dropout=0.2)

# 3. Create utilities
optimizer = cml.create_optimizer(model, "adam", lr=0.01)
scheduler = cml.LearningRateScheduler(optimizer, decay=0.95)
metrics = cml.MetricsTracker()
early_stop = cml.EarlyStopping(patience=5)

# 4. Training loop
model.set_training(True)
for epoch in range(100):
    # Train
    optimizer.zero_grad()
    output = model(X_train)
    loss = cml.mse_loss(output, y_train)
    cml.backward(loss)
    optimizer.step()
    scheduler.step()

    # Validate
    test_loss = cml.evaluate_model(model, X_test, y_test)

    # Track metrics
    metrics.log("train_loss", 0.5)  # Simplified
    metrics.log("test_loss", test_loss)

    # Check early stopping
    if early_stop(test_loss):
        print("Early stopping!")
        break

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: {metrics}")

# 5. Final evaluation
final_loss = cml.evaluate_model(model, X_test, y_test)
print(f"\nFinal test loss: {final_loss:.4f}")

# 6. Predictions
predictions = cml.predict(model, X_test)

cml.cleanup()
```

---

## Common Patterns

### Classification

```python
# Build classifier
model = cml.build_model([784, 256, 128, 10])

# Train with cross entropy
losses = cml.train_model(
    model,
    X_train,
    y_train,
    loss_fn="cross_entropy",
    optimizer="adam"
)
```

### Regression

```python
# Build regressor
model = cml.build_model([10, 32, 16, 1])

# Train with MSE
losses = cml.train_model(
    model,
    X_train,
    y_train,
    loss_fn="mse",
    optimizer="adam"
)
```

### With Batch Processing

```python
# Create dataloader
dataset = cml.create_dataset(X, y)
loader = cml.create_dataloader(dataset, batch_size=32)

# Train on batches
model.set_training(True)
for epoch in range(10):
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = cml.mse_loss(output, y_batch)
        cml.backward(loss)
        optimizer.step()
```

### Hyperparameter Tuning

```python
best_loss = float("inf")
best_lr = None

for lr in [0.001, 0.01, 0.1]:
    model = cml.build_model([20, 32, 5])
    losses = cml.train_model(
        model,
        X,
        y,
        epochs=5,
        learning_rate=lr
    )

    final_loss = losses[-1]
    if final_loss < best_loss:
        best_loss = final_loss
        best_lr = lr

print(f"Best LR: {best_lr}")
```

---

## Advanced Features

### Custom Training Loop

For full control, use standard API:

```python
model.set_training(True)
for epoch in range(10):
    optimizer.zero_grad()
    output = model(X)
    loss = cml.mse_loss(output, y)
    cml.backward(loss)
    optimizer.step()
```

### Batch Iterator

```python
for X_batch, y_batch in cml.batch_iterator(X, y, batch_size=32):
    # Process batch
```

### Loss Function Selection

```python
loss_fn = cml.get_loss_function("cross_entropy")
loss = loss_fn(output, target)
```

---

## API Layers

### Layer 1: Quickest (Few lines)

```python
cml.init()
model = cml.build_model([10, 20, 5])
cml.train_model(model, X, y)
cml.cleanup()
```

### Layer 2: Convenience (More control)

```python
cml.init()
model = cml.build_model([10, 20, 5])
opt = cml.create_optimizer(model)
for epoch in range(10):
    opt.zero_grad()
    out = model(X)
    loss = cml.mse_loss(out, y)
    cml.backward(loss)
    opt.step()
cml.cleanup()
```

### Layer 3: Standard API (Full control)

```python
from cml import Sequential, Linear, ReLU, Adam

cml.init()
model = Sequential()
model.add(Linear(10, 20))
model.add(ReLU())
model.add(Linear(20, 5))

opt = Adam(model, lr=0.001)
# ... training loop ...
```

### Layer 4: Low-level (Direct C access)

```python
from cml._cml_lib import lib, ffi
# Direct CFFI calls for maximum control
```

---

## Error Handling

Easy API includes helpful error messages:

```python
try:
    # Automatic error checking
    model = cml.build_model([])  # Error: need >=2 layers
except ValueError as e:
    print(f"Error: {e}")

# Check CML errors
if cml.has_error():
    print(f"CML error: {cml.get_error()}")
    cml.clear_error()
```

---

## Performance Tips

1. **Batch Processing**: Use DataLoader for large datasets
2. **Device Selection**: Use GPU when available
3. **Learning Rate**: Start with 0.001 for Adam
4. **Dropout**: Use 0.2-0.5 for regularization
5. **Early Stopping**: Stop when validation loss plateaus

---

## Troubleshooting

### "Optimizer not converging"

```python
# Try:
- Reduce learning rate (0.001 -> 0.0001)
- Add dropout (dropout=0.5)
- Normalize data (cml.normalize(X))
- Increase epochs (epochs=20)
```

### "Model not training"

```python
# Ensure training mode is set
model.set_training(True)

# Check gradients are computed
cml.backward(loss)
grad = cml.get_grad(param)
assert grad is not None
```

### "Memory issues"

```python
# Reduce batch size
loader = cml.create_dataloader(dataset, batch_size=8)

# Or reduce model size
model = cml.build_model([10, 16, 5])  # Smaller hidden layer
```

---

## Summary

CML's Easy API provides:

- **One-line model building**: `cml.build_model()`
- **One-line training**: `cml.train_model()`
- **Data utilities**: Loaders, splitting, normalization
- **Training helpers**: Metrics, early stopping, scheduling
- **Device management**: Easy GPU/CPU switching
- **Full control**: Drop to standard API anytime

Perfect for beginners and rapid prototyping!

---

## Next Steps

1. **Start simple**: Use `build_model()` and `train_model()`
2. **Add features**: Try DataLoader, metrics tracking
3. **Customize**: Drop to standard API for custom training
4. **Deploy**: Export trained models for inference

See `examples/05_easy_api.py` for working examples!
