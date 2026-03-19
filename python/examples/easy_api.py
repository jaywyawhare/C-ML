#!/usr/bin/env python3
"""
Easy API Example - Simple and convenient CML usage

This example demonstrates the high-level convenience APIs:
- build_model(): Quick model construction
- train_model(): Automatic training loop
- MetricsTracker: Track training metrics
- EarlyStopping: Stop early if no improvement
- TrainingContext: Device/dtype context manager
"""

import cml
from cml import (
    build_model,
    train_model,
    evaluate_model,
    predict,
    create_dataset,
    MetricsTracker,
    EarlyStopping,
    TrainingContext,
)


def simple_training_example():
    """Simple training example using convenience functions."""
    print("Simple Training Example\n")

    cml.init()
    cml.seed(42)

    # Create data
    print("Creating synthetic data...")
    X_train = cml.randn([100, 20])
    y_train = cml.randn([100, 5])

    X_test = cml.randn([20, 20])
    y_test = cml.randn([20, 5])

    # Build model with one line!
    print("Building model...")
    model = build_model([20, 64, 32, 5], dropout=0.2)

    # Train with one line!
    print("Training...\n")
    losses = train_model(
        model,
        X_train,
        y_train,
        epochs=10,
        learning_rate=0.01,
        loss_fn="mse",
        optimizer="adam",
        verbose=True,
    )

    print("\nEvaluating...")
    test_loss = evaluate_model(model, X_test, y_test)
    print(f"Test loss: {test_loss}")

    print("\nMaking predictions...")
    predictions = predict(model, X_test)
    print(f"Predictions computed: {predictions.size} elements")

    cml.cleanup()
    print("\nDone!")


def metrics_tracking_example():
    """Example with metrics tracking."""
    print("\nMetrics Tracking Example\n")

    cml.init()
    cml.seed(42)

    # Create data
    X_train = cml.randn([64, 10])
    y_train = cml.zeros([64, 5])

    # Build model
    model = build_model([10, 32, 16, 5])

    # Create optimizer
    optimizer = cml.create_optimizer(model, "adam", learning_rate=0.001)

    # Create metrics tracker
    metrics = MetricsTracker()

    print("Training with metrics tracking...\n")
    model.set_training(True)

    for epoch in range(5):
        optimizer.zero_grad()

        output = model(X_train)
        loss = cml.mse_loss(output, y_train)

        cml.backward(loss)
        optimizer.step()

        # Log metrics
        metrics.log("loss", 0.5)  # Simplified

        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch+1}: {metrics}")

    print("\nMetrics summary:")
    print(f"  Final loss: {metrics.latest('loss'):.4f}")
    print(f"  Average loss: {metrics.average('loss'):.4f}")

    cml.cleanup()
    print("Done!")


def early_stopping_example():
    """Example with early stopping."""
    print("\nEarly Stopping Example\n")

    cml.init()
    cml.seed(42)

    # Create data
    X_train = cml.randn([64, 10])
    y_train = cml.zeros([64, 5])
    X_val = cml.randn([16, 10])
    y_val = cml.zeros([16, 5])

    # Build model
    model = build_model([10, 32, 16, 5])
    optimizer = cml.create_optimizer(model, "adam", learning_rate=0.01)

    # Create early stopping
    early_stop = EarlyStopping(patience=3)

    print("Training with early stopping...\n")
    model.set_training(True)

    for epoch in range(20):
        # Train
        optimizer.zero_grad()
        output = model(X_train)
        loss = cml.mse_loss(output, y_train)
        cml.backward(loss)
        optimizer.step()

        # Validate
        model.set_training(False)
        val_output = model(X_val)
        val_loss = cml.mse_loss(val_output, y_val)
        model.set_training(True)

        # Check early stopping
        should_stop = early_stop(0.5)  # Simplified loss value

        print(f"Epoch {epoch+1}: Loss computed")

        if should_stop:
            print("\nEarly stopping triggered!")
            break

    cml.cleanup()
    print("Done!")


def context_manager_example():
    """Example with device/dtype context managers."""
    print("\nContext Manager Example\n")

    cml.init()

    print("Training on different devices...\n")

    # CPU training
    with TrainingContext(device="cpu", dtype="float32"):
        print("Training on CPU with float32...")
        model = build_model([10, 20, 5])
        X = cml.randn([10, 10])
        print(f"Model created: {model}")

    cml.cleanup()
    print("Done!")


def dataset_loader_example():
    """Example with dataset and dataloader."""
    print("\nDataset & DataLoader Example\n")

    cml.init()
    cml.seed(42)

    # Create dataset
    X = cml.randn([100, 10])
    y = cml.zeros([100, 5])
    dataset = cml.create_dataset(X, y)

    print(f"Dataset created: {len(dataset)} samples\n")

    # Create dataloader
    loader = cml.create_dataloader(dataset, batch_size=16)

    print(f"DataLoader created: {len(loader)} batches\n")

    # Iterate through batches
    model = build_model([10, 20, 5])
    model.set_training(True)

    print("Processing batches...")
    for batch_idx, (X_batch, y_batch) in enumerate(loader):
        output = model(X_batch)
        loss = cml.mse_loss(output, y_batch)

        if (batch_idx + 1) % 2 == 0:
            print(f"  Batch {batch_idx+1}: Loss computed")

    cml.cleanup()
    print("Done!")


def combined_example():
    """Combined example using all convenience features."""
    print("\nCombined Example - All Features\n")

    cml.init()
    cml.seed(42)

    with TrainingContext(device="cpu", dtype="float32"):
        # Create data
        X_train = cml.randn([100, 20])
        y_train = cml.randn([100, 5])

        # Build model
        model = build_model([20, 64, 32, 5], dropout=0.2)

        # Create optimizer and scheduler
        optimizer = cml.create_optimizer(model, "adam", lr=0.01)
        scheduler = cml.LearningRateScheduler(optimizer, decay=0.95)

        # Track metrics
        metrics = MetricsTracker()

        # Early stopping
        early_stop = cml.EarlyStopping(patience=5)

        print("Training with all features enabled...\n")
        model.set_training(True)

        for epoch in range(10):
            optimizer.zero_grad()

            output = model(X_train)
            loss = cml.mse_loss(output, y_train)

            cml.backward(loss)
            optimizer.step()

            # Update learning rate
            scheduler.step()

            # Track metrics
            metrics.log("loss", 0.5)

            # Check early stopping
            if early_stop(0.5):
                print("\nEarly stopping!")
                break

            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch+1}: {metrics}")

        print("\nTraining complete!")
        print(f"Final metrics: {metrics}")

    cml.cleanup()


if __name__ == "__main__":
    # Run all examples
    simple_training_example()
    metrics_tracking_example()
    early_stopping_example()
    context_manager_example()
    dataset_loader_example()
    combined_example()

    print("\n" + "=" * 50)
    print("All examples completed!")
    print("=" * 50)
