#!/usr/bin/env python3
"""
Neural Network Training - Build and train a simple neural network in Python

This example demonstrates:
- Building a neural network with Sequential
- Creating an optimizer
- Training loop with gradient descent
- Loss computation
"""

import cml


def main():
    # Initialize CML
    cml.init()
    cml.seed(42)  # Reproducible results

    print("=== Neural Network Training Example ===\n")

    # Create sample data
    print("Creating sample data...")
    X = cml.randn([32, 10])  # 32 samples, 10 features
    y = cml.randn([32, 1])  # 32 targets

    # Build model
    print("Building model...")
    model = cml.Sequential()
    model.add(cml.Linear(10, 20))  # Input: 10 → Hidden: 20
    model.add(cml.ReLU())  # ReLU activation
    model.add(cml.Linear(20, 10))  # Hidden: 20 → 10
    model.add(cml.ReLU())  # ReLU activation
    model.add(cml.Linear(10, 1))  # Output: 1

    print(f"Model has {len(model)} layers")

    # Create optimizer
    print("Creating optimizer...")
    optimizer = cml.Adam(model, lr=0.001)

    # Set training mode
    model.set_training(True)

    # Training loop
    print("\nTraining for 10 epochs...\n")
    for epoch in range(10):
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(X)

        # Compute loss
        loss = cml.mse_loss(output, y)

        # Backward pass (compute gradients)
        cml.backward(loss)

        # Update parameters
        optimizer.step()

        # Print progress
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1:2d}: Loss computed")

    print("\nTraining complete!")

    # Switch to inference mode
    model.set_training(False)

    # Make predictions
    print("\nMaking predictions...")
    predictions = model(X)

    print(f"Predictions computed, shape: unknown (internal representation)")
    print(f"Number of predictions: {predictions.size}")

    # Cleanup
    cml.cleanup()
    print("\nDone!")


if __name__ == "__main__":
    main()
