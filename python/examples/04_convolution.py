#!/usr/bin/env python3
"""
Convolutional Neural Network Example

This example demonstrates:
- Building a CNN with convolutional and pooling layers
- Typical image processing architecture
- Training a CNN model
"""

import cml


def main():
    # Initialize
    cml.init()
    cml.seed(42)

    print("=== Convolutional Neural Network Example ===\n")

    # Parameters
    batch_size = 16
    num_classes = 10
    epochs = 10

    # Create synthetic image data
    print("Creating synthetic image data...")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: 32x32x3 (height x width x channels)")
    print(f"  Number of classes: {num_classes}\n")

    # Input: batch_size x 3 x 32 x 32 (BCHW format)
    X_train = cml.randn([batch_size, 3, 32, 32])
    y_train = cml.zeros([batch_size, num_classes])

    # Build CNN model
    print("Building CNN model...")
    model = cml.Sequential()

    # First convolutional block
    model.add(
        cml.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
    )
    model.add(cml.ReLU())
    model.add(cml.MaxPool2d(kernel_size=2, stride=2))

    # Second convolutional block
    model.add(
        cml.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
    )
    model.add(cml.ReLU())
    model.add(cml.MaxPool2d(kernel_size=2, stride=2))

    # Classification head
    # After 2 max pooling layers: 32x32 -> 16x16 -> 8x8
    # 8x8x64 = 4096 features
    model.add(cml.Linear(64 * 8 * 8, 128))  # Flatten + FC layer
    model.add(cml.ReLU())
    model.add(cml.Dropout(p=0.5))
    model.add(cml.Linear(128, num_classes))

    print(f"CNN model built with {len(model)} layers")

    # Create optimizer
    optimizer = cml.Adam(model, lr=0.001)

    # Training
    print(f"\nTraining for {epochs} epochs...\n")
    model.set_training(True)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass through CNN
        features = model(X_train)

        # Compute loss
        loss = cml.cross_entropy_loss(features, y_train)

        # Backward pass
        cml.backward(loss)

        # Update parameters
        optimizer.step()

        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1:2d}/{epochs}: CNN training progress...")

    print("\nTraining complete!")

    # Inference
    print("\nRunning inference...")
    model.set_training(False)  # Disable dropout

    test_input = cml.randn([batch_size, 3, 32, 32])
    predictions = model(test_input)

    print(f"Inference complete!")
    print(f"Output predictions shape: {predictions.size} elements")

    # Cleanup
    cml.cleanup()
    print("\nDone!")


if __name__ == "__main__":
    main()
