#!/usr/bin/env python3
"""
Classification Example - Training a classifier with CML

This example shows how to:
- Build a classification network
- Use cross entropy loss
- Train for multi-class classification
"""

import cml


def main():
    # Initialize
    cml.init()
    cml.seed(42)

    print("Classification Example\n")

    # Hyperparameters
    num_samples = 64
    input_dim = 20
    num_classes = 5
    epochs = 20
    learning_rate = 0.01

    # Create synthetic data
    print(f"Creating synthetic classification data...")
    print(f"  Samples: {num_samples}")
    print(f"  Input dimensions: {input_dim}")
    print(f"  Number of classes: {num_classes}\n")

    X_train = cml.randn([num_samples, input_dim])
    y_train = cml.zeros([num_samples, num_classes])  # One-hot encoded (simplified)

    # Build classifier model
    print("Building classification model...")
    model = cml.Sequential()
    model.add(cml.Linear(input_dim, 64))
    model.add(cml.ReLU())
    model.add(cml.Dropout(p=0.2))
    model.add(cml.Linear(64, 32))
    model.add(cml.ReLU())
    model.add(cml.Linear(32, num_classes))

    print(f"Model built with {len(model)} layers")

    # Create optimizer
    optimizer = cml.SGD(model, lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    # Training
    print(f"\nTraining for {epochs} epochs...\n")
    model.set_training(True)

    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = cml.cross_entropy_loss(logits, y_train)
        cml.backward(loss)
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d}/{epochs}: Loss computed")

    print("\nTraining complete!")

    # Evaluation
    print("\nEvaluating model...")
    model.set_training(False)

    eval_logits = model(X_train)
    eval_loss = cml.cross_entropy_loss(eval_logits, y_train)

    print("Evaluation complete!")

    # Cleanup
    cml.cleanup()
    print("\nDone!")


if __name__ == "__main__":
    main()
