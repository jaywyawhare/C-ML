#!/usr/bin/env python3
"""
Hello CML - First Python program with CML

This example shows the simplest way to use CML from Python.
"""

import cml

# Initialize CML
cml.init()

print("=== Hello CML ===\n")

# Create some tensors
print("Creating tensors...")
x = cml.zeros([2, 3])
y = cml.ones([2, 3])

print(f"x size: {x.size}")
print(f"y size: {y.size}")

# Simple operations
print("\nTensor operations...")
z = x + y
result = z.sum()

print("Operations completed!")

# Cleanup
cml.cleanup()
print("\nDone!")
