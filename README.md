# Perceptron-Logic-Jax
All Gates Perceptron with JAX – A minimal Python implementation of perceptrons to simulate basic logic gates (AND, OR, NAND, XOR, etc.) using JAX for fast, GPU-accelerated computations. This repo demonstrates how simple neural networks can learn logical operations with concise, high-performance code.

## Features
* Perceptron implementations for common logic gates
* Simple, readable code for learning and experimentation
* Step-by-step training to observe how weights and biases evolve

## Concept Covered 
* Perceptron Learning Rule with gradient descent
* Sigmoid activation function and its derivative:
```math
  \sigma(x) = \frac{1}{1 + e^{-x}}
```
```math
  \sigma'(x) = \sigma(x)\,(1 - \sigma(x))
```
* High learning rate (10) for faster convergence on small datasets
* Training with JAX for efficient computation

The use of sigmoid + high learning rate provides:
- Smooth probability-like outputs (e.g., 0.99 instead of hard 1)
- Better gradient flow for weight updates
- Faster convergence compared to a step function
