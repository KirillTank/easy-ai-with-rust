# easy-ai-with-rust

Minimalist implementation of a linear adaptive neural network written in pure Rust, without external ML libraries — relying solely on the standard library (`std`).

## What's Inside

- **`DynamicNet`** — A lightweight adaptive network supporting a customizable number of inputs and weight scaling per input (`weights_per_input`). Prediction is calculated as a weighted sum of input signals, where each input's effective weight is composed of a group of multiple trained parameters.
- **Adaptive NLMS Training** — Training is powered by the **Normalized Least Mean Squares (NLMS)** algorithm. Unlike standard gradient descent, NLMS automatically normalizes the learning rate at each step using the energy of the input signal ($\sum x^2$). This avoids manual hyperparameter tuning for learning rates and provides quick, stable convergence.

## Features & API

- `DynamicNet::new(inputs_count, weights_per_input)` — Initializes a network with a custom input count and number of weights per group.
- `predict(&self, inputs)` — Returns the prediction for a given input vector.
- `train(&mut self, inputs, target)` — Executes a single training step using the NLMS algorithm.
- `start_train(&mut self, max_epoch, dataset)` — Runs a full training cycle over a dataset with automatic stopping upon reaching target accuracy.
- `print_res_train(&self, dataset)` — Prints final predictions alongside loss metrics for each example in the dataset.

## Usage Example

```rust
fn main() {
    let mut net = DynamicNet::new(5, 2);

    let dataset = vec![
        (vec![1.0, 2.0, 3.0, 0.5, 1.5], 100.0),
        (vec![2.0, 0.5, 1.0, 4.0, 0.0], 50.0),
        // ...
    ];

    net.start_train(1000000, &dataset);
    net.print_res_train(&dataset);
}
```
---
# Licence MIT
