# Tiny_NN_Engine — brief README

This repository contains a minimal scalar automatic differentiation engine implemented in Python inside the notebook `NN_ENGINE.ipynb`.

Short summary
- Item: core class representing a scalar value with a gradient, an operation label, and links to child nodes.
- Supported operations: addition, subtraction, multiplication, true division (via power), power, negation, and tanh activation.
- Autograd: each operation registers a local `_backward` closure. Calling `.backward()` on a root Item builds a topological ordering and runs backprop to compute gradients.
- Visualization: `trace` and `draw_dot` functions use graphviz to render the computation graph.
- Example usage: the notebook demonstrates creating values (a, b, d), composing operations (c = a + b, e = c * d, f = tanh(e)), visualizing the graph, and running backward to get gradients.
- Comparison: the notebook includes a short PyTorch snippet that computes the same example and prints gradients for reference.

How to run
- Open `NN_ENGINE.ipynb` in Jupyter or JupyterLab and run the cells in order.
- Requirements: Python 3, numpy, matplotlib, graphviz (and python-graphviz), optionally torch to run the comparison.

Location
- Main file: `NN_ENGINE.ipynb` at the repository root.

License / Notes
- Educational demo of how an autograd engine works for scalar values. Not optimized for performance or multi-dimensional tensors.
