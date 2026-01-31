# micrograd-vibecode

A tiny scalar Autograd engine (vibecode edition).

## Features
- `Value` class for scalar operations.
- Supports `+`, `-`, `*`, `/`, `**`.
- Implements `relu` activation.
- **Autograd Engine**: Supports backpropagation and gradient computation (`backward()`).
- **Neural Network Module**: `nn.py` containing `Neuron`, `Layer`, and `MLP`.
- Implements basic arithmetic dunders for intuitive usage.



## Installation
Requires Python 3.9+.

```bash
uv sync
```

## Usage

```python
from engine import Value

a = Value(2.0)
b = Value(3.0)
c = a + b
print(c) # Value(data=5.0)
```

### Neural Network Examples

#### 1. Single Neuron
```python
from nn import Neuron
from engine import Value

x = [Value(1.0), Value(-2.0)]
n = Neuron(2) # 2 inputs
out = n(x)
print(f"Neuron output: {out}")
```

#### 2. Layer of Neurons
```python
from nn import Layer
from engine import Value

x = [Value(1.0), Value(2.0)]
l = Layer(2, 3) # 2 inputs, 3 neurons
out = l(x)
print(f"Layer output: {out}") # List of 3 Values
```

#### 3. Multi-Layer Perceptron (MLP)
```python
from nn import MLP
from engine import Value

x = [Value(2.0), Value(3.0), Value(-1.0)]
# 3 inputs, two layers of 4 neurons, one output
n = MLP(3, [4, 4, 1])
out = n(x)
out.backward()
print(f"MLP Output: {out.data}")
```


Run the demo:
```bash
uv run main.py
```

## Testing
Run the unit tests:
```bash
uv run pytest
```

## Development History
See [prompts.md](prompts.md) for the list of prompts used to generate this project.

