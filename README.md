# micrograd-vibecode

A tiny scalar Autograd engine (vibecode edition).

## Features
- `Value` class for scalar operations.
- Supports `+`, `-`, `*`, `/`, `**`.
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

