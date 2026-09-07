# Transformer from Scratch

A decoder-only transformer built on a hand-written NumPy reverse-mode
autograd engine. The project extends an earlier MLP engine with the operations
attention requires—batched matrix multiplication, axis-aware transpose,
reshape, division, and numerically stable softmax—then verifies each new
gradient against central-difference numerical differentiation.

## Built so far

- A NumPy tensor engine with reverse-mode automatic differentiation.
- Multi-head scaled dot-product attention with optional causal masking.
- A pre-norm causal transformer block: LayerNorm, attention/residual, and
  widened ReLU MLP/residual paths.
- Gradient checks for every new engine operation, LayerNorm, attention, and
  the full transformer block, plus behavioral causal-mask tests.

## Run locally

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

The first command creates an isolated Python environment; the second installs
the exact test dependencies; the third runs the gradient and behavior tests.

## Next steps

1. Add token and positional embeddings.
2. Stack transformer blocks and add a vocabulary output head.
3. Train the small decoder model on a compact next-token prediction task.
4. Measure training loss and generate samples to validate learned behavior.
