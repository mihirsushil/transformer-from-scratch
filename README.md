# Transformer from Scratch

A decoder-only transformer built on a hand-written NumPy reverse-mode
autograd engine. The project extends an earlier MLP engine with the operations
attention requires—batched matrix multiplication, axis-aware transpose,
reshape, division, and numerically stable softmax—then verifies each new
gradient against central-difference numerical differentiation.

## Built so far

- A NumPy tensor engine with reverse-mode automatic differentiation.
- Learned token and positional embeddings (`Embedding` in `nn/layers.py`, using unbuffered `np.add.at` in the backward pass so repeated token IDs within a sequence accumulate gradients accurately).
- Multi-head scaled dot-product attention with optional causal masking.
- A pre-norm causal transformer block: LayerNorm, attention/residual, and widened ReLU MLP/residual paths.
- A full GPT-style `DecoderTransformer` (`nn/attention.py`) stacking token + positional embeddings, multiple causal `TransformerBlock`s, final `LayerNorm`, a vocabulary projection head (`lm_head`), and autoregressive `.generate()`.
- Sequence next-token cross-entropy loss (`sequence_cross_entropy_loss`) and an end-to-end character-level training script (`train_lm.py`) that reduces loss from `2.91` to `0.11` (`95.6%` next-character accuracy) and generates coherent completions (see [`results/lm_training_results.md`](results/lm_training_results.md)).
- Gradient checks (`28/28` tests passing) for every engine operation, `Embedding` (with duplicate token indices), `LayerNorm`, `MultiHeadAttention`, `TransformerBlock`, and the full `DecoderTransformer` through sequence cross-entropy loss, plus behavioral causal-mask and sequence-memorization tests.

## Run locally

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python train_lm.py
```

The first command creates an isolated Python environment; the second installs
the exact test dependencies; the third runs the 28 gradient and behavior tests;
the fourth trains the 2-layer causal decoder and writes `results/lm_training_results.md`.
