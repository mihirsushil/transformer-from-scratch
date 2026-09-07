"""Scaled dot-product & multi-head attention, built entirely from ops the
engine already has an autograd rule for -- no attention-specific backward
code anywhere in this file. That's the payoff of extending the engine with
general primitives (batched matmul, transpose, reshape, softmax) instead of
one attention-shaped op: the chain rule composes the correct gradient for
free.
"""

import numpy as np

from .engine import Tensor
from .layers import LayerNorm, Linear, Module, ReLU, Sequential


def scaled_dot_product_attention(q, k, v, causal_mask=False):
    """q, k, v: (batch, heads, seq, head_dim) -> (batch, heads, seq, head_dim).

    The three matrices answer three different questions per token: what am I
    looking for (query), what do I contain that others might want (key), and
    what do I actually pass along if picked (value). Query/key comparison
    decides *how much* attention to pay; value is *what* gets mixed in.
    """
    head_dim = q.shape[-1]

    # Scores: how well each query matches each key, for every pair of
    # positions at once via one batched matmul.
    scores = q @ k.transpose(-2, -1)

    # Without this, larger head_dim pushes dot products to larger magnitudes
    # (variance grows with head_dim for random Q/K), which pushes softmax
    # into a near-one-hot regime and starves most positions of gradient.
    scores = scores / (head_dim ** 0.5)

    if causal_mask:
        seq = q.shape[-2]
        # Position i may only attend to positions <= i -- a decoder can't
        # peek at future tokens it's supposed to predict. Adding -inf before
        # softmax (rather than zeroing after) is what makes disallowed
        # positions get exactly zero probability instead of a small nonzero
        # share of the softmax normalization.
        mask = np.triu(np.full((seq, seq), -np.inf), k=1)
        scores = scores + Tensor(mask)

    weights = scores.softmax(axis=-1)
    return weights @ v


class MultiHeadAttention(Module):
    def __init__(self, d_model, num_heads, causal=False):
        assert d_model % num_heads == 0, "d_model must divide evenly into num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.out_proj = Linear(d_model, d_model)

    def _split_heads(self, x):
        # (batch, seq, d_model) -> (batch, heads, seq, head_dim). Splitting
        # d_model into (heads, head_dim) then moving heads before seq is what
        # lets one batched matmul compute every head's attention in parallel
        # instead of looping over heads in Python.
        batch, seq, _ = x.shape
        return x.reshape(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        # Inverse of _split_heads: (batch, heads, seq, head_dim) -> (batch, seq, d_model).
        batch, heads, seq, head_dim = x.shape
        return x.transpose(1, 2).reshape(batch, seq, heads * head_dim)

    def __call__(self, x):
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        attended = scaled_dot_product_attention(q, k, v, causal_mask=self.causal)
        return self.out_proj(self._merge_heads(attended))

    def parameters(self):
        return (
            self.q_proj.parameters()
            + self.k_proj.parameters()
            + self.v_proj.parameters()
            + self.out_proj.parameters()
        )


class TransformerBlock(Module):
    """A pre-norm decoder block: attention + residual, then MLP + residual.

    Pre-normalization keeps each sublayer's input numerically well scaled and
    gives gradients a direct residual path through a deep stack. The MLP widens
    each token independently; attention mixes information *between* tokens.
    """

    def __init__(self, d_model, num_heads, mlp_ratio=4, causal=True):
        assert mlp_ratio >= 1, "mlp_ratio must be positive"
        self.norm1 = LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads, causal=causal)
        self.norm2 = LayerNorm(d_model)
        self.mlp = Sequential(
            Linear(d_model, d_model * mlp_ratio),
            ReLU(),
            Linear(d_model * mlp_ratio, d_model),
        )

    def __call__(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))

    def parameters(self):
        return (
            self.norm1.parameters()
            + self.attention.parameters()
            + self.norm2.parameters()
            + self.mlp.parameters()
        )
