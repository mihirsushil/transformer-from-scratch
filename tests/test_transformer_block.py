import numpy as np

from nn.attention import TransformerBlock
from nn.engine import Tensor
from nn.layers import LayerNorm
from tests.test_engine import numerical_gradient


def test_layer_norm_normalizes_each_token_and_learns_affine_transform():
    x = Tensor(np.array([[[1.0, 3.0, 5.0], [2.0, 4.0, 8.0]]]))
    norm = LayerNorm(3)
    norm.gamma.data[:] = [2.0, 3.0, 4.0]
    norm.beta.data[:] = [-1.0, 0.5, 2.0]

    out = norm(x).data
    standardized = (out - norm.beta.data) / norm.gamma.data

    assert np.allclose(standardized.mean(axis=-1), 0.0, atol=1e-7)
    assert np.allclose(standardized.var(axis=-1), 1.0, atol=2e-5)


def test_layer_norm_gradient_check():
    np.random.seed(30)
    x_data = np.random.randn(2, 3, 4)
    upstream = np.random.randn(2, 3, 4)
    norm = LayerNorm(4)

    def forward_np(x):
        centered = x - x.mean(axis=-1, keepdims=True)
        variance = (centered**2).mean(axis=-1, keepdims=True)
        return ((centered / np.sqrt(variance + norm.eps)) * norm.gamma.data + norm.beta.data)

    numeric = numerical_gradient(lambda x: (forward_np(x) * upstream).sum(), x_data.copy())
    x = Tensor(x_data, requires_grad=True)
    (norm(x) * Tensor(upstream)).sum().backward()

    assert np.allclose(x.grad, numeric, atol=1e-5)


def test_transformer_block_output_shape_and_parameters():
    np.random.seed(31)
    block = TransformerBlock(d_model=8, num_heads=2)
    out = block(Tensor(np.random.randn(2, 5, 8)))

    assert out.shape == (2, 5, 8)
    # Two LayerNorm scale/shift pairs + four attention projections + two MLP linears.
    assert len(block.parameters()) == 16


def test_transformer_block_gradient_check():
    """Checks the composed attention, LayerNorm, MLP, and residual gradients."""
    np.random.seed(32)
    block = TransformerBlock(d_model=4, num_heads=2, mlp_ratio=2, causal=False)
    x_data = np.random.randn(1, 2, 4)
    upstream = np.random.randn(1, 2, 4)

    def layer_norm_np(x, norm):
        centered = x - x.mean(axis=-1, keepdims=True)
        variance = (centered**2).mean(axis=-1, keepdims=True)
        return centered / np.sqrt(variance + norm.eps) * norm.gamma.data + norm.beta.data

    def linear_np(layer, x):
        return x @ layer.weight.data + layer.bias.data

    def attention_np(x):
        batch, seq, _ = x.shape
        heads, head_dim = block.attention.num_heads, block.attention.head_dim

        def split(y):
            return y.reshape(batch, seq, heads, head_dim).transpose(0, 2, 1, 3)

        q = split(linear_np(block.attention.q_proj, x))
        k = split(linear_np(block.attention.k_proj, x))
        v = split(linear_np(block.attention.v_proj, x))
        scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(head_dim)
        exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attended = exp / exp.sum(axis=-1, keepdims=True) @ v
        merged = attended.transpose(0, 2, 1, 3).reshape(batch, seq, -1)
        return linear_np(block.attention.out_proj, merged)

    def forward_np(x):
        after_attention = x + attention_np(layer_norm_np(x, block.norm1))
        hidden = np.maximum(0.0, linear_np(block.mlp.layers[0], layer_norm_np(after_attention, block.norm2)))
        return after_attention + linear_np(block.mlp.layers[2], hidden)

    numeric = numerical_gradient(lambda x: (forward_np(x) * upstream).sum(), x_data.copy())
    x = Tensor(x_data, requires_grad=True)
    (block(x) * Tensor(upstream)).sum().backward()

    assert np.allclose(x.grad, numeric, atol=1e-4)


def test_causal_transformer_block_blocks_future_token_changes():
    np.random.seed(33)
    block = TransformerBlock(d_model=8, num_heads=2, causal=True)
    x_data = np.random.randn(1, 4, 8)
    out_a = block(Tensor(x_data)).data

    changed = x_data.copy()
    changed[:, -1, :] = np.random.randn(8)
    out_b = block(Tensor(changed)).data

    assert np.allclose(out_a[:, :-1, :], out_b[:, :-1, :], atol=1e-10)
    assert not np.allclose(out_a[:, -1, :], out_b[:, -1, :])
