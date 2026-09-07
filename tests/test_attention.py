import numpy as np

from nn.attention import MultiHeadAttention
from nn.engine import Tensor
from tests.test_engine import numerical_gradient


def test_output_shape():
    np.random.seed(20)
    batch, seq, d_model, heads = 2, 5, 8, 2
    mha = MultiHeadAttention(d_model, heads)

    x = Tensor(np.random.randn(batch, seq, d_model), requires_grad=True)
    out = mha(x)

    assert out.shape == (batch, seq, d_model)


def test_gradient_check():
    """Full grad check through split-heads -> attention -> merge-heads ->
    output projection, using only the input's own gradient (the composition
    is what's under test, not any single op in isolation again)."""
    np.random.seed(21)
    batch, seq, d_model, heads = 1, 3, 4, 2
    mha = MultiHeadAttention(d_model, heads)
    x_data = np.random.randn(batch, seq, d_model)

    def forward_np(x):
        # Mirror the module's forward pass in plain numpy for the numerical
        # gradient reference, reusing the module's own trained weights.
        def project(w, b, inp):
            return inp @ w + b

        def split(x):
            b, s, _ = x.shape
            return x.reshape(b, s, heads, d_model // heads).transpose(0, 2, 1, 3)

        q = split(project(mha.q_proj.weight.data, mha.q_proj.bias.data, x))
        k = split(project(mha.k_proj.weight.data, mha.k_proj.bias.data, x))
        v = split(project(mha.v_proj.weight.data, mha.v_proj.bias.data, x))

        scores = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(d_model // heads)
        m = scores.max(axis=-1, keepdims=True)
        exp = np.exp(scores - m)
        weights = exp / exp.sum(axis=-1, keepdims=True)
        attended = weights @ v

        merged = attended.transpose(0, 2, 1, 3).reshape(batch, seq, d_model)
        return project(mha.out_proj.weight.data, mha.out_proj.bias.data, merged)

    numeric = numerical_gradient(lambda x: forward_np(x).sum(), x_data.copy())

    x = Tensor(x_data, requires_grad=True)
    out = mha(x)
    out.sum().backward()

    assert np.allclose(x.grad, numeric, atol=1e-4)


def test_causal_mask_blocks_future_positions():
    """Changing a future token must not change an earlier position's output
    under causal masking -- the actual behavior a shape check can't catch."""
    np.random.seed(22)
    batch, seq, d_model, heads = 1, 4, 8, 2
    mha = MultiHeadAttention(d_model, heads, causal=True)

    x_data = np.random.randn(batch, seq, d_model)
    out_a = mha(Tensor(x_data)).data

    x_data_changed = x_data.copy()
    x_data_changed[:, -1, :] = np.random.randn(d_model)  # perturb only the last position
    out_b = mha(Tensor(x_data_changed)).data

    # Every position except the last (whose own input changed) must be identical.
    assert np.allclose(out_a[:, :-1, :], out_b[:, :-1, :], atol=1e-10)
    assert not np.allclose(out_a[:, -1, :], out_b[:, -1, :])


def test_causal_mask_first_position_attends_only_to_itself():
    """Position 0 has no earlier positions, so its attention weights must
    put all probability mass on itself -- checked directly, not inferred
    from output shape."""
    q = k = Tensor(np.random.randn(1, 1, 3, 5))
    v = Tensor(np.random.randn(1, 1, 3, 5))
    seq = q.shape[-2]
    scores = (q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    mask = np.triu(np.full((seq, seq), -np.inf), k=1)
    weights = (scores + Tensor(mask)).softmax(axis=-1)

    assert np.allclose(weights.data[0, 0, 0], [1.0, 0.0, 0.0], atol=1e-6)
