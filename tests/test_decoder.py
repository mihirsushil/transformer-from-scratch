import numpy as np

from nn.attention import DecoderTransformer
from nn.engine import Tensor
from nn.layers import Embedding, sequence_cross_entropy_loss
from tests.test_engine import numerical_gradient


def test_embedding_gradient_accumulates_duplicate_token_indices():
    """Repeated token IDs in the same sequence must accumulate every occurrence's
    gradient via `np.add.at`, not drop duplicates like naive fancy indexing `+=`."""
    np.random.seed(40)
    emb = Embedding(num_embeddings=5, embedding_dim=3)
    # Token 2 appears three times, token 0 twice
    idx = np.array([[2, 0, 2], [1, 2, 0]], dtype=np.int64)
    upstream = np.random.randn(2, 3, 3)

    def forward_np(w):
        old = emb.weight.data.copy()
        emb.weight.data[:] = w
        out = emb.weight.data[idx]
        emb.weight.data[:] = old
        return (out * upstream).sum()

    numeric = numerical_gradient(forward_np, emb.weight.data.copy())
    out = emb(idx)
    (out * Tensor(upstream)).sum().backward()

    assert np.allclose(emb.weight.grad, numeric, atol=1e-6)


def test_decoder_transformer_gradient_check_through_sequence_loss():
    """End-to-end gradient check of token embeddings, positional embeddings,
    stacked causal blocks, final LayerNorm, and LM head through cross-entropy loss."""
    np.random.seed(41)
    model = DecoderTransformer(
        vocab_size=6,
        d_model=4,
        num_heads=2,
        num_layers=2,
        max_seq_len=5,
        mlp_ratio=2,
    )
    tokens = np.array([[1, 3, 1, 4], [2, 0, 5, 2]], dtype=np.int64)
    targets = np.array([[3, 1, 4, 2], [0, 5, 2, 1]], dtype=np.int64)

    loss = sequence_cross_entropy_loss(model(tokens), targets)
    loss.backward()

    # Check both token_embedding.weight (which exercises repeated indices) and lm_head.weight
    for param in (model.token_embedding.weight, model.pos_embedding.weight, model.lm_head.weight):
        analytic = param.grad.copy()

        def f_np(w):
            old = param.data.copy()
            param.data[:] = w
            val = sequence_cross_entropy_loss(model(tokens), targets).data.copy()
            param.data[:] = old
            return val

        numeric = numerical_gradient(f_np, param.data.copy())
        assert np.allclose(analytic, numeric, atol=1e-5)


def test_decoder_transformer_is_strictly_causal_on_token_ids():
    np.random.seed(42)
    model = DecoderTransformer(
        vocab_size=8, d_model=8, num_heads=2, num_layers=2, max_seq_len=6
    )
    seq_a = np.array([[1, 2, 3, 4]], dtype=np.int64)
    seq_b = np.array([[1, 2, 3, 7]], dtype=np.int64)  # only last token differs

    logits_a = model(seq_a).data
    logits_b = model(seq_b).data

    assert np.allclose(logits_a[:, :-1, :], logits_b[:, :-1, :], atol=1e-10)
    assert not np.allclose(logits_a[:, -1, :], logits_b[:, -1, :])


def test_decoder_learns_deterministic_next_token_pattern():
    """Verify the full decoder + sequence loss + SGD loop can memorize a periodic
    sequence and generate the exact continuation."""
    np.random.seed(43)
    # Periodic cycle: 0 -> 1 -> 2 -> 3 -> 0
    model = DecoderTransformer(
        vocab_size=4, d_model=16, num_heads=2, num_layers=2, max_seq_len=8
    )
    x = np.array([[0, 1, 2, 3, 0, 1, 2], [1, 2, 3, 0, 1, 2, 3]], dtype=np.int64)
    y = np.array([[1, 2, 3, 0, 1, 2, 3], [2, 3, 0, 1, 2, 3, 0]], dtype=np.int64)

    initial_loss = float(sequence_cross_entropy_loss(model(x), y).data)
    lr = 0.15
    for _ in range(120):
        for p in model.parameters():
            p.grad = np.zeros_like(p.data)
        loss = sequence_cross_entropy_loss(model(x), y)
        loss.backward()
        for p in model.parameters():
            p.data -= lr * p.grad

    final_loss = float(sequence_cross_entropy_loss(model(x), y).data)
    assert final_loss < 0.05
    assert final_loss < initial_loss * 0.05

    generated = model.generate([0, 1], max_new_tokens=4, temperature=0.0)
    assert generated == [0, 1, 2, 3, 0, 1]
