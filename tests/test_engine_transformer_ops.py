"""Gradient checks for the ops added on top of nn-from-scratch's engine
specifically because attention needs them: batched matmul, transpose,
reshape, division, and softmax. Same central-difference method as
test_engine.py, kept in a separate file so it's clear which ops came from
the original MLP engine and which were added for this project.
"""

import numpy as np

from nn.engine import Tensor
from tests.test_engine import numerical_gradient


def test_batched_matmul():
    np.random.seed(10)
    # (batch, heads, seq, head_dim) @ (batch, heads, head_dim, seq) -- the
    # exact shape pattern attention scores use (Q @ K^T).
    a_data = np.random.randn(2, 3, 4, 5)
    b_data = np.random.randn(2, 3, 5, 4)

    numeric = numerical_gradient(lambda a: (a @ b_data).sum(), a_data.copy())

    a = Tensor(a_data, requires_grad=True)
    b = Tensor(b_data, requires_grad=True)
    out = (a @ b).sum()
    out.backward()

    assert np.allclose(a.grad, numeric, atol=1e-5)
    assert np.allclose(
        b.grad, numerical_gradient(lambda b_: (a_data @ b_).sum(), b_data.copy()), atol=1e-5
    )


def test_matmul_still_2d_correct():
    """The batched rewrite must not break the plain 2D case the MLP relies on."""
    np.random.seed(11)
    a_data = np.random.randn(4, 5)
    b_data = np.random.randn(5, 2)

    numeric = numerical_gradient(lambda a: (a @ b_data).sum(), a_data.copy())

    a = Tensor(a_data, requires_grad=True)
    b = Tensor(b_data, requires_grad=True)
    out = (a @ b).sum()
    out.backward()

    assert np.allclose(a.grad, numeric, atol=1e-5)


def test_transpose():
    np.random.seed(12)
    a_data = np.random.randn(2, 3, 4)

    numeric = numerical_gradient(lambda a: np.swapaxes(a, 1, 2).sum(), a_data.copy())

    a = Tensor(a_data, requires_grad=True)
    out = a.transpose(1, 2).sum()
    out.backward()

    assert np.allclose(a.grad, numeric, atol=1e-5)


def test_reshape():
    np.random.seed(13)
    a_data = np.random.randn(2, 6)

    numeric = numerical_gradient(lambda a: (a.reshape(2, 3, 2) ** 2).sum(), a_data.copy())

    a = Tensor(a_data, requires_grad=True)
    out = (a.reshape(2, 3, 2) ** 2).sum()
    out.backward()

    assert np.allclose(a.grad, numeric, atol=1e-5)


def test_division():
    np.random.seed(14)
    a_data = np.random.randn(3, 4)
    b_data = np.random.rand(3, 4) + 0.5  # keep denominator away from 0

    numeric = numerical_gradient(lambda a: (a / b_data).sum(), a_data.copy())

    a = Tensor(a_data, requires_grad=True)
    b = Tensor(b_data, requires_grad=True)
    out = (a / b).sum()
    out.backward()

    assert np.allclose(a.grad, numeric, atol=1e-5)
    assert np.allclose(
        b.grad, numerical_gradient(lambda b_: (a_data / b_).sum(), b_data.copy()), atol=1e-5
    )


def test_softmax_sums_to_one():
    a = Tensor(np.random.randn(4, 5), requires_grad=True)
    probs = a.softmax(axis=-1)
    assert np.allclose(probs.data.sum(axis=-1), 1.0, atol=1e-6)


def test_softmax_gradient():
    np.random.seed(15)
    a_data = np.random.randn(3, 4)
    # Weighted sum, not plain .sum() -- softmax's Jacobian has off-diagonal
    # terms that a uniform-weight sum happens to cancel out, which would
    # let a broken backward pass this check by accident.
    weights = np.random.randn(3, 4)

    def softmax_np(x):
        shifted = x - x.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=-1, keepdims=True)

    numeric = numerical_gradient(lambda a: (softmax_np(a) * weights).sum(), a_data.copy())

    a = Tensor(a_data, requires_grad=True)
    out = (a.softmax(axis=-1) * Tensor(weights)).sum()
    out.backward()

    assert np.allclose(a.grad, numeric, atol=1e-5)
