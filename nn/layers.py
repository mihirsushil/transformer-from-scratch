"""Neural net building blocks on top of the Tensor autograd engine."""

import numpy as np

from .engine import Tensor


class Module:
    def parameters(self):
        return []

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class Linear(Module):
    def __init__(self, in_features, out_features):
        # He initialization: keeps activation variance stable through ReLUs.
        limit = np.sqrt(2.0 / in_features)
        self.weight = Tensor(np.random.randn(in_features, out_features) * limit, requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

    def __call__(self, x):
        return x @ self.weight + self.bias

    def parameters(self):
        return [self.weight, self.bias]


class ReLU(Module):
    def __call__(self, x):
        return x.relu()


class LayerNorm(Module):
    """Normalize each token across its feature dimension.

    BatchNorm would make one token's output depend on the other examples in
    the batch. Transformers instead normalize independently over `d_model`,
    then learn a per-feature scale and shift.
    """

    def __init__(self, d_model, eps=1e-5):
        self.gamma = Tensor(np.ones(d_model), requires_grad=True)
        self.beta = Tensor(np.zeros(d_model), requires_grad=True)
        self.eps = eps

    def __call__(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        centered = x - mean
        variance = (centered ** 2).mean(axis=-1, keepdims=True)
        normalized = centered / ((variance + self.eps) ** 0.5)
        return normalized * self.gamma + self.beta

    def parameters(self):
        return [self.gamma, self.beta]


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


def cross_entropy_loss(logits, targets):
    """Numerically-stable softmax cross-entropy, built from Tensor ops so it's
    differentiable through the engine with no special-cased gradient.

    logits: Tensor of shape (batch, num_classes)
    targets: int array of shape (batch,), class indices
    """
    batch, num_classes = logits.shape

    # Stop-gradient max subtraction for numerical stability -- mathematically
    # this doesn't change softmax's gradient, so treating it as a constant is
    # correct, not just convenient.
    max_logits = Tensor(logits.data.max(axis=1, keepdims=True))
    shifted = logits - max_logits
    log_sum_exp = shifted.exp().sum(axis=1, keepdims=True).log()
    log_probs = shifted - log_sum_exp

    one_hot = np.zeros((batch, num_classes))
    one_hot[np.arange(batch), targets] = 1.0

    nll = -(log_probs * Tensor(one_hot)).sum(axis=1)
    return nll.mean()
