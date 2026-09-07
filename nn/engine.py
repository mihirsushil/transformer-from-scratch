"""A tiny reverse-mode automatic differentiation engine, NumPy-array valued.

Same idea as Karpathy's micrograd (build a computation graph as you go,
walk it backwards to accumulate gradients via the chain rule) but operating
on NumPy arrays instead of scalars, so it's fast enough to train a real
model on real data.
"""

import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op!r})"

    @property
    def shape(self):
        return self.data.shape

    @staticmethod
    def _as_tensor(x):
        return x if isinstance(x, Tensor) else Tensor(x)

    @staticmethod
    def _unbroadcast(grad, shape):
        """Sum a broadcast-result gradient back down to `shape`."""
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(shape):
            if dim == 1 and grad.shape[i] != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    # -- elementwise ops --------------------------------------------------

    def __add__(self, other):
        other = self._as_tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data + other.data, requires_grad, (self, other), "+")

        def _backward():
            if self.requires_grad:
                self.grad += Tensor._unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += Tensor._unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-self._as_tensor(other))

    def __rsub__(self, other):
        return self._as_tensor(other) + (-self)

    def __mul__(self, other):
        other = self._as_tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data * other.data, requires_grad, (self, other), "*")

        def _backward():
            if self.requires_grad:
                self.grad += Tensor._unbroadcast(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += Tensor._unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, power):
        assert isinstance(power, (int, float)), "only scalar powers are supported"
        out = Tensor(self.data ** power, self.requires_grad, (self,), f"**{power}")

        def _backward():
            if self.requires_grad:
                self.grad += (power * self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    # -- matrix ops ---------------------------------------------------------
    # Batched: leading dims are treated as independent matmul batches, same
    # rule as np.matmul/@. Transposing only the last two axes (not .T, which
    # would reverse every axis) is what makes the backward correct for
    # attention's (batch, heads, seq, head_dim) tensors, not just 2D ones.

    def __matmul__(self, other):
        other = self._as_tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data @ other.data, requires_grad, (self, other), "@")

        def _backward():
            if self.requires_grad:
                grad = out.grad @ np.swapaxes(other.data, -1, -2)
                self.grad += Tensor._unbroadcast(grad, self.data.shape)
            if other.requires_grad:
                grad = np.swapaxes(self.data, -1, -2) @ out.grad
                other.grad += Tensor._unbroadcast(grad, other.data.shape)

        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (self._as_tensor(other) ** -1)

    def __rtruediv__(self, other):
        return self._as_tensor(other) * (self ** -1)

    def transpose(self, axis1=-2, axis2=-1):
        out = Tensor(np.swapaxes(self.data, axis1, axis2), self.requires_grad, (self,), "transpose")

        def _backward():
            if self.requires_grad:
                self.grad += np.swapaxes(out.grad, axis1, axis2)

        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), self.requires_grad, (self,), "reshape")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        """No hand-derived backward here: softmax = shift, exp, sum, divide,
        all of which are already-differentiable ops, so autograd composes
        the correct Jacobian-vector product for free through the chain rule.
        The max-subtraction is stop-gradient (same trick as cross_entropy_loss)
        since it's a constant shift that doesn't change softmax's value or
        gradient -- only done for numerical stability against exp overflow.
        """
        max_val = Tensor(self.data.max(axis=axis, keepdims=True))
        shifted = self - max_val
        exp = shifted.exp()
        return exp / exp.sum(axis=axis, keepdims=True)

    # -- unary / reduction ops --------------------------------------------

    def relu(self):
        out = Tensor(np.maximum(0, self.data), self.requires_grad, (self,), "ReLU")

        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out_data = np.exp(self.data)
        out = Tensor(out_data, self.requires_grad, (self,), "exp")

        def _backward():
            if self.requires_grad:
                self.grad += out_data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), self.requires_grad, (self,), "log")

        def _backward():
            if self.requires_grad:
                self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.requires_grad, (self,), "sum")

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if not keepdims and axis is not None:
                    grad = np.expand_dims(grad, axis)
                self.grad += np.ones_like(self.data) * grad

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # -- graph traversal ----------------------------------------------------

    def backward(self):
        assert self.requires_grad, "backward() called on a tensor that doesn't require grad"

        topo = []
        visited = set()

        def build(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build(child)
                topo.append(node)

        build(self)

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()
