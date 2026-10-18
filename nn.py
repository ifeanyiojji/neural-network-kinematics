"""
nn.py - A small multilayer perceptron implemented with NumPy only.

Maths implemented by hand:
  * forward pass:        a_l = tanh(W_l a_{l-1} + b_l), linear output layer
  * loss:                mean squared error
  * backpropagation:     chain rule applied layer by layer
  * Adam optimiser:      first/second moment estimates with bias correction
  * input Jacobian:      d(output)/d(input) by the same chain rule (used to
                         compare against the analytic robot Jacobian)
"""
import numpy as np


class MLP:
    def __init__(self, sizes, seed=0):
        """sizes e.g. [2, 64, 64, 2]. Xavier init suits tanh units."""
        rng = np.random.default_rng(seed)
        self.sizes = sizes
        self.W, self.b = [], []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            limit = np.sqrt(6.0 / (n_in + n_out))
            self.W.append(rng.uniform(-limit, limit, size=(n_in, n_out)))
            self.b.append(np.zeros(n_out))
        # Adam state
        self._m = [np.zeros_like(p) for p in self.params()]
        self._v = [np.zeros_like(p) for p in self.params()]
        self._t = 0

    # ---------- parameters ----------
    def params(self):
        return [p for pair in zip(self.W, self.b) for p in pair]

    # ---------- forward ----------
    def forward(self, X, cache=False):
        a = X
        acts = [a]
        for l in range(len(self.W) - 1):
            a = np.tanh(a @ self.W[l] + self.b[l])
            acts.append(a)
        out = a @ self.W[-1] + self.b[-1]  # linear output (regression)
        return (out, acts) if cache else out

    # ---------- loss ----------
    @staticmethod
    def mse(pred, y):
        return np.mean((pred - y) ** 2)

    # ---------- backward ----------
    def gradients(self, X, y):
        """Return (loss, [dW0, db0, dW1, db1, ...]) for MSE loss."""
        n = X.shape[0]
        out, acts = self.forward(X, cache=True)
        loss = self.mse(out, y)

        # dL/d(out) for L = mean over all elements of (out - y)^2
        delta = 2.0 * (out - y) / out.size
        dW = [None] * len(self.W)
        db = [None] * len(self.b)

        # output layer
        dW[-1] = acts[-1].T @ delta
        db[-1] = delta.sum(axis=0)
        # hidden layers: tanh'(z) = 1 - tanh(z)^2 = 1 - a^2
        for l in range(len(self.W) - 2, -1, -1):
            delta = (delta @ self.W[l + 1].T) * (1.0 - acts[l + 1] ** 2)
            dW[l] = acts[l].T @ delta
            db[l] = delta.sum(axis=0)

        grads = [g for pair in zip(dW, db) for g in pair]
        return loss, grads

    # ---------- optimiser ----------
    def adam_step(self, grads, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        for i, (p, g) in enumerate(zip(self.params(), grads)):
            self._m[i] = beta1 * self._m[i] + (1 - beta1) * g
            self._v[i] = beta2 * self._v[i] + (1 - beta2) * g ** 2
            m_hat = self._m[i] / (1 - beta1 ** self._t)
            v_hat = self._v[i] / (1 - beta2 ** self._t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)  # in-place update

    # ---------- input Jacobian ----------
    def jacobian(self, x):
        """d(output)/d(input) at a single input x, shape (n_out, n_in).
        Chain rule through the network: J = W_L^T diag(tanh') W_{L-1}^T ..."""
        a = x[None, :]
        J = np.eye(self.sizes[0])
        for l in range(len(self.W) - 1):
            z = a @ self.W[l] + self.b[l]
            a = np.tanh(z)
            J = J @ self.W[l] @ np.diag(1.0 - a[0] ** 2)
        J = J @ self.W[-1]
        return J.T

    # ---------- verification ----------
    def gradient_check(self, X, y, eps=1e-6):
        """Compare backprop gradients to centred finite differences.
        Returns the maximum relative error over all parameters."""
        _, grads = self.gradients(X, y)
        worst = 0.0
        for p, g in zip(self.params(), grads):
            it = np.nditer(p, flags=["multi_index"])
            for _ in it:
                idx = it.multi_index
                old = p[idx]
                p[idx] = old + eps
                lp = self.mse(self.forward(X), y)
                p[idx] = old - eps
                lm = self.mse(self.forward(X), y)
                p[idx] = old
                num = (lp - lm) / (2 * eps)
                denom = max(1e-12, abs(num) + abs(g[idx]))
                worst = max(worst, abs(num - g[idx]) / denom)
        return worst
