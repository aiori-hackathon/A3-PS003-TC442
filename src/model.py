"""
model.py - the compact forecasting model, written with NumPy only.

TrafficMLP is a small neural network: features -> 64 -> 32 -> 1 with ReLU.
We write forward and backward passes by hand (about 40 lines) so that every
step of training - and therefore every number that FedAvg averages - is
visible. tests/test_model.py checks the gradients numerically.

The SAME class with the SAME settings is used by the centralised baseline,
the local-only baseline and every federated client, so results are comparable.
"""
import numpy as np

from . import config


class TrafficMLP:
    def __init__(self, num_features, hidden=config.HIDDEN_SIZES, rng=None, dtype=np.float32):
        rng = rng if rng is not None else np.random.default_rng(0)
        sizes = [num_features, *hidden, 1]
        self.num_layers = len(sizes) - 1
        self.params = {}
        for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            limit = np.sqrt(6.0 / fan_in) if i < self.num_layers - 1 else np.sqrt(1.0 / fan_in)
            self.params[f"W{i}"] = rng.uniform(-limit, limit, (fan_in, fan_out)).astype(dtype)
            self.params[f"b{i}"] = np.zeros(fan_out, dtype=dtype)

    # -- parameters as a dictionary of arrays: this is what FedAvg sends around --
    def get_params(self) -> dict:
        return {k: v.copy() for k, v in self.params.items()}

    def set_params(self, params: dict) -> None:
        for k in self.params:
            if params[k].shape != self.params[k].shape:
                raise ValueError(f"Shape mismatch for {k}: {params[k].shape} vs {self.params[k].shape}")
            self.params[k] = params[k].astype(self.params[k].dtype).copy()

    def num_parameters(self) -> int:
        return int(sum(v.size for v in self.params.values()))

    # -- forward / backward ------------------------------------------------------
    def forward(self, x, keep_cache=False):
        cache, a = [x], x
        for i in range(self.num_layers):
            z = a @ self.params[f"W{i}"] + self.params[f"b{i}"]
            a = np.maximum(z, 0) if i < self.num_layers - 1 else z  # ReLU, linear output
            cache.append(z)
        if keep_cache:
            self._cache = cache
        return a[:, 0]

    def backward(self, grad_out):
        """grad_out = dLoss/dPrediction, shape (batch,). Returns dLoss/dParam."""
        grads, cache = {}, self._cache
        delta = grad_out[:, None]
        for i in reversed(range(self.num_layers)):
            a_prev = cache[0] if i == 0 else np.maximum(cache[i], 0)
            grads[f"W{i}"] = a_prev.T @ delta
            grads[f"b{i}"] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.params[f"W{i}"].T) * (cache[i] > 0)
        return grads


class Adam:
    """The Adam optimiser. A new one is created for every local training run."""

    def __init__(self, params, lr=config.LEARNING_RATE, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= (self.lr * m_hat / (np.sqrt(v_hat) + self.eps)).astype(params[k].dtype)


# ---------------------------------------------------------------------------
# Standardisation that can be computed in a federated way
# ---------------------------------------------------------------------------
def local_stats(x: np.ndarray, y: np.ndarray) -> dict:
    """Each client only shares counts, sums and sums of squares - never raw rows."""
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    return {"n": np.array(len(y)), "sx": x.sum(0), "sxx": (x ** 2).sum(0),
            "sy": np.array(y.sum()), "syy": np.array((y ** 2).sum())}


class Standardizer:
    """z = (value - mean) / std, fitted on TRAINING data only."""

    def __init__(self, cont_mask: np.ndarray):
        self.cont = np.asarray(cont_mask, dtype=bool)

    def fit_from_stats(self, stats_list):
        n = float(sum(s["n"] for s in stats_list))
        sx = sum(s["sx"] for s in stats_list)
        sxx = sum(s["sxx"] for s in stats_list)
        sy = float(sum(s["sy"] for s in stats_list))
        syy = float(sum(s["syy"] for s in stats_list))
        self.x_mean = sx / n
        self.x_std = np.sqrt(np.maximum(sxx / n - self.x_mean ** 2, 0)) + 1e-8
        self.y_mean = sy / n
        self.y_std = float(np.sqrt(max(syy / n - self.y_mean ** 2, 0)) + 1e-8)
        return self

    def fit(self, x, y):
        return self.fit_from_stats([local_stats(x, y)])

    def transform_x(self, x):
        x = x.astype(np.float32).copy()
        x[:, self.cont] = (x[:, self.cont] - self.x_mean[self.cont]) / self.x_std[self.cont]
        return x

    def transform_y(self, y):
        return ((y - self.y_mean) / self.y_std).astype(np.float32)

    def inverse_y(self, y_scaled):
        return y_scaled * self.y_std + self.y_mean


# ---------------------------------------------------------------------------
# Training and prediction
# ---------------------------------------------------------------------------
def train_epochs(model, x, y, epochs, rng, lr=config.LEARNING_RATE,
                 batch_size=config.BATCH_SIZE, prox_mu=0.0, global_params=None):
    """
    Mini-batch training with Adam and mean-squared-error loss.
    prox_mu > 0 adds the FedProx term (mu/2)*||w - w_global||^2, which keeps a
    client's model close to the global model on non-IID data.
    Returns the average training loss (MSE, scaled units) of the last epoch.
    """
    opt = Adam(model.params, lr=lr)
    last = float("nan")
    for _ in range(epochs):
        perm = rng.permutation(len(y))
        total = 0.0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            pred = model.forward(x[idx], keep_cache=True)
            err = pred - y[idx]
            total += float((err ** 2).sum())
            grads = model.backward(2.0 * err / len(idx))
            if prox_mu > 0 and global_params is not None:
                for k in grads:
                    grads[k] = grads[k] + prox_mu * (model.params[k] - global_params[k])
            opt.step(model.params, grads)
        last = total / len(y)
    return last


def predict(model, x: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    return np.concatenate([model.forward(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
