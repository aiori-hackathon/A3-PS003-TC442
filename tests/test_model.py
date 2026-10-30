"""Checks that the hand-written neural network computes correct gradients."""
import numpy as np

from src.model import Standardizer, TrafficMLP, local_stats, mae, rmse, train_epochs


def test_gradients_match_finite_differences():
    """Back-propagation must agree with a numerical derivative of the loss."""
    rng = np.random.default_rng(0)
    model = TrafficMLP(5, hidden=(4, 3), rng=rng, dtype=np.float64)
    x, y = rng.normal(size=(8, 5)), rng.normal(size=8)

    def loss():
        return np.mean((model.forward(x) - y) ** 2)

    pred = model.forward(x, keep_cache=True)
    grads = model.backward(2 * (pred - y) / len(y))
    eps = 1e-6
    for k, p in model.params.items():
        for idx in np.ndindex(p.shape):
            old = p[idx]
            p[idx] = old + eps; up = loss()
            p[idx] = old - eps; down = loss()
            p[idx] = old
            assert abs((up - down) / (2 * eps) - grads[k][idx]) < 1e-5, k


def test_training_reduces_loss():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 6)).astype(np.float32)
    y = (x @ rng.normal(size=6)).astype(np.float32)
    model = TrafficMLP(6, rng=rng)
    before = np.mean((model.forward(x) - y) ** 2)
    train_epochs(model, x, y, epochs=20, rng=rng)
    assert np.mean((model.forward(x) - y) ** 2) < 0.2 * before


def test_federated_scaler_equals_pooled_scaler():
    """Combining client sums must give exactly the statistics of the pooled data."""
    rng = np.random.default_rng(2)
    x, y = rng.normal(3, 2, size=(300, 4)), rng.normal(5, 1, size=300)
    mask = np.array([True, True, False, True])
    pooled = Standardizer(mask).fit(x, y)
    parts = np.array_split(np.arange(300), 3)
    fed = Standardizer(mask).fit_from_stats([local_stats(x[p], y[p]) for p in parts])
    assert np.allclose(pooled.x_mean, fed.x_mean) and np.allclose(pooled.x_std, fed.x_std)
    assert np.isclose(pooled.y_mean, fed.y_mean) and np.isclose(pooled.y_std, fed.y_std)
    assert np.allclose(fed.inverse_y(fed.transform_y(y)), y, atol=1e-4)


def test_metrics():
    assert mae(np.array([1., 2.]), np.array([2., 4.])) == 1.5
    assert np.isclose(rmse(np.array([0., 0.]), np.array([3., 4.])), np.sqrt(12.5))
