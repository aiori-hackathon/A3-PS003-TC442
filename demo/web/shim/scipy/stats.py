"""Exact NumPy re-implementation of scipy.stats.wasserstein_distance (unweighted 1-D case)."""
import numpy as np


def wasserstein_distance(u_values, v_values):
    u = np.sort(np.asarray(u_values, dtype=float))
    v = np.sort(np.asarray(v_values, dtype=float))
    all_values = np.sort(np.concatenate([u, v]))
    deltas = np.diff(all_values)
    u_cdf = np.searchsorted(u, all_values[:-1], side="right") / u.size
    v_cdf = np.searchsorted(v, all_values[:-1], side="right") / v.size
    return float(np.sum(np.abs(u_cdf - v_cdf) * deltas))
