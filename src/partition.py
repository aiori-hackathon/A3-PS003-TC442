"""
partition.py - split the TRAINING rows across simulated clients (intersections).

IID      : every row (sensor, time step) is shuffled and dealt out evenly.
           Each client sees a random mix of all sensors -> similar distributions.
Non-IID  : each sensor belongs to exactly ONE client, like a real intersection
           that only sees its own detectors. Sensors are sorted by their mean
           training volume and cut into contiguous groups, so client 0 gets the
           quietest roads and the last client the busiest -> very different
           distributions (and different sample counts when 36 is not divisible).
"""
import numpy as np
from scipy.stats import wasserstein_distance


def iid_partition(num_rows: int, num_clients: int, rng: np.random.Generator):
    perm = rng.permutation(num_rows)
    return [np.sort(p) for p in np.array_split(perm, num_clients)]


def noniid_partition(sensor: np.ndarray, y: np.ndarray, num_clients: int):
    """Returns (list of row indices, list of sensor ids per client)."""
    sensors = np.unique(sensor)
    mean_flow = np.array([y[sensor == s].mean() for s in sensors])
    ordered = sensors[np.argsort(mean_flow, kind="stable")]
    groups = np.array_split(ordered, num_clients)
    parts = [np.where(np.isin(sensor, g))[0] for g in groups]
    return parts, [g.tolist() for g in groups]


def heterogeneity(y: np.ndarray, parts) -> dict:
    """
    How different is each client's target distribution from the whole training set?
    We use the Wasserstein (earth-mover) distance divided by the global standard
    deviation: 0 = identical distributions, larger = more non-IID.
    """
    std = y.std() if y.std() > 0 else 1.0
    dists = [float(wasserstein_distance(y[p], y) / std) for p in parts]
    return {
        "per_client_distance": dists,
        "mean_distance": float(np.mean(dists)),
        "per_client_samples": [int(len(p)) for p in parts],
        "per_client_mean_target": [float(y[p].mean()) for p in parts],
    }
