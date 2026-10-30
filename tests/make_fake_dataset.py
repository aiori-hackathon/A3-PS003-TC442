"""
Create a SYNTHETIC .mat file with the same structure as the UCI dataset.

Used only by the automated tests and GitHub Actions so they can run without
downloading anything. Never report results computed on this fake data.

    python tests/make_fake_dataset.py data/raw/fake_traffic_dataset.mat
"""
import sys
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse


def make(path: Path, n_train=1261, n_test=840, sensors=36, newest_last=True, seed=0):
    rng = np.random.default_rng(seed)
    total = n_train + n_test + 12
    t = np.arange(total)
    level = rng.uniform(0.05, 0.6, sensors)
    daily = np.sin(2 * np.pi * (t % 96) / 96 - np.pi / 2) + 1          # 0..2
    series = level[None, :] * (0.3 + daily[:, None]) + rng.normal(0, 0.02, (total, sensors))
    series = np.clip(series, 0, None)

    def cells(start, n):
        out = np.empty((1, n), dtype=object)
        for i in range(n):
            step = start + i                      # index of the current (newest) value
            lags = series[step - 9:step + 1].T    # (S, 10) oldest -> newest
            if not newest_last:
                lags = lags[:, ::-1]
            # same layout as the real file: lags | day(5) | hour(24) | lanes | direction(4) | road(4)
            day = np.zeros((sensors, 5)); day[:, (step // 96) % 5] = 1
            hour = np.zeros((sensors, 24)); hour[:, (step % 96) // 4] = 1
            lanes = (np.arange(sensors) % 5 + 1)[:, None].astype(float)
            direction = np.zeros((sensors, 4)); direction[np.arange(sensors), np.arange(sensors) % 4] = 1
            road = np.zeros((sensors, 4)); road[np.arange(sensors), (np.arange(sensors) // 9) % 4] = 1
            m = np.hstack([lags, day, hour, lanes, direction, road])
            out[0, i] = scipy.sparse.csc_matrix(m)
        return out

    def targets(start, n):
        # like the real file: target = value 2 steps after the newest lag
        return series[start + 2:start + 2 + n].T

    s_tr, s_te = 9, 9 + n_train
    scipy.io.savemat(path, {
        "tra_X_tr": cells(s_tr, n_train), "tra_Y_tr": targets(s_tr, n_train),
        "tra_X_te": cells(s_te, n_test), "tra_Y_te": targets(s_te, n_test),
        "tra_adj_mat": np.eye(sensors),
    })
    return path


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/fake_traffic_dataset.mat")
    out.parent.mkdir(parents=True, exist_ok=True)
    make(out)
    print(f"Wrote synthetic dataset to {out}")
