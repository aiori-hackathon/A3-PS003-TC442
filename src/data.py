"""
data.py - load the UCI traffic .mat file, check it, clean it and build features.

Shapes used in this file:
    T = number of 15-minute time steps, S = number of sensors (36), F = raw features
    X : (T, S, F)  raw features for every sensor at every time step
    Y : (T, S)     traffic volume 15 minutes after the time step (the target)
"""
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse

from . import config


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------
def _cells_to_array(cells) -> np.ndarray:
    """MATLAB cell array of (S, F) matrices (often sparse) -> dense (T, S, F)."""
    flat = np.asarray(cells, dtype=object).ravel()
    mats = []
    for m in flat:
        if scipy.sparse.issparse(m):
            m = m.toarray()
        mats.append(np.asarray(m, dtype=np.float64))
    return np.stack(mats, axis=0)


def _targets_to_array(y, num_steps: int) -> np.ndarray:
    """Target matrix stored as (S, T) or (T, S) -> (T, S)."""
    y = y.toarray() if scipy.sparse.issparse(y) else np.asarray(y, dtype=np.float64)
    if y.shape[1] == num_steps:
        y = y.T
    if y.shape[0] != num_steps:
        raise ValueError(f"Target shape {y.shape} does not match {num_steps} time steps.")
    return y


def load_raw(mat_path: Path = config.MAT_FILE) -> dict:
    """Read traffic_dataset.mat and return dense NumPy arrays."""
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(
            f"{mat_path} not found. Run `python step1_download_data.py` first."
        )
    try:
        mat = scipy.io.loadmat(mat_path)
    except NotImplementedError as e:  # MATLAB v7.3 files need h5py instead
        raise RuntimeError("This .mat file is in MATLAB v7.3 format, which scipy cannot read.") from e

    required = ["tra_X_tr", "tra_Y_tr", "tra_X_te", "tra_Y_te"]
    missing = [k for k in required if k not in mat]
    if missing:
        keys = [k for k in mat if not k.startswith("__")]
        raise KeyError(f"Missing variables {missing} in the .mat file. Found: {keys}")

    X_tr = _cells_to_array(mat["tra_X_tr"])
    X_te = _cells_to_array(mat["tra_X_te"])
    Y_tr = _targets_to_array(mat["tra_Y_tr"], X_tr.shape[0])
    Y_te = _targets_to_array(mat["tra_Y_te"], X_te.shape[0])
    return {
        "X_tr": X_tr, "Y_tr": Y_tr, "X_te": X_te, "Y_te": Y_te,
        "adj": np.asarray(mat["tra_adj_mat"]) if "tra_adj_mat" in mat else None,
    }


# ---------------------------------------------------------------------------
# 2. Sanity checks
# ---------------------------------------------------------------------------
BLOCKS = {"day": "COLS_DAY", "hour": "COLS_HOUR", "direction": "COLS_DIRECTION", "road": "COLS_ROAD"}


def check_column_blocks(X: np.ndarray) -> dict:
    """Check that every one-hot block in config.py really is one-hot."""
    F = X.shape[2]
    checks = {"num_features": F, "enough_columns": F >= config.NUM_RAW_COLUMNS}
    flat = X.reshape(-1, F)
    for name, attr in BLOCKS.items():
        a, b = getattr(config, attr)
        if b > F:
            checks[name] = "out of range"
            continue
        block = flat[:, a:b]
        checks[name] = {
            "binary": bool(np.isin(block, [0, 1]).all()),
            "share_of_rows_with_exactly_one_1": float(np.mean(block.sum(axis=1) == 1)),
        }
    if config.COL_LANES < F:
        lanes = flat[:, config.COL_LANES]
        checks["lanes"] = {"min": float(lanes.min()), "max": float(lanes.max()),
                           "whole_numbers": bool(np.all(lanes == np.round(lanes)))}
    return checks


def find_newest_lag(X_tr: np.ndarray, Y_tr: np.ndarray) -> int:
    """
    The lag columns are not labelled. The most recent lag is the one most
    correlated with the future value, so we pick it from the data.
    """
    a, b = config.COLS_LAGS
    y = Y_tr.ravel()
    corrs = []
    for j in range(a, b):
        x = X_tr[:, :, j].ravel()
        corrs.append(np.corrcoef(x, y)[0, 1] if x.std() > 0 else -1.0)
    return int(a + np.nanargmax(corrs))


def _lag_slices(newest: int):
    """(older part, newer part) of the lag window, depending on its direction."""
    a, b = config.COLS_LAGS
    if newest == b - 1:                  # oldest -> newest
        return slice(a + 1, b), slice(a, b - 1)
    return slice(a, b - 1), slice(a + 1, b)


def lag_window_continuity(X_prev: np.ndarray, X_next: np.ndarray, newest: int) -> float:
    """
    Share of (row, sensor) pairs where row 'next' is exactly one 15-minute step
    after row 'prev': its lag window is the previous window shifted by one.
    1.0 = perfectly consecutive rows.
    """
    src, dst = _lag_slices(newest)
    return float(np.isclose(X_next[:, :, dst], X_prev[:, :, src], rtol=1e-5, atol=1e-7).all(axis=2).mean())


def verify_temporal_order(X: np.ndarray, newest: int) -> float:
    """Are consecutive rows consecutive 15-minute steps?"""
    if X.shape[0] < 2:
        return float("nan")
    return lag_window_continuity(X[:-1], X[1:], newest)


def verify_split_boundary(X_tr: np.ndarray, X_te: np.ndarray, newest: int) -> float:
    """Does the first test row come exactly one step after the last training row?"""
    return lag_window_continuity(X_tr[-1:], X_te[:1], newest)


def detect_target_offset(X: np.ndarray, Y: np.ndarray, newest: int, max_steps: int = 4):
    """
    How many steps after the newest lag is the provided target?
    Returns (steps, match share) for the best offset; steps is None if nothing fits.
    """
    best = (None, 0.0)
    for d in range(1, max_steps + 1):
        if X.shape[0] <= d:
            break
        share = float(np.isclose(X[d:, :, newest], Y[:-d], rtol=1e-5, atol=1e-7).mean())
        if share > best[1]:
            best = (d, share)
    return best if best[1] >= 0.9 else (None, best[1])


def next_step_target(X: np.ndarray, Y: np.ndarray, newest: int, offset: int) -> np.ndarray:
    """
    Value 15 minutes after each row's newest reading.
    Rows are consecutive, so it is the next row's newest lag. For the very last
    row we use the provided target: Y[t] = value at t + offset, so value at T = Y[T - offset].
    """
    v = np.empty_like(Y)
    v[:-1] = X[1:, :, newest]
    v[-1] = Y[len(Y) - offset]
    return v


# ---------------------------------------------------------------------------
# 3. Missing values
# ---------------------------------------------------------------------------
def fill_missing(X_tr, Y_tr, X_te, Y_te) -> dict:
    """
    Replace NaN/inf. Targets: carry the last valid value forward in time
    (per sensor). Anything still missing gets the sensor's TRAINING median,
    so no information from the test period is used.
    """
    report = {}
    arrays = {"X_tr": X_tr, "Y_tr": Y_tr, "X_te": X_te, "Y_te": Y_te}
    for k, arr in arrays.items():
        arr[~np.isfinite(arr)] = np.nan
        report[f"missing_{k}"] = int(np.isnan(arr).sum())

    train_median = np.nanmedian(Y_tr, axis=0)                     # (S,)
    train_median = np.where(np.isnan(train_median), 0.0, train_median)

    for Y in (Y_tr, Y_te):
        for s in range(Y.shape[1]):
            col = Y[:, s]
            for t in range(1, len(col)):
                if np.isnan(col[t]):
                    col[t] = col[t - 1]
            col[np.isnan(col)] = train_median[s]

    a, b = config.COLS_LAGS
    for X in (X_tr, X_te):
        lags = X[:, :, a:b]
        rows, sens, cols = np.where(np.isnan(lags))
        lags[rows, sens, cols] = train_median[sens]
        X[np.isnan(X)] = 0.0  # one-hot / static columns
    return report


# ---------------------------------------------------------------------------
# 4. Feature engineering
# ---------------------------------------------------------------------------
def build_features(X: np.ndarray, newest: int):
    """
    Turn raw columns into model inputs. Returns (features (T,S,D), names, cont_mask)
    where cont_mask marks continuous columns that should be standardised.
    """
    a, b = config.COLS_LAGS
    lags = X[:, :, a:b]
    if newest - a >= (b - a) / 2:      # newest lag is at the end -> reverse
        lags = lags[:, :, ::-1]
    # now lags[..., 0] is the most recent value, lags[..., 9] the oldest

    lag_mean = lags.mean(axis=2, keepdims=True)
    lag_std = lags.std(axis=2, keepdims=True)
    diff1 = lags[:, :, :1] - lags[:, :, 1:2]       # change over the last 15 min
    trend = lags[:, :, :1] - lags[:, :, -1:]       # change over the whole window

    hour = np.argmax(X[:, :, config.COLS_HOUR[0]:config.COLS_HOUR[1]], axis=2)
    hour_sin = np.sin(2 * np.pi * hour / 24)[..., None]
    hour_cos = np.cos(2 * np.pi * hour / 24)[..., None]

    day = X[:, :, config.COLS_DAY[0]:config.COLS_DAY[1]]
    direction = X[:, :, config.COLS_DIRECTION[0]:config.COLS_DIRECTION[1]]
    road = X[:, :, config.COLS_ROAD[0]:config.COLS_ROAD[1]]
    lanes = X[:, :, config.COL_LANES:config.COL_LANES + 1]

    parts = [
        (lags, [f"lag_{i+1}" for i in range(lags.shape[2])], True),
        (lag_mean, ["lag_mean"], True),
        (lag_std, ["lag_std"], True),
        (diff1, ["diff_1"], True),
        (trend, ["trend_10"], True),
        (hour_sin, ["hour_sin"], False),
        (hour_cos, ["hour_cos"], False),
        (day, [f"day_{i}" for i in range(day.shape[2])], False),
        (direction, [f"direction_{i}" for i in range(direction.shape[2])], False),
        (road, [f"road_{i}" for i in range(road.shape[2])], False),
        (lanes, ["lanes"], True),
    ]
    feats = np.concatenate([p[0] for p in parts], axis=2).astype(np.float32)
    names = [n for p in parts for n in p[1]]
    cont_mask = np.array([p[2] for p in parts for _ in p[1]])
    return feats, names, cont_mask


def to_long(feats: np.ndarray, Y: np.ndarray, t_offset: int = 0) -> dict:
    """(T, S, D) -> one row per (time step, sensor), ordered by time."""
    T, S, D = feats.shape
    return {
        "x": feats.reshape(T * S, D),
        "y": Y.reshape(T * S).astype(np.float32),
        "sensor": np.tile(np.arange(S), T),
        "t": np.repeat(np.arange(T) + t_offset, S),
    }


# ---------------------------------------------------------------------------
# 5. Everything together
# ---------------------------------------------------------------------------
def prepare(mat_path: Path = config.MAT_FILE, target_mode: str = None) -> dict:
    target_mode = target_mode or config.TARGET_MODE
    if target_mode not in ("next_15min", "dataset"):
        raise ValueError("TARGET_MODE must be 'next_15min' or 'dataset'")
    raw = load_raw(mat_path)
    X_tr, Y_tr, X_te, Y_te = raw["X_tr"], raw["Y_tr"], raw["X_te"], raw["Y_te"]

    checks = check_column_blocks(X_tr)
    if not checks["enough_columns"]:
        raise ValueError(f"Expected {config.NUM_RAW_COLUMNS} feature columns, got {checks['num_features']}.")
    for block in BLOCKS:
        info = checks[block]
        if not isinstance(info, dict) or not info["binary"] or info["share_of_rows_with_exactly_one_1"] < 0.99:
            raise ValueError(
                f"Column block '{block}' is not one-hot - the column layout in config.py does not "
                f"match this file. Run step2b_diagnose_layout.py and adjust config.py."
            )

    missing = fill_missing(X_tr, Y_tr, X_te, Y_te)
    newest = find_newest_lag(X_tr, Y_tr)
    X_all, Y_all = np.concatenate([X_tr, X_te]), np.concatenate([Y_tr, Y_te])
    offset, offset_share = detect_target_offset(X_all, Y_all, newest)

    meta = {
        "num_train_steps": int(X_tr.shape[0]),
        "num_test_steps": int(X_te.shape[0]),
        "num_sensors": int(X_tr.shape[1]),
        "num_raw_features": int(X_tr.shape[2]),
        "newest_lag_column": newest,
        "temporal_order_match_train": verify_temporal_order(X_tr, newest),
        "temporal_order_match_test": verify_temporal_order(X_te, newest),
        "train_test_boundary_match": verify_split_boundary(X_tr, X_te, newest),
        "dataset_target_steps_ahead": offset,
        "dataset_target_offset_match": offset_share,
        "column_checks": checks,
        **missing,
    }

    contiguous = meta["train_test_boundary_match"] > 0.99 and min(
        meta["temporal_order_match_train"], meta["temporal_order_match_test"]) > 0.99
    if target_mode == "next_15min":
        if offset is None or not contiguous:
            raise ValueError("Cannot rebuild the 15-minute target: rows are not consecutive or the "
                             "target offset is unclear. Set TARGET_MODE = 'dataset' in config.py.")
        V = next_step_target(X_all, Y_all, newest, offset)
        Y_tr, Y_te = V[:len(Y_tr)], V[len(Y_tr):]
    meta["target_mode"] = target_mode
    meta["target_steps_after_newest_lag"] = 1 if target_mode == "next_15min" else offset

    f_tr, names, cont = build_features(X_tr, newest)
    f_te, _, _ = build_features(X_te, newest)
    train = to_long(f_tr, Y_tr, 0)
    test = to_long(f_te, Y_te, X_tr.shape[0])
    meta["num_features_model"] = len(names)
    return {"train": train, "test": test, "feature_names": names,
            "cont_mask": cont, "meta": meta}


def load_processed(path: Path = None) -> dict:
    """Load the output of step3_prepare_data.py."""
    path = Path(path or config.DATA_PROCESSED / "dataset.npz")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python step3_prepare_data.py` first.")
    z = np.load(path, allow_pickle=False)
    return {
        "train": {"x": z["x_train"], "y": z["y_train"], "sensor": z["s_train"], "t": z["t_train"]},
        "test": {"x": z["x_test"], "y": z["y_test"], "sensor": z["s_test"], "t": z["t_test"]},
        "cont_mask": z["cont_mask"], "feature_names": z["feature_names"].tolist(),
    }
