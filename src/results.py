"""results.py - read saved experiment JSON files and compute mean / std across seeds."""
import csv

import numpy as np

from . import config
from .experiments import EXPERIMENTS
from .utils import load_json


def load_all(results_dir=config.RESULTS_DIR) -> dict:
    """{experiment name: [result dict per seed]} for every experiment that has results."""
    out = {}
    for name in EXPERIMENTS:
        files = sorted((results_dir / name).glob("seed_*.json"))
        if files:
            out[name] = [load_json(f) for f in files]
    return out


def mean_std(values):
    """Mean and sample standard deviation (ddof=1). std is None for a single value."""
    v = np.asarray(values, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else None)


def summarize(results: dict) -> dict:
    summary = {}
    for name, runs in results.items():
        mae_m, mae_s = mean_std([r["final"]["mae"] for r in runs])
        rmse_m, rmse_s = mean_std([r["final"]["rmse"] for r in runs])
        row = {"n_seeds": len(runs), "seeds": [r["seed"] for r in runs],
               "mae_mean": mae_m, "mae_std": mae_s, "rmse_mean": rmse_m, "rmse_std": rmse_s,
               "quick_mode": any(r.get("quick_mode") for r in runs),
               "rounds": runs[0].get("rounds")}
        if "comm" in runs[0] and "total_bytes" in runs[0]["comm"]:
            row["total_bytes_mean"], _ = mean_std([r["comm"]["total_bytes"] for r in runs])
            row["bytes_per_round_mean"], _ = mean_std([r["comm"]["bytes_per_round_mean"] for r in runs])
        summary[name] = row
    return summary


def write_summary_csv(results: dict, path=config.RESULTS_DIR / "summary.csv") -> None:
    rows = []
    for name, runs in results.items():
        for r in runs:
            rows.append({"experiment": name, "seed": r["seed"], "mae": r["final"]["mae"],
                         "rmse": r["final"]["rmse"],
                         "total_bytes": r.get("comm", {}).get("total_bytes", ""),
                         "rounds": r.get("rounds"), "quick_mode": r.get("quick_mode")})
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def rounds_to_reach(history, target_mae):
    """First round whose test MAE is within the target; None if never reached."""
    for h in history:
        if h.get("test_mae") is not None and h["test_mae"] <= target_mae:
            return h["round"]
    return None
