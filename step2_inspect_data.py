"""
STEP 2 - Inspect the raw dataset before trusting it.

    python step2_inspect_data.py

Prints shapes, value ranges, missing values and checks that
  * the column layout matches config.py,
  * rows are in time order (newest lag at t+1 == target at t),
  * the test period starts right after the training period.
Writes reports/data_inspection.json and figures/00_raw_traffic.png.
"""
import sys

import matplotlib
matplotlib.use("Agg")  # save figures without opening windows
import matplotlib.pyplot as plt
import numpy as np

from src import config, data
from src.utils import save_json


def main() -> int:
    try:
        raw = data.load_raw()
    except (FileNotFoundError, KeyError, RuntimeError) as e:
        print(f"ERROR: {e}")
        return 1

    X_tr, Y_tr, X_te, Y_te = raw["X_tr"], raw["Y_tr"], raw["X_te"], raw["Y_te"]
    print("=== Shapes (time steps, sensors, features) ===")
    print(f"  X train {X_tr.shape} | Y train {Y_tr.shape}")
    print(f"  X test  {X_te.shape} | Y test  {Y_te.shape}")
    if raw["adj"] is not None:
        print(f"  adjacency matrix {raw['adj'].shape}")

    print("\n=== Target as provided in the file (tra_Y; see time-order checks for its horizon) ===")
    for name, Y in [("train", Y_tr), ("test", Y_te)]:
        print(f"  {name}: min {np.nanmin(Y):.4f}  mean {np.nanmean(Y):.4f}  "
              f"max {np.nanmax(Y):.4f}  missing {int(np.isnan(Y).sum())}")

    print("\n=== First sensor, first time step, all raw features ===")
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    print(X_tr[0, 0])

    print("\n=== Column-block checks (layout from config.py) ===")
    checks = data.check_column_blocks(X_tr)
    for k, v in checks.items():
        print(f"  {k}: {v}")

    newest = data.find_newest_lag(X_tr, Y_tr)
    order_tr = data.verify_temporal_order(X_tr, newest)
    order_te = data.verify_temporal_order(X_te, newest)
    boundary = data.verify_split_boundary(X_tr, X_te, newest)
    offset, offset_share = data.detect_target_offset(
        np.concatenate([X_tr, X_te]), np.concatenate([Y_tr, Y_te]), newest)
    print("\n=== Time-order checks (lag window shifts by exactly one step per row) ===")
    print(f"  most recent lag is raw column {newest}")
    print(f"  train rows consecutive: {order_tr:.1%} of (t, sensor) pairs")
    print(f"  test rows consecutive:  {order_te:.1%}")
    print(f"  test starts one step after train ends: {boundary:.1%} of sensors")
    if offset:
        print(f"  provided target = value {offset} step(s) ({15 * offset} min) after the newest lag "
              f"({offset_share:.1%} match)")
    else:
        print("  could not tell how far ahead the provided target is")
    print(f"  target used by this project (config.TARGET_MODE = '{config.TARGET_MODE}'): "
          + ("next 15 minutes, rebuilt from consecutive rows" if config.TARGET_MODE == "next_15min"
             else "provided target as-is"))
    ok = min(order_tr, order_te, boundary) > 0.99
    print("  RESULT: " + ("chronological and contiguous - the provided split is a valid "
                          "chronological split." if ok else "WARNING: rows are not consecutive. "
                          "Run step2b_diagnose_layout.py."))

    sensor_means = Y_tr.mean(axis=0)
    print("\n=== Mean training volume per sensor (basis of the non-IID split) ===")
    print(np.round(sensor_means, 3))

    report = {
        "shapes": {"X_tr": X_tr.shape, "Y_tr": Y_tr.shape, "X_te": X_te.shape, "Y_te": Y_te.shape},
        "target_train": {"min": float(np.nanmin(Y_tr)), "mean": float(np.nanmean(Y_tr)),
                         "max": float(np.nanmax(Y_tr))},
        "missing": {"X_tr": int(np.isnan(X_tr).sum()), "Y_tr": int(np.isnan(Y_tr).sum()),
                    "X_te": int(np.isnan(X_te).sum()), "Y_te": int(np.isnan(Y_te).sum())},
        "column_checks": checks, "newest_lag_column": newest,
        "temporal_order_match_train": order_tr, "temporal_order_match_test": order_te,
        "train_test_boundary_match": boundary, "dataset_target_steps_ahead": offset,
        "dataset_target_offset_match": offset_share,
        "sensor_mean_volume": sensor_means,
    }
    save_json(report, config.REPORTS_DIR / "data_inspection.json")

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4))
    steps = min(96 * 7, Y_tr.shape[0])
    for s in [int(np.argmin(sensor_means)), int(np.argmax(sensor_means))]:
        ax.plot(Y_tr[:steps, s], label=f"sensor {s} (mean {sensor_means[s]:.2f})", lw=1)
    ax.set_xlabel("15-minute step (first week of the training period)")
    ax.set_ylabel("traffic volume")
    ax.set_title("Quietest vs busiest sensor - first week of training data")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "00_raw_traffic.png", dpi=150)

    print(f"\nSaved reports/data_inspection.json and figures/00_raw_traffic.png")
    print("Next: python step3_prepare_data.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
