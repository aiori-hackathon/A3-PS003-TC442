"""
STEP 3 - Clean the data, build features and preview the client partitions.

    python step3_prepare_data.py

Saves data/processed/dataset.npz (model-ready rows) and
reports/preprocessing.json (what was done + partition statistics).
"""
import sys

import numpy as np

from src import config, data
from src.partition import heterogeneity, iid_partition, noniid_partition
from src.utils import save_json


def main() -> int:
    try:
        ds = data.prepare()
    except Exception as e:
        print(f"ERROR while preparing data: {e}")
        return 1

    train, test, meta = ds["train"], ds["test"], ds["meta"]
    print(f"Training rows: {len(train['y']):,}  (time steps {train['t'].min()}-{train['t'].max()})")
    print(f"Test rows:     {len(test['y']):,}  (time steps {test['t'].min()}-{test['t'].max()})")
    assert train["t"].max() < test["t"].min(), "test period must come after training period"
    print("Chronological split OK: every test step comes after every training step.")
    print(f"Model features ({len(ds['feature_names'])}): {', '.join(ds['feature_names'])}")
    print(f"Missing values filled: { {k: v for k, v in meta.items() if k.startswith('missing_')} }")

    K = config.NUM_CLIENTS
    iid_parts = iid_partition(len(train["y"]), K, np.random.default_rng(0))
    non_parts, non_sensors = noniid_partition(train["sensor"], train["y"], K)
    het_iid = heterogeneity(train["y"], iid_parts)
    het_non = heterogeneity(train["y"], non_parts)

    print(f"\n=== {K} simulated clients ===")
    print("client | IID rows | IID mean | non-IID rows | non-IID mean | non-IID sensors")
    for k in range(K):
        print(f"  {k:4d} | {het_iid['per_client_samples'][k]:8,} | "
              f"{het_iid['per_client_mean_target'][k]:8.3f} | "
              f"{het_non['per_client_samples'][k]:12,} | "
              f"{het_non['per_client_mean_target'][k]:12.3f} | {non_sensors[k]}")
    print(f"\nHeterogeneity score (0 = identical to the global distribution):")
    print(f"  IID     {het_iid['mean_distance']:.3f}")
    print(f"  non-IID {het_non['mean_distance']:.3f}")

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.DATA_PROCESSED / "dataset.npz",
        x_train=train["x"], y_train=train["y"], s_train=train["sensor"], t_train=train["t"],
        x_test=test["x"], y_test=test["y"], s_test=test["sensor"], t_test=test["t"],
        cont_mask=ds["cont_mask"], feature_names=np.array(ds["feature_names"]),
    )
    save_json({"meta": meta, "feature_names": ds["feature_names"],
               "heterogeneity_iid": het_iid, "heterogeneity_noniid": het_non,
               "noniid_client_sensors": non_sensors},
              config.REPORTS_DIR / "preprocessing.json")
    print("\nSaved data/processed/dataset.npz and reports/preprocessing.json")
    print("Next: python step4_run_experiments.py --quick   (then without --quick)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
