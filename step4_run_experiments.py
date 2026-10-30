"""
STEP 4 - Run every experiment for every seed.

    python step4_run_experiments.py --quick        # 3 rounds, 1 seed: checks everything works
    python step4_run_experiments.py                # full run (all seeds, all experiments)
    python step4_run_experiments.py --only centralized fedavg_iid

Writes results/<experiment>/seed_<n>.json, prediction files for the first seed,
and results/summary.csv (one row per experiment and seed, rebuilt from all saved runs).
"""
import argparse
import sys
import time

import numpy as np

from src import config, data
from src.experiments import EXPERIMENTS, run_experiment
from src.results import load_all, write_summary_csv
from src.utils import save_json, set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="3 rounds and 1 seed (smoke test)")
    p.add_argument("--only", nargs="+", choices=list(EXPERIMENTS), help="run only these experiments")
    p.add_argument("--seeds", nargs="+", type=int, default=None, help="override the seeds")
    p.add_argument("--clients", type=int, default=config.NUM_CLIENTS, help="number of clients (3-5)")
    p.add_argument("--rounds", type=int, default=config.NUM_ROUNDS)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not 3 <= args.clients <= 5:
        print("PS-003 asks for 3-5 clients.")
        return 1
    rounds = 3 if args.quick else args.rounds
    seeds = args.seeds or ([config.SEEDS[0]] if args.quick else config.SEEDS)
    names = args.only or list(EXPERIMENTS)

    try:
        ds = data.load_processed()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(config.RESULTS_DIR / "test_targets.npz",
                        y=ds["test"]["y"], sensor=ds["test"]["sensor"], t=ds["test"]["t"])
    start = time.time()
    for name in names:
        for seed in seeds:
            set_seed(seed)
            print(f"-> {name:20s} seed {seed} ...", end=" ", flush=True)
            try:
                res = run_experiment(name, ds, seed, num_clients=args.clients, rounds=rounds)
            except Exception as e:
                print(f"FAILED: {e}")
                raise
            pred = res.pop("pred")
            res["quick_mode"] = args.quick
            out_dir = config.RESULTS_DIR / name
            save_json(res, out_dir / f"seed_{seed}.json")
            if seed == seeds[0]:
                np.save(out_dir / f"predictions_seed{seed}.npy", pred.astype(np.float32))
            f = res["final"]
            total = res.get("comm", {}).get("total_bytes")
            print(f"MAE {f['mae']:.4f}  RMSE {f['rmse']:.4f}"
                  + (f"  comm {total / 1e6:.2f} MB" if total else "")
                  + f"  ({res['runtime_sec']}s)")

    # summary.csv is rebuilt from ALL saved JSON files, so --only never loses old results
    write_summary_csv(load_all())
    print(f"\nDone in {time.time() - start:.0f}s. Saved results/summary.csv")
    if args.quick:
        print("This was a QUICK smoke test. Run again without --quick for real results.")
    print("Next: python step5_make_plots.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
