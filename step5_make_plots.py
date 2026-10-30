"""
STEP 5 - Turn saved results into figures (saved in figures/).

    python step5_make_plots.py            # real results -> figures/
    python step5_make_plots.py --quick    # smoke-test results -> figures_quick/

Only experiments that have results are plotted; missing ones are skipped
with a message, nothing is invented.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.partition import iid_partition
from src.results import load_all, summarize
from src.utils import load_json

LABELS = {
    "persistence": "Persistence (naive)",
    "centralized": "Centralised",
    "fedavg_iid": "FedAvg IID",
    "fedavg_noniid": "FedAvg non-IID",
    "local_only_noniid": "Local-only non-IID",
    "pulsefed_noniid": "PulseFed non-IID",
    "fedprox_noniid": "FedProx non-IID",
    "fedavg_noniid_fp16": "FedAvg non-IID + FP16",
    "fedavg_noniid_int8": "FedAvg non-IID + INT8",
}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / name, dpi=150)
    plt.close(fig)
    print(f"  saved {config.FIGURES_DIR.name}/{name}")


def history_matrix(runs, key):
    """Stack a per-round metric from every seed -> (seeds, rounds)."""
    lengths = {len(r["history"]) for r in runs}
    if len(lengths) > 1:   # e.g. leftovers from an old run with a different number of rounds
        print(f"  WARNING: seeds have different numbers of rounds {sorted(lengths)}; "
              f"plotting only the first {min(lengths)}. Re-run step4 to make them consistent.")
    n = min(lengths)
    return np.array([[h[key] for h in r["history"][:n]] for r in runs])


def plot_accuracy(summary):
    names = [n for n in LABELS if n in summary]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric in zip(axes, ["mae", "rmse"]):
        means = [summary[n][f"{metric}_mean"] for n in names]
        stds = [summary[n][f"{metric}_std"] or 0 for n in names]
        bars = ax.bar(range(len(names)), means, yerr=stds, capsize=4,
                      color=["#999" if n == "persistence" else "#1f77b4" if n == "centralized" else "#ff7f0e"
                             for n in names])
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([LABELS[n] for n in names], rotation=35, ha="right", fontsize=8)
        ax.set_title(f"Test {metric.upper()} (mean ± std over seeds, lower is better)")
    save(fig, "01_accuracy_comparison.png")


def plot_convergence(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric in zip(axes, ["test_mae", "test_rmse"]):
        for name, color in [("centralized", "#1f77b4"), ("fedavg_iid", "#2ca02c"),
                            ("fedavg_noniid", "#d62728"), ("pulsefed_noniid", "#ff7f0e"),
                            ("fedprox_noniid", "#9467bd")]:
            if name not in results:
                continue
            m = history_matrix(results[name], metric)
            x = np.arange(1, m.shape[1] + 1)
            ax.plot(x, m.mean(0), label=LABELS[name], color=color)
            if m.shape[0] > 1:
                ax.fill_between(x, m.mean(0) - m.std(0, ddof=1), m.mean(0) + m.std(0, ddof=1),
                                color=color, alpha=0.15)
        if "persistence" in results:
            key = metric.replace("test_", "")
            ax.axhline(results["persistence"][0]["final"][key], ls="--", color="gray",
                       label=LABELS["persistence"])
        ax.set_xlabel("communication round (centralised: epoch)")
        ax.set_ylabel(metric.replace("_", " ").upper())
        ax.set_title(f"Convergence: {metric.replace('test_', '').upper()} per round")
        ax.legend(fontsize=8)
    save(fig, "02_convergence_iid_vs_noniid.png")


def plot_communication(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    raw, labelled = None, False
    for name in ["fedavg_iid", "fedavg_noniid", "pulsefed_noniid", "fedavg_noniid_fp16", "fedavg_noniid_int8"]:
        if name not in results:
            continue
        r = results[name][0]
        h = r["history"]
        rounds = [x["round"] for x in h]
        axes[0].plot(rounds, np.array([x["cumulative_bytes"] for x in h]) / 1e6, label=LABELS[name],
                     ls="--" if name == "fedavg_iid" else "-", lw=2.5 if name == "fedavg_iid" else 1.5)
        raw = r["comm"]["raw_volume_only_bytes"]
        axes[1].bar(LABELS[name], r["comm"]["uplink_bytes_per_round_mean"] / 1e3, color="#ff7f0e",
                    label=None if labelled else "uplink")
        axes[1].bar(LABELS[name], r["comm"]["downlink_bytes_per_round"] / 1e3,
                    bottom=r["comm"]["uplink_bytes_per_round_mean"] / 1e3, color="#1f77b4",
                    label=None if labelled else "downlink")
        labelled = True
    if raw:
        axes[0].axhline(raw / 1e6, ls="--", color="black",
                        label="raw volumes sent once to cloud (centralised)")
    axes[0].set_xlabel("communication round")
    axes[0].set_ylabel("cumulative MB (measured serialized payloads)")
    axes[0].set_title("Cumulative communication")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("KB per round (all clients)")
    axes[1].set_title("Bytes per round: uplink vs downlink")
    axes[1].tick_params(axis="x", rotation=30, labelsize=8)
    axes[1].legend(fontsize=8)
    save(fig, "03_communication_cost.png")


def plot_pareto(summary):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pts = sorted([(n, r) for n, r in summary.items() if "total_bytes_mean" in r],
                 key=lambda nr: nr[1]["total_bytes_mean"])
    for i, (name, row) in enumerate(pts):
        ax.errorbar(row["total_bytes_mean"] / 1e6, row["mae_mean"], yerr=row["mae_std"] or 0,
                    fmt="o", capsize=4)
        ax.annotate(LABELS[name], (row["total_bytes_mean"] / 1e6, row["mae_mean"]),
                    textcoords="offset points", xytext=(6, 6 if i % 2 == 0 else -12), fontsize=8)
    if "centralized" in summary:
        ax.axhline(summary["centralized"]["mae_mean"], ls="--", color="#1f77b4", label="Centralised MAE")
        ax.legend(fontsize=8)
    ax.set_xlabel("total communication (MB)")
    ax.set_ylabel("test MAE")
    ax.set_title("Accuracy vs communication cost (bottom-left is best)")
    save(fig, "04_accuracy_vs_bytes.png")


def plot_actual_vs_predicted(results):
    tt = config.RESULTS_DIR / "test_targets.npz"
    if not tt.exists():
        return
    z = np.load(tt)
    mask = z["sensor"] == config.PLOT_SENSOR
    y = z["y"][mask][:config.PLOT_STEPS]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(y, color="black", lw=1.4, label="Actual")
    for name, style in [("centralized", "-"), ("fedavg_noniid", "--"), ("fedavg_iid", ":")]:
        files = sorted((config.RESULTS_DIR / name).glob("predictions_seed*.npy"))
        if files:
            p = np.load(files[0])[mask][:config.PLOT_STEPS]
            ax.plot(p, ls=style, lw=1.1, label=LABELS[name])
    ax.set_xlabel("15-minute step in the test period")
    ax.set_ylabel("traffic volume")
    ax.set_title(f"Actual vs predicted, sensor {config.PLOT_SENSOR} (first seed)")
    ax.legend(fontsize=8)
    save(fig, "05_actual_vs_predicted.png")


def plot_fairness(results):
    names = [n for n in ["centralized", "fedavg_noniid", "pulsefed_noniid", "local_only_noniid", "persistence"] if n in results]
    if not names:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        per = np.array([[c["mae"] for c in r["final"]["per_client"]] for r in results[name]])
        x = np.arange(per.shape[1]) + i * width
        ax.bar(x, per.mean(0), width, yerr=per.std(0, ddof=1) if per.shape[0] > 1 else None,
               capsize=3, label=LABELS[name])
    ax.set_xticks(np.arange(per.shape[1]) + width * (len(names) - 1) / 2)
    ax.set_xticklabels([f"client {k}" for k in range(per.shape[1])])
    ax.set_ylabel("test MAE on the client's own sensors")
    ax.set_title("Per-client accuracy (non-IID sensor groups, quietest -> busiest)")
    ax.legend(fontsize=8)
    save(fig, "06_per_client_fairness.png")


def plot_heterogeneity():
    prep = config.REPORTS_DIR / "preprocessing.json"
    ds_file = config.DATA_PROCESSED / "dataset.npz"
    if not (prep.exists() and ds_file.exists()):
        return
    info = load_json(prep)
    z = np.load(ds_file)
    y, sensor = z["y_train"], z["s_train"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    rng = np.random.default_rng(0)
    iid_parts = iid_partition(len(y), len(info["noniid_client_sensors"]), rng)  # same as step 3
    bins = np.linspace(y.min(), y.max(), 50)
    for k, p in enumerate(iid_parts):
        axes[0].hist(y[p], bins=bins, histtype="step", density=True, label=f"client {k}")
    for k, sens in enumerate(info["noniid_client_sensors"]):
        axes[1].hist(y[np.isin(sensor, sens)], bins=bins, histtype="step", density=True,
                     label=f"client {k}")
    axes[0].set_title(f"IID split (score {info['heterogeneity_iid']['mean_distance']:.3f})")
    axes[1].set_title(f"Non-IID split (score {info['heterogeneity_noniid']['mean_distance']:.3f})")
    for ax in axes:
        ax.set_xlabel("traffic volume")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density")
    save(fig, "07_client_distributions.png")


def plot_participation(results):
    """PulseFed: which client talked in which round (first seed)."""
    if "pulsefed_noniid" not in results:
        return
    h = results["pulsefed_noniid"][0]["history"]
    K = results["pulsefed_noniid"][0]["num_clients"]
    grid = np.zeros((K, len(h)))            # 0 = asleep, 1 = talked, 2 = forced refresh
    for j, row in enumerate(h):
        for k in row["active_clients"]:
            grid[k, j] = 2 if k in row.get("forced_clients", []) else 1
    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.imshow(grid, aspect="auto", cmap=ListedColormap(["#f1f5f9", "#2a9d8f", "#f59e0b"]), vmin=0, vmax=2)
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"client {k}" for k in range(K)])
    ax.set_xticks(range(0, len(h), max(1, len(h) // 15)))
    ax.set_xticklabels([str(i + 1) for i in range(0, len(h), max(1, len(h) // 15))])
    ax.set_xlabel("communication round")
    total = int((grid > 0).sum())
    ax.set_title(f"PulseFed participation (first seed): green = talked, amber = forced refresh, "
                 f"grey = asleep  |  {total} uploads vs {K * len(h)} for FedAvg")
    save(fig, "08_pulsefed_participation.png")


def main() -> int:
    if "--quick" in sys.argv:
        config.use_quick_outputs()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results = load_all()
    if not results:
        print("No results found. Run step4_run_experiments.py first.")
        return 1
    summary = summarize(results)
    if any(r["quick_mode"] for r in summary.values()):
        print("NOTE: some results come from --quick runs; figures are for checking only.")
    plot_accuracy(summary)
    plot_convergence(results)
    plot_communication(results)
    plot_pareto(summary)
    plot_actual_vs_predicted(results)
    plot_fairness(results)
    plot_heterogeneity()
    plot_participation(results)
    print("Next: python step6_make_report.py" + (" --quick" if config.QUICK_MODE else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
