"""
NovaFlow live demo (Team Nova, AIORI-3, A3-PS003-TC442).

Runs the REAL simulator from src/ on the UCI Traffic Flow Forecasting data, in
the browser, round by round. Hosted on Hugging Face Spaces; also runs locally:
    pip install gradio pandas && python app.py
"""
import queue
import sys
import threading
from pathlib import Path

# Works both in the Space (src/ next to app.py) and in the repository (python demo/app.py)
_here = Path(__file__).resolve().parent
ROOT = _here if (_here / "src").exists() else _here.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(_here))
import time  # noqa: E402

import viz  # noqa: E402  (animated map)

import gradio as gr
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src import config, data
from src.experiments import run_experiment
from src.results import load_all, summarize

COLORS = {"centralized": "#1f77b4", "fedavg_iid": "#2ca02c", "fedavg_noniid": "#d62728",
          "fedavg_noniid_int8": "#9467bd", "pulsefed_noniid": "#ff7f0e", "pulsefed_v2_noniid": "#8c564b"}
LABELS = {"centralized": "Centralised", "fedavg_iid": "FedAvg (IID junctions)",
          "fedavg_noniid": "FedAvg (non-IID junctions)", "fedavg_noniid_int8": "FedAvg + INT8 (non-IID)",
          "pulsefed_noniid": "PulseFed (non-IID)", "pulsefed_v2_noniid": "PulseFed v2 (non-IID)"}
STAGES = [
    ("centralized", "**Stage 1 · Centralised.** Every junction uploads its raw data; one model trains on all of it. "
                    "This is the accuracy reference."),
    ("fedavg_noniid", "**Stage 2 · FedAvg.** Raw data stays at 4 junctions (quiet → busy roads). Each round the server "
                      "sends the model down, junctions train locally, and the server averages their updates. Every byte is counted."),
    ("pulsefed_noniid", "**Stage 3 · PulseFed (our idea).** Junctions whose model barely changed skip the next round "
                        "(at least 2 always talk, nobody sleeps more than 2 rounds). Fewer bytes, but who pays?"),
    ("pulsefed_v2_noniid", "**Stage 4 · PulseFed v2 (follow-up).** Same skipping, but the server reuses a sleeping "
                           "junction's last update so every road group still counts."),
]

# ---------------------------------------------------------------- data (loaded once)
def _load_dataset():
    if not config.MAT_FILE.exists():
        import step1_download_data  # noqa: F401  (only present in the full repository)
        step1_download_data.main()
    return data.prepare()

DS = _load_dataset()
META = DS["meta"]
PERSIST = float(np.mean(np.abs(DS["test"]["y"] - DS["test"]["x"][:, 0])))   # "next = last" baseline MAE
RAW_MB = len(DS["train"]["y"]) * 4 / 1e6

# cars per lane on the map: busier junction group -> more cars (non-IID); equal for IID
from src.partition import noniid_partition  # noqa: E402
_parts, _ = noniid_partition(DS["train"]["sensor"], DS["train"]["y"], 4)
_means = np.array([DS["train"]["y"][p].mean() for p in _parts])
DENSITY_NONIID = tuple(int(round(1 + 4 * (m - _means.min()) / max(_means.max() - _means.min(), 1e-9))) for m in _means)
DENSITY_IID = (3, 3, 3, 3)
SPEEDS = {"Normal (2 s per round)": 2.0, "Fast (0.7 s per round)": 0.7}


def scene_for(name, title, row, rounds, seed, res=None):
    """Build the animated map for one round (or the finished state when res is given)."""
    mode = "centralized" if name == "centralized" else "federated"
    dens = DENSITY_IID if name == "fedavg_iid" else DENSITY_NONIID
    kw = dict(mode=mode, title=title, total_rounds=rounds, seed=seed, density=dens,
              iid=(name == "fedavg_iid"), raw_mb=RAW_MB if mode == "centralized" else None)
    if res is not None:
        last = res.get("history", [{}])[-1] if res.get("history") else {}
        return viz.scene_html(round_no=rounds, mae=res["final"]["mae"],
                              mb=(res["comm"]["total_bytes"] / 1e6) if "comm" in res and "total_bytes" in res["comm"] else None,
                              per_client=[c["mae"] for c in res["final"].get("per_client", [])] or None,
                              finished=True, **kw)
    active = row.get("active_clients", list(range(4)))
    n = max(len(active), 1)
    return viz.scene_html(round_no=row["round"], mae=row.get("test_mae"), active=active,
                          forced=row.get("forced_clients", []),
                          mb=row["cumulative_bytes"] / 1e6 if "cumulative_bytes" in row else None,
                          down_kb=row["downlink_bytes"] / n / 1000 if "downlink_bytes" in row else None,
                          up_kb=row["uplink_bytes"] / n / 1000 if "uplink_bytes" in row else None, **kw)


INTRO_SCENE = viz.scene_html("federated", "Press Run to start the simulation", 0, 10, density=DENSITY_NONIID,
                             finished=True)
STAGE_TITLES = {"centralized": "Stage 1 · Centralised", "fedavg_noniid": "Stage 2 · FedAvg",
                "pulsefed_noniid": "Stage 3 · PulseFed", "pulsefed_v2_noniid": "Stage 4 · PulseFed v2"}


# ---------------------------------------------------------------- plotting
def curves_figure(histories, rounds):
    fig = Figure(figsize=(11, 3.8), dpi=100)
    ax1, ax2 = fig.subplots(1, 2)
    for name, hist in histories.items():
        if not hist:
            continue
        r = [h["round"] for h in hist]
        ax1.plot(r, [h["test_mae"] for h in hist], marker="o", ms=3, color=COLORS[name], label=LABELS[name])
        if "cumulative_bytes" in hist[0]:
            ax2.plot(r, [h["cumulative_bytes"] / 1e6 for h in hist], color=COLORS[name], label=LABELS[name])
    ax1.axhline(PERSIST, ls="--", color="gray", label="Naive 'next = last'")
    ax1.set_xlim(0.5, rounds + 0.5)
    ax1.set_xlabel("round (centralised: epoch)")
    ax1.set_ylabel("test MAE (lower is better)")
    ax1.set_title("Accuracy on unseen weeks, live")
    ax1.legend(fontsize=8)
    ax2.axhline(RAW_MB, ls="--", color="black", label=f"raw counts sent once ({RAW_MB:.2f} MB)")
    ax2.set_xlim(0.5, rounds + 0.5)
    ax2.set_xlabel("round")
    ax2.set_ylabel("cumulative MB sent (measured)")
    ax2.set_title("Communication cost, live")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    return fig


def per_client_figure(results):
    fig = Figure(figsize=(11, 3.4), dpi=100)
    ax = fig.subplots()
    names = [n for n in results if results[n] and "per_client" in results[n]["final"]]
    if not names:
        return None
    width = 0.8 / len(names)
    K = len(results[names[0]]["final"]["per_client"])
    for i, n in enumerate(names):
        vals = [c["mae"] for c in results[n]["final"]["per_client"]]
        ax.bar(np.arange(K) + i * width, vals, width, color=COLORS[n], label=LABELS[n])
    ax.set_xticks(np.arange(K) + width * (len(names) - 1) / 2)
    ax.set_xticklabels(["junction 0\n(quietest)", "junction 1", "junction 2", "junction 3\n(busiest)"][:K])
    ax.set_ylabel("test MAE on own roads")
    ax.set_title("Who pays the price? Error per junction group")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def participation_figure(hist, name):
    if not hist or "active_clients" not in hist[0] or name not in ("pulsefed_noniid", "pulsefed_v2_noniid"):
        return None
    K = 4
    grid = np.zeros((K, len(hist)))
    for j, row in enumerate(hist):
        for k in row["active_clients"]:
            grid[k, j] = 2 if k in row.get("forced_clients", []) else 1
    from matplotlib.colors import ListedColormap
    fig = Figure(figsize=(11, 2.2), dpi=100)
    ax = fig.subplots()
    ax.imshow(grid, aspect="auto", cmap=ListedColormap(["#f1f5f9", "#2a9d8f", "#f59e0b"]), vmin=0, vmax=2)
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"junction {k}" for k in range(K)])
    ax.set_xticks(range(len(hist)))
    ax.set_xticklabels([str(i + 1) for i in range(len(hist))], fontsize=7)
    ax.set_xlabel("round")
    ups = int((grid > 0).sum())
    ax.set_title(f"{LABELS[name]}: green = talked, amber = forced back in, grey = asleep · "
                 f"{ups} uploads vs {K * len(hist)} for FedAvg", fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- running experiments live
def _run_streaming(name, seed, rounds):
    """Start a run in a thread; yield ('round', row) per round and finally ('done', result)."""
    q, box = queue.Queue(), {}

    def worker():
        try:
            box["res"] = run_experiment(name, DS, seed, rounds=rounds, on_round=q.put)
        except Exception as e:  # show errors in the UI instead of hanging
            box["err"] = e
        q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        row = q.get()
        if row is None:
            break
        yield "round", row
    if "err" in box:
        raise gr.Error(f"Run failed: {box['err']}")
    yield "done", box["res"]


def summary_table(results):
    rows = []
    for n, r in results.items():
        if not r:
            continue
        pcs = [c["mae"] for c in r["final"].get("per_client", [])]
        rows.append({"Method": LABELS[n], "Test MAE": round(r["final"]["mae"], 4),
                     "Test RMSE": round(r["final"]["rmse"], 4),
                     "Data sent (MB)": round(r["comm"]["total_bytes"] / 1e6, 2) if "comm" in r and "total_bytes" in r["comm"]
                     else f"raw upload {RAW_MB:.2f}",
                     "Worst junction MAE": round(max(pcs), 4) if pcs else None})
    rows.append({"Method": "Naive 'next = last'", "Test MAE": round(PERSIST, 4), "Test RMSE": None,
                 "Data sent (MB)": None, "Worst junction MAE": None})
    return pd.DataFrame(rows)


def run_tour(rounds, seed, speed):
    rounds, seed = int(rounds), int(seed)
    frame = SPEEDS.get(speed, 2.0)
    histories = {n: [] for n, _ in STAGES}
    results = {n: None for n, _ in STAGES}
    last = time.monotonic()
    for idx, (name, text) in enumerate(STAGES, 1):
        for kind, payload in _run_streaming(name, seed, rounds):
            if kind == "round":
                histories[name].append(payload)
                time.sleep(max(0.0, frame - (time.monotonic() - last)))   # let the animation play
                last = time.monotonic()
                status = f"{text}\n\nRunning stage {idx}/4 · round {payload['round']}/{rounds} · " \
                         f"test MAE {payload['test_mae']:.4f}"
                yield (scene_for(name, STAGE_TITLES[name], payload, rounds, seed), status,
                       curves_figure(histories, rounds), None, None, None)
            else:
                results[name] = payload
                yield (scene_for(name, STAGE_TITLES[name] + " · finished", None, rounds, seed, res=payload),
                       f"{text}\n\nStage {idx}/4 finished · test MAE {payload['final']['mae']:.4f}. "
                       f"Badges show each junction's own error.",
                       curves_figure(histories, rounds), per_client_figure({k: v for k, v in results.items() if v}),
                       None, None)
                time.sleep(frame * 1.5)
                last = time.monotonic()
    part = participation_figure(histories["pulsefed_noniid"], "pulsefed_noniid")
    final_scene = scene_for("pulsefed_v2_noniid", "Tour complete · PulseFed v2 shown", None, rounds, seed,
                            res=results["pulsefed_v2_noniid"])
    yield (final_scene,
           "**Tour complete.** Compare accuracy, bytes and fairness below. "
           "(Live runs use one seed and fewer rounds; the *Full results* tab has the 3-seed, 30-round numbers.)",
           curves_figure(histories, rounds), per_client_figure(results), part, summary_table(results))


def run_single(choice, rounds, seed, speed):
    name = {v: k for k, v in LABELS.items()}[choice]
    rounds, seed = int(rounds), int(seed)
    frame = SPEEDS.get(speed, 2.0)
    hist, last = [], time.monotonic()
    for kind, payload in _run_streaming(name, seed, rounds):
        if kind == "round":
            hist.append(payload)
            time.sleep(max(0.0, frame - (time.monotonic() - last)))
            last = time.monotonic()
            mb = f" · {payload['cumulative_bytes'] / 1e6:.2f} MB sent" if "cumulative_bytes" in payload else ""
            yield (scene_for(name, LABELS[name], payload, rounds, seed),
                   f"Round {payload['round']}/{rounds} · test MAE {payload['test_mae']:.4f}{mb}",
                   curves_figure({name: hist}, rounds), None, None)
        else:
            res = payload
    f = res["final"]
    msg = f"**Done.** Test MAE **{f['mae']:.4f}**, RMSE **{f['rmse']:.4f}** (naive baseline {PERSIST:.4f})."
    if "comm" in res and "total_bytes" in res["comm"]:
        msg += f" Total data sent: **{res['comm']['total_bytes'] / 1e6:.2f} MB** (raw counts: {RAW_MB:.2f} MB)."
    yield (scene_for(name, LABELS[name] + " · finished", None, rounds, seed, res=res), msg,
           curves_figure({name: hist}, rounds), per_client_figure({name: res}), participation_figure(hist, name))


# ---------------------------------------------------------------- full (pre-computed) results
def full_results_table():
    res = load_all()
    if not res:
        return pd.DataFrame({"note": ["No pre-computed results found."]})
    s = summarize(res)
    names = {"persistence": "Persistence (naive)", "centralized": "Centralised", "fedavg_iid": "FedAvg IID",
             "fedavg_noniid": "FedAvg non-IID", "local_only_noniid": "Local-only non-IID",
             "pulsefed_noniid": "PulseFed non-IID", "pulsefed_v2_noniid": "PulseFed v2 non-IID",
             "fedprox_noniid": "FedProx non-IID", "fedavg_noniid_fp16": "FedAvg + FP16",
             "fedavg_noniid_int8": "FedAvg + INT8"}
    rows = []
    for n, r in s.items():
        rows.append({"Method": names.get(n, n), "MAE (mean)": round(r["mae_mean"], 4),
                     "MAE (std)": round(r["mae_std"], 4) if r["mae_std"] is not None else None,
                     "RMSE (mean)": round(r["rmse_mean"], 4),
                     "Total MB": round(r["total_bytes_mean"] / 1e6, 2) if "total_bytes_mean" in r else None,
                     "Seeds": r["n_seeds"]})
    return pd.DataFrame(rows)


def gallery_items():
    caps = {"01": "Accuracy of all methods", "02": "Convergence per round", "03": "Communication cost",
            "04": "Accuracy vs bytes", "05": "Actual vs predicted traffic", "06": "Error per junction group",
            "07": "IID vs non-IID client data", "08": "PulseFed participation", "09": "PulseFed v2 participation",
            "00": "Raw traffic: quietest vs busiest sensor"}
    figs = sorted((ROOT / "figures").glob("*.png"))
    return [(str(f), caps.get(f.name[:2], f.stem)) for f in figs]


def about_markdown():
    pseudo = (ROOT / "docs" / "FEDAVG_PSEUDOCODE.md")
    return f"""
## The problem (AIORI-3 · PS-003)
Roadside sensors normally stream raw traffic data to the cloud. **Federated learning** keeps data at each
junction and only sends model updates. We measure what that really costs and gains: **accuracy**,
**fairness across junctions**, and **bytes**. Motto: *measure before you federate.*

## The data
UCI *Traffic Flow Forecasting*: {META['num_sensors']} sensors, 15-minute volumes, {META['num_train_steps']} training and
{META['num_test_steps']} later test steps (chronological, verified). Task: predict each sensor's volume 15 minutes ahead.
Our audit found the file has {META['num_raw_features']} columns (documentation says 47) and that the provided target is
{META.get('dataset_target_steps_ahead')} steps ahead, so we rebuilt the true 15-minute target.

## The junctions
Four simulated junctions. *IID*: rows shuffled evenly. *Non-IID*: each junction owns a group of sensors,
from the quietest to the busiest roads.

## Honest notes
Live runs here use one seed and a reduced number of rounds so they finish quickly; numbers will differ slightly
from the *Full results* tab (3 seeds × 30 rounds). Single-machine simulation; no formal privacy guarantee.

---
{pseudo.read_text(encoding="utf-8") if pseudo.exists() else ""}
"""


# ---------------------------------------------------------------- UI
with gr.Blocks(title="NovaFlow · Federated Traffic Forecasting") as demo:
    gr.Markdown("# 🚦 NovaFlow: Federated Traffic-Flow Prediction with Communication-Cost Evaluation\n"
                "**Team Nova · A3-PS003-TC442 · AIORI-3 Hackathon** · K. Nivedsagar, Stibin Steephen · "
                "Mentor: Dr. A. Immanuel Selvakumar\n\n"
                "Everything below runs our real simulator on real traffic data, live in your browser session.")
    with gr.Tab("▶ Stage-by-stage tour"):
        gr.Markdown("Runs **Centralised → FedAvg → PulseFed → PulseFed v2** one after another on the non-IID junctions. "
                    "Watch the packets: **blue** = model sent down, **amber** = update sent up, **red** = raw data "
                    "(centralised). Sleeping junctions turn grey. About 1–2 minutes at normal speed.")
        with gr.Row():
            t_rounds = gr.Slider(5, 30, value=10, step=1, label="Rounds per stage")
            t_seed = gr.Dropdown([0, 1, 2], value=0, label="Random seed")
            t_speed = gr.Radio(list(SPEEDS), value=list(SPEEDS)[0], label="Animation speed")
            t_btn = gr.Button("Run the tour", variant="primary")
        t_scene = gr.HTML(INTRO_SCENE)
        t_status = gr.Markdown()
        t_curves = gr.Plot(label="Accuracy and bytes")
        t_clients = gr.Plot(label="Fairness")
        t_part = gr.Plot(label="PulseFed participation")
        t_table = gr.Dataframe(label="Summary of this run")
        t_btn.click(run_tour, [t_rounds, t_seed, t_speed], [t_scene, t_status, t_curves, t_clients, t_part, t_table])
    with gr.Tab("🔬 Run one method"):
        with gr.Row():
            s_method = gr.Dropdown(list(LABELS.values()), value=LABELS["fedavg_noniid"], label="Method")
            s_rounds = gr.Slider(5, 30, value=10, step=1, label="Rounds")
            s_seed = gr.Dropdown([0, 1, 2], value=0, label="Random seed")
            s_speed = gr.Radio(list(SPEEDS), value=list(SPEEDS)[0], label="Animation speed")
            s_btn = gr.Button("Run", variant="primary")
        s_scene = gr.HTML(INTRO_SCENE)
        s_status = gr.Markdown()
        s_curves = gr.Plot(label="Accuracy and bytes")
        s_clients = gr.Plot(label="Fairness")
        s_part = gr.Plot(label="Participation (PulseFed only)")
        s_btn.click(run_single, [s_method, s_rounds, s_seed, s_speed], [s_scene, s_status, s_curves, s_clients, s_part])
    with gr.Tab("📊 Full results (3 seeds × 30 rounds)"):
        gr.Markdown("Pre-computed with `step4_run_experiments.py` on the real dataset (mean of seeds 0, 1, 2).")
        gr.Dataframe(full_results_table())
        gr.Gallery(gallery_items(), columns=2, height="auto", label="Figures")
    with gr.Tab("ℹ How it works"):
        gr.Markdown(about_markdown())

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch()
