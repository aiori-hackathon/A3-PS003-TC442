"""
Build the static, in-browser demo into build/web/ (free Hugging Face static Space).

    python demo/build_web.py

Packs: the web page, the simulator code (src/), the animated map (demo/viz.py),
the prepared dataset (data.npz) and your real 3-seed results + figures.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config, data  # noqa: E402
from src.results import load_all, summarize  # noqa: E402

WEB = ROOT / "demo" / "web"
OUT = ROOT / "build" / "web"
PY_SRC = ["__init__.py", "config.py", "model.py", "comm.py", "federated.py", "partition.py", "experiments.py"]
NAMES = {"persistence": "Persistence (naive)", "centralized": "Centralised", "fedavg_iid": "FedAvg IID",
         "fedavg_noniid": "FedAvg non-IID", "local_only_noniid": "Local-only non-IID",
         "pulsefed_noniid": "PulseFed non-IID", "pulsefed_v2_noniid": "PulseFed v2 non-IID",
         "fedprox_noniid": "FedProx non-IID", "fedavg_noniid_fp16": "FedAvg + FP16", "fedavg_noniid_int8": "FedAvg + INT8"}
CAPTIONS = {"00": "Raw traffic: quietest vs busiest sensor", "01": "Accuracy of all methods",
            "02": "Convergence per round", "03": "Communication cost", "04": "Accuracy vs bytes",
            "05": "Actual vs predicted traffic", "06": "Error per junction group", "07": "IID vs non-IID client data",
            "08": "PulseFed participation", "09": "PulseFed v2 participation"}
README = """---
title: NovaFlow Federated Traffic Demo
emoji: 🚦
colorFrom: indigo
colorTo: yellow
sdk: static
pinned: false
short_description: Live federated traffic forecasting simulator
---

# NovaFlow · Team Nova · A3-PS003-TC442 (AIORI-3, PS-003)

Live demo: our real Python simulator runs **in your browser** (Pyodide) on the UCI Traffic Flow
Forecasting dataset, comparing centralised training, FedAvg, PulseFed and PulseFed v2 on accuracy,
fairness across junctions and measured bytes, with an animated map of every round.

Team: K. Nivedsagar, Stibin Steephen · Mentor: Dr. A. Immanuel Selvakumar

Dataset: Zhao, L. (2019). *Traffic Flow Forecasting* [Dataset]. UCI Machine Learning Repository,
CC BY 4.0. https://doi.org/10.24432/C57897
"""


def main() -> int:
    if not config.MAT_FILE.exists():
        print("Dataset not found. Run: python step1_download_data.py")
        return 1
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "py" / "src").mkdir(parents=True)
    (OUT / "py" / "scipy").mkdir(parents=True)
    (OUT / "figures").mkdir()

    for f in ["index.html", "app.js", "worker.js", "style.css"]:
        shutil.copy(WEB / f, OUT / f)
    (OUT / "README.md").write_text(README, encoding="utf-8")

    files = []
    for f in PY_SRC:
        shutil.copy(ROOT / "src" / f, OUT / "py" / "src" / f); files.append(f"src/{f}")
    shutil.copy(ROOT / "demo" / "viz.py", OUT / "py" / "viz.py"); files.append("viz.py")
    shutil.copy(WEB / "bridge.py", OUT / "py" / "bridge.py"); files.append("bridge.py")
    for f in ["__init__.py", "stats.py"]:
        shutil.copy(WEB / "shim" / "scipy" / f, OUT / "py" / "scipy" / f); files.append(f"scipy/{f}")

    print("Preparing the dataset ...")
    ds = data.prepare()
    np.savez_compressed(OUT / "py" / "data.npz",
                        x_train=ds["train"]["x"], y_train=ds["train"]["y"], s_train=ds["train"]["sensor"],
                        t_train=ds["train"]["t"], x_test=ds["test"]["x"], y_test=ds["test"]["y"],
                        s_test=ds["test"]["sensor"], t_test=ds["test"]["t"], cont_mask=ds["cont_mask"],
                        meta_json=np.array(json.dumps(ds["meta"], default=str)))
    files.append("data.npz")
    (OUT / "py" / "manifest.json").write_text(json.dumps({"files": files}), encoding="utf-8")

    rows = []
    res = load_all()
    for n, r in summarize(res).items():
        std = f" ± {r['mae_std']:.4f}" if r["mae_std"] is not None else ""
        rows.append({"method": NAMES.get(n, n), "mae": f"{r['mae_mean']:.4f}{std}", "rmse": f"{r['rmse_mean']:.4f}",
                     "mb": round(r["total_bytes_mean"] / 1e6, 2) if "total_bytes_mean" in r else None,
                     "seeds": r["n_seeds"]})
    figs = []
    for f in sorted(config.FIGURES_DIR.glob("*.png")):
        shutil.copy(f, OUT / "figures" / f.name)
        figs.append({"file": f.name, "caption": CAPTIONS.get(f.name[:2], f.stem)})
    (OUT / "results_summary.json").write_text(json.dumps({"rows": rows, "figures": figs}), encoding="utf-8")

    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"Built {OUT} ({size:.1f} MB): {len(rows)} result rows, {len(figs)} figures")
    print("Preview:  python -m http.server 8000 --directory build/web   then open http://localhost:8000")
    print("Deploy:   python demo/deploy_space.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
