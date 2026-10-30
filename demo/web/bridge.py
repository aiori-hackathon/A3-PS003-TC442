"""
bridge.py - runs inside the browser (Pyodide web worker). Loads the prepared data,
runs the REAL simulator from src/, and posts every round to the page as JSON.
"""
import json

import numpy as np
from js import postMessage

import viz

DS, META, PERSIST, RAW_MB = None, {}, None, None
DENS_NONIID, DENS_IID = (1, 2, 3, 4), (3, 3, 3, 3)


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _post(obj):
    postMessage(json.dumps(obj, default=_default))


def init():
    global DS, META, PERSIST, RAW_MB, DENS_NONIID
    z = np.load("data.npz", allow_pickle=False)
    DS = {"train": {"x": z["x_train"], "y": z["y_train"], "sensor": z["s_train"], "t": z["t_train"]},
          "test": {"x": z["x_test"], "y": z["y_test"], "sensor": z["s_test"], "t": z["t_test"]},
          "cont_mask": z["cont_mask"]}
    META = json.loads(str(z["meta_json"]))
    PERSIST = float(np.mean(np.abs(DS["test"]["y"] - DS["test"]["x"][:, 0])))
    RAW_MB = len(DS["train"]["y"]) * 4 / 1e6
    from src.partition import noniid_partition
    parts, _ = noniid_partition(DS["train"]["sensor"], DS["train"]["y"], 4)
    m = np.array([DS["train"]["y"][p].mean() for p in parts])
    DENS_NONIID = tuple(int(round(1 + 4 * (v - m.min()) / max(m.max() - m.min(), 1e-9))) for v in m)


def info_json():
    return json.dumps({"persist": PERSIST, "raw_mb": RAW_MB, "meta": META}, default=_default)


def _scene(name, title, rounds, seed, row=None, res=None):
    mode = "centralized" if name == "centralized" else "federated"
    kw = dict(mode=mode, title=title, total_rounds=rounds, seed=seed,
              density=DENS_IID if name == "fedavg_iid" else DENS_NONIID, iid=(name == "fedavg_iid"),
              raw_mb=RAW_MB if mode == "centralized" else None)
    if res is not None:
        comm = res.get("comm", {})
        return viz.scene_html(round_no=rounds, mae=res["final"]["mae"],
                              mb=comm["total_bytes"] / 1e6 if "total_bytes" in comm else None,
                              per_client=[c["mae"] for c in res["final"].get("per_client", [])] or None,
                              finished=True, **kw)
    active = row.get("active_clients", list(range(4)))
    n = max(len(active), 1)
    return viz.scene_html(round_no=row["round"], mae=row.get("test_mae"), active=active,
                          forced=row.get("forced_clients", []),
                          mb=row["cumulative_bytes"] / 1e6 if "cumulative_bytes" in row else None,
                          down_kb=row["downlink_bytes"] / n / 1000 if "downlink_bytes" in row else None,
                          up_kb=row["uplink_bytes"] / n / 1000 if "uplink_bytes" in row else None, **kw)


def intro_scene():
    return viz.scene_html("federated", "Press Run to start the simulation", 0, 10, density=DENS_NONIID, finished=True)


def run(name, rounds, seed, title, job):
    from src.experiments import run_experiment

    def on_round(row):
        _post({"type": "round", "job": job, "name": name, "row": row,
               "scene": _scene(name, title, rounds, seed, row=row)})

    res = run_experiment(name, DS, seed, rounds=rounds, on_round=on_round)
    keep = {"final": res["final"], "comm": res.get("comm", {})}
    _post({"type": "done", "job": job, "name": name, "res": keep,
           "scene": _scene(name, title + " · finished", rounds, seed, res=keep)})
