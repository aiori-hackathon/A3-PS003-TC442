"""
experiments.py - one function per experiment type, all evaluated identically.

Every experiment is evaluated on the SAME held-out test period (all sensors,
all test time steps), in ORIGINAL traffic-volume units. The test set is never
used to choose anything (no early stopping, no tuning); we always report the
model after a fixed number of rounds/epochs.
"""
import time

import numpy as np

from . import comm, config
from .federated import run_fedavg
from .model import Standardizer, TrafficMLP, local_stats, mae, predict, rmse, train_epochs
from .partition import heterogeneity, iid_partition, noniid_partition

# name -> settings. "extra": True marks experiments beyond the PS-003 minimum.
EXPERIMENTS = {
    "persistence":        {"kind": "persistence", "extra": True},
    "centralized":        {"kind": "centralized", "extra": False},
    "fedavg_iid":         {"kind": "federated", "partition": "iid", "extra": False},
    "fedavg_noniid":      {"kind": "federated", "partition": "noniid", "extra": False},
    "local_only_noniid":  {"kind": "local_only", "partition": "noniid", "extra": True},
    "pulsefed_noniid":    {"kind": "federated", "partition": "noniid", "participation": "pulse", "extra": True},
    "fedprox_noniid":     {"kind": "federated", "partition": "noniid", "prox_mu": config.FEDPROX_MU, "extra": True},
    "fedavg_noniid_fp16": {"kind": "federated", "partition": "noniid", "compression": "fp16", "extra": True},
    "fedavg_noniid_int8": {"kind": "federated", "partition": "noniid", "compression": "int8", "extra": True},
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def make_partition(name, train, num_clients, seed):
    if name == "iid":
        parts = iid_partition(len(train["y"]), num_clients, np.random.default_rng(seed))
        return parts, None
    if name == "noniid":
        return noniid_partition(train["sensor"], train["y"], num_clients)
    raise ValueError(f"Unknown partition '{name}'")


def metrics_for(y_true, y_pred, sensor, client_sensors):
    """Global metrics + per-client metrics on each client's own sensors."""
    out = {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred)}
    if client_sensors is not None:
        out["per_client"] = []
        for k, sens in enumerate(client_sensors):
            m = np.isin(sensor, sens)
            out["per_client"].append({"client": k, "mae": mae(y_true[m], y_pred[m]),
                                      "rmse": rmse(y_true[m], y_pred[m])})
    return out


def make_eval_fn(scaler, test):
    x_test = scaler.transform_x(test["x"])

    def _eval(model):
        pred = scaler.inverse_y(predict(model, x_test))
        return {"test_mae": mae(test["y"], pred), "test_rmse": rmse(test["y"], pred)}

    return _eval, x_test


# ---------------------------------------------------------------------------
# Experiment types
# ---------------------------------------------------------------------------
def run_persistence(ds, settings, seed, num_clients):
    """'Next value = current value'. Any useful model must beat this."""
    test = ds["test"]
    pred = test["x"][:, 0]  # feature 0 is lag_1 (the most recent observed volume)
    _, sens = noniid_partition(ds["train"]["sensor"], ds["train"]["y"], num_clients)
    return {"final": metrics_for(test["y"], pred, test["sensor"], sens), "pred": pred}


def run_centralized(ds, settings, seed, num_clients, epochs):
    """All clients upload their raw training data; one model is trained on it."""
    train, test = ds["train"], ds["test"]
    scaler = Standardizer(ds["cont_mask"]).fit(train["x"], train["y"])
    eval_fn, x_test = make_eval_fn(scaler, test)
    x, y = scaler.transform_x(train["x"]), scaler.transform_y(train["y"])

    model = TrafficMLP(x.shape[1], rng=np.random.default_rng(seed))  # same init as FedAvg server
    rng = np.random.default_rng(seed + 12345)
    history = []
    for ep in range(1, epochs + 1):
        loss = train_epochs(model, x, y, 1, rng)
        history.append({"round": ep, "train_loss": loss, **eval_fn(model)})

    pred = scaler.inverse_y(predict(model, x_test))
    _, sens = noniid_partition(train["sensor"], train["y"], num_clients)
    # Communication if raw data were streamed to the cloud instead of federating:
    comm_info = {
        "raw_volume_only_bytes": comm.raw_data_bytes(len(train["y"])),
        "raw_feature_rows_bytes": comm.numpy_payload_bytes(x=train["x"], y=train["y"]),
    }
    return {"final": metrics_for(test["y"], pred, test["sensor"], sens),
            "history": history, "comm": comm_info, "pred": pred,
            "model_params": model.num_parameters()}


def run_local_only(ds, settings, seed, num_clients, epochs):
    """Each intersection trains alone on its own sensors - no collaboration."""
    train, test = ds["train"], ds["test"]
    parts, sens = make_partition(settings["partition"], train, num_clients, seed)
    pred = np.zeros_like(test["y"])
    for k, idx in enumerate(parts):
        scaler = Standardizer(ds["cont_mask"]).fit(train["x"][idx], train["y"][idx])
        model = TrafficMLP(train["x"].shape[1], rng=np.random.default_rng(seed))
        train_epochs(model, scaler.transform_x(train["x"][idx]),
                     scaler.transform_y(train["y"][idx]), epochs,
                     np.random.default_rng(seed * 1000 + k))
        m = np.isin(test["sensor"], sens[k])       # each sensor predicted by its owner
        pred[m] = scaler.inverse_y(predict(model, scaler.transform_x(test["x"][m])))
    return {"final": metrics_for(test["y"], pred, test["sensor"], sens), "pred": pred,
            "heterogeneity": heterogeneity(train["y"], parts)}


def run_federated(ds, settings, seed, num_clients, rounds, local_epochs):
    train, test = ds["train"], ds["test"]
    parts, sens = make_partition(settings["partition"], train, num_clients, seed)

    # Federated standardisation: clients send only (count, sum, sum of squares).
    stats = [local_stats(train["x"][p], train["y"][p]) for p in parts]
    stats_up = sum(len(comm.to_bytes(s)) for s in stats)
    scaler = Standardizer(ds["cont_mask"]).fit_from_stats(stats)
    scaler_payload = comm.to_bytes({"x_mean": scaler.x_mean, "x_std": scaler.x_std,
                                    "y_mean": scaler.y_mean, "y_std": scaler.y_std})
    setup_bytes = stats_up + len(scaler_payload) * num_clients

    client_data = [(scaler.transform_x(train["x"][p]), scaler.transform_y(train["y"][p]))
                   for p in parts]
    eval_fn, x_test = make_eval_fn(scaler, test)
    compression = settings.get("compression", "none")
    history, server = run_fedavg(client_data, train["x"].shape[1], seed, rounds, local_epochs,
                                 compression=compression,
                                 prox_mu=settings.get("prox_mu", 0.0), eval_fn=eval_fn,
                                 participation=settings.get("participation", "all"),
                                 min_active=config.PULSE_MIN_ACTIVE, max_sleep=config.PULSE_MAX_SLEEP)

    pred = scaler.inverse_y(predict(server.model, x_test))
    if sens is None:  # IID: every client has every sensor, show owner groups anyway
        _, sens_eval = noniid_partition(train["sensor"], train["y"], num_clients)
    else:
        sens_eval = sens
    total = history[-1]["cumulative_bytes"] + setup_bytes
    comm_info = {
        "compression": compression,
        "setup_bytes": setup_bytes,
        "bytes_per_round_mean": float(np.mean([h["round_bytes"] for h in history])),
        "downlink_bytes_per_round": float(np.mean([h["downlink_bytes"] for h in history])),
        "control_bytes_total": int(sum(h.get("control_bytes", 0) for h in history)),
        "uploads_total": int(sum(len(h["active_clients"]) for h in history)),
        "uploads_if_all_clients": int(num_clients * rounds),
        "uplink_bytes_per_round_mean": float(np.mean([h["uplink_bytes"] for h in history])),
        "total_bytes": total,
        "model_float32_bytes": server.model.num_parameters() * 4,
        "raw_volume_only_bytes": comm.raw_data_bytes(len(train["y"])),
    }
    return {"final": metrics_for(test["y"], pred, test["sensor"], sens_eval),
            "history": history, "comm": comm_info, "pred": pred,
            "heterogeneity": heterogeneity(train["y"], parts),
            "client_sensors": sens, "model_params": server.model.num_parameters()}


def run_experiment(name, ds, seed, num_clients=config.NUM_CLIENTS,
                   rounds=config.NUM_ROUNDS, local_epochs=config.LOCAL_EPOCHS):
    settings = EXPERIMENTS[name]
    kind = settings["kind"]
    start = time.time()
    if kind == "persistence":
        res = run_persistence(ds, settings, seed, num_clients)
    elif kind == "centralized":
        res = run_centralized(ds, settings, seed, num_clients, rounds * local_epochs)
    elif kind == "local_only":
        res = run_local_only(ds, settings, seed, num_clients, rounds * local_epochs)
    elif kind == "federated":
        res = run_federated(ds, settings, seed, num_clients, rounds, local_epochs)
    else:
        raise ValueError(kind)
    res.update({"experiment": name, "seed": seed, "settings": settings,
                "num_clients": num_clients, "rounds": rounds, "local_epochs": local_epochs,
                "runtime_sec": round(time.time() - start, 2)})
    return res
