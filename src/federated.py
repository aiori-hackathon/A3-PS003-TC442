"""
federated.py - the Federated Averaging (FedAvg) simulator.

One communication round:
  1. Server serialises the global model and sends it to every client  (downlink)
  2. Each client loads it and trains on its OWN data for a few epochs
  3. Each client serialises its update and sends it back               (uplink)
  4. Server averages the client models, weighted by their sample counts
See docs/FEDAVG_PSEUDOCODE.md for the algorithm in pseudocode.
"""
import numpy as np

from . import comm
from .model import TrafficMLP, train_epochs


def fedavg_aggregate(params_list, num_samples):
    """
    Weighted average of client parameters:  w = sum_k (n_k / n) * w_k
    Clients with more data have proportionally more influence.
    """
    if len(params_list) == 0:
        raise ValueError("No client updates to aggregate.")
    if len(params_list) != len(num_samples):
        raise ValueError("Need one sample count per client.")
    total = float(sum(num_samples))
    if total <= 0:
        raise ValueError("Total number of samples must be positive.")
    return {k: sum(p[k].astype(np.float64) * (n / total) for p, n in zip(params_list, num_samples))
               .astype(np.float32)
            for k in params_list[0]}


class Client:
    """A simulated intersection holding private (already standardised) training data."""

    def __init__(self, cid, x, y, num_features, seed):
        self.cid, self.x, self.y, self.n = cid, x, y, len(y)
        self.model = TrafficMLP(num_features)
        self.rng = np.random.default_rng(seed * 1000 + cid)  # own shuffling per client

    def local_round(self, global_payload, epochs, compression, prox_mu):
        global_params = comm.decode_global(global_payload)
        self.model.set_params(global_params)
        loss = train_epochs(self.model, self.x, self.y, epochs, self.rng,
                            prox_mu=prox_mu, global_params=global_params if prox_mu > 0 else None)
        return comm.encode_update(self.model.get_params(), global_params, compression), loss


class Server:
    """Holds the global model and aggregates client updates."""

    def __init__(self, num_features, seed):
        self.model = TrafficMLP(num_features, rng=np.random.default_rng(seed))

    def broadcast(self) -> bytes:
        return comm.encode_global(self.model.get_params())

    def aggregate(self, payloads, num_samples, compression):
        """Average the client models. Returns the decoded client models (for PulseFed scoring)."""
        global_params = self.model.get_params()
        updates = [comm.decode_update(p, global_params, compression) for p in payloads]
        self.model.set_params(fedavg_aggregate(updates, num_samples))
        return updates, global_params


# ---------------------------------------------------------------------------
# PulseFed: update-aware participation (our extension of FedAvg)
# ---------------------------------------------------------------------------
def update_score(local: dict, global_: dict, eps: float = 1e-12) -> float:
    """How much did this client's model move?  r = ||w_local - w_global|| / ||w_global||"""
    diff = np.sqrt(sum(float(((local[k] - global_[k]) ** 2).sum()) for k in global_))
    norm = np.sqrt(sum(float((global_[k] ** 2).sum()) for k in global_))
    return diff / (norm + eps)


def select_active(scores, asleep_streak, min_active, max_sleep):
    """
    Choose who talks next round.
      1. clients whose last score is at or above the average score talk;
      2. if fewer than min_active, add the next-highest scorers;
      3. any client that has slept max_sleep rounds in a row is forced back in.
    Returns (sorted list of active clients, list of force-refreshed clients).
    """
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    chosen = {int(i) for i in np.where(scores >= scores.mean())[0]}
    for i in order:
        if len(chosen) >= min_active:
            break
        chosen.add(int(i))
    forced = [i for i in range(len(scores)) if asleep_streak[i] >= max_sleep and i not in chosen]
    chosen.update(forced)
    return sorted(chosen), forced


def aggregate_with_memory(old_global: dict, memory: dict, sizes: dict) -> dict:
    """
    PulseFed v2: w_new = w_old + sum_k (n_k / n) * delta_k over ALL clients, where
    delta_k is the client's freshest update (this round if it talked, otherwise the
    last update it sent). Sleeping clients send nothing; the server reuses memory.
    With every client active this equals plain FedAvg.
    """
    total = float(sum(sizes[k] for k in memory))
    return {key: (old_global[key].astype(np.float64)
                  + sum(memory[k][key].astype(np.float64) * (sizes[k] / total) for k in memory)
                  ).astype(np.float32)
            for key in old_global}


def run_fedavg(client_data, num_features, seed, num_rounds, local_epochs,
               compression="none", prox_mu=0.0, eval_fn=None,
               participation="all", min_active=2, max_sleep=2, on_round=None):
    """
    client_data: list of (x_scaled, y_scaled) tuples, one per client.
    participation: "all" = plain FedAvg (everyone, every round)
                   "pulse_reuse" = PulseFed v2: same selection, sleeping clients' last update reused
                   "pulse" = PulseFed (only clients whose model moved most, see select_active)
    eval_fn(model) -> dict of test metrics, called after every round.
    on_round(row) -> optional callback after every round (used by the live demo).
    Returns the per-round history (metrics + measured bytes) and the server.
    """
    if participation not in ("all", "pulse", "pulse_reuse"):
        raise ValueError("participation must be 'all', 'pulse' or 'pulse_reuse'")
    pulse = participation in ("pulse", "pulse_reuse")
    server = Server(num_features, seed)
    clients = [Client(k, x, y, num_features, seed) for k, (x, y) in enumerate(client_data)]
    K = len(clients)
    if pulse and not 1 <= min_active <= K:
        raise ValueError("min_active must be between 1 and the number of clients")
    scores = np.zeros(K)          # last known update size of each client
    streak = np.zeros(K, int)     # rounds asleep in a row
    memory = {}                   # PulseFed v2: last update (delta) received from each client

    history, cum_bytes = [], 0
    for rnd in range(1, num_rounds + 1):
        if not pulse or rnd == 1:                  # PulseFed: everyone talks in round 1
            active, forced = list(range(K)), []
        else:
            active, forced = select_active(scores, streak, min_active, max_sleep)

        control = 0
        if pulse:                                  # server tells every client who is active
            mask = np.zeros(K, np.uint8)
            mask[active] = 1
            control = len(comm.to_bytes({"active": mask})) * K

        payload = server.broadcast()
        down = len(payload) * len(active)          # one copy per ACTIVE client
        ups, losses = [], []
        for k in active:
            up, loss = clients[k].local_round(payload, local_epochs, compression, prox_mu)
            ups.append(up)
            losses.append(loss)
        up_bytes = sum(len(u) for u in ups)
        sizes = [clients[k].n for k in active]
        updates, old_global = server.aggregate(ups, sizes, compression)
        if participation == "pulse_reuse":         # replace the active-only average
            for k, upd in zip(active, updates):
                memory[k] = {key: upd[key] - old_global[key] for key in old_global}
            all_sizes = {k: clients[k].n for k in memory}
            server.model.set_params(aggregate_with_memory(old_global, memory, all_sizes))

        for k, upd in zip(active, updates):
            scores[k] = update_score(upd, old_global)
        streak = np.where(np.isin(np.arange(K), active), 0, streak + 1)

        round_bytes = down + up_bytes + control
        cum_bytes += round_bytes
        row = {"round": rnd, "downlink_bytes": down, "uplink_bytes": up_bytes,
               "control_bytes": control, "round_bytes": round_bytes, "cumulative_bytes": cum_bytes,
               "active_clients": active, "forced_clients": forced,
               "train_loss": float(np.average(losses, weights=sizes))}
        if eval_fn is not None:
            row.update(eval_fn(server.model))
        history.append(row)
        if on_round is not None:
            on_round(row)
    return history, server
