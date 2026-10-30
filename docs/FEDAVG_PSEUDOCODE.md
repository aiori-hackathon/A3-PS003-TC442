# FedAvg pseudocode (as implemented in `src/federated.py`)

Based on McMahan et al., *Communication-Efficient Learning of Deep Networks from
Decentralized Data*, AISTATS 2017 (https://proceedings.mlr.press/v54/mcmahan17a.html).

```
INPUT:  K clients, client k holds n_k training rows (n = n_1 + ... + n_K)
        R communication rounds, E local epochs, batch size B, learning rate η
        compression method C ∈ {none, fp16, int8}

# --- Setup: federated standardisation (src/experiments.py) -----------------
for each client k:
    send (n_k, Σx, Σx², Σy, Σy²) to server                  # bytes counted
server computes global mean and std, sends them to every client  # bytes counted
each client standardises its own data with these values

# --- Training ----------------------------------------------------------------
server initialises global weights w_1 (seeded)

for r = 1 .. R:
    payload ← serialise(w_r)                                 # numpy.savez -> bytes
    downlink_bytes ← len(payload) × K

    for each client k:                                       # sequential in the simulator
        w ← deserialise(payload)
        repeat E times:                                      # local epochs
            shuffle local rows (client's own seeded generator)
            for each mini-batch b of size B:
                loss ← mean((model_w(x_b) − y_b)²)
                [FedProx only] loss ← loss + (μ/2)·‖w − w_r‖²
                w ← AdamStep(w, ∇loss)                       # fresh optimiser each round
        if C = none:  update_k ← serialise(w)                # full weights (plain FedAvg)
        else:         update_k ← serialise(compress_C(w − w_r))
        uplink_bytes += len(update_k)

    for each client k:
        w_k ← w_r + decompress_C(update_k)    (or w_k ← update_k when C = none)

    w_{r+1} ← Σ_k (n_k / n) · w_k                            # sample-count-weighted average

    record test MAE, RMSE, downlink/uplink/cumulative bytes for round r

OUTPUT: w_{R+1}  (evaluated on the held-out chronological test period)
```

## INT8 compression of an update tensor Δ
```
lo ← min(Δ),  scale ← (max(Δ) − min(Δ)) / 255
q  ← round((Δ − lo) / scale)  stored as uint8      # 1 byte per value instead of 4
Δ̂  ← q · scale + lo                                # server side, error ≤ scale/2
```

## PulseFed: update-aware participation (our extension, `participation="pulse"`)

```
PARAMETERS: min_active = 2, max_sleep = 2
score_k ← 0, asleep_k ← 0 for every client k

for r = 1 .. R:
    if r = 1:  active ← all clients                          # everyone starts
    else:
        active ← { k : score_k ≥ mean(score) }               # biggest movers talk
        while |active| < min_active:                         # but at least min_active
            add the highest-scoring client not yet in active
        active ← active ∪ { k : asleep_k ≥ max_sleep }        # forced refresh
    send the active/asleep mask to every client              # control bytes counted

    run one FedAvg round with ONLY the active clients        # downlink + uplink counted
    for each active client k:
        score_k ← ‖w_k − w_r‖ / ‖w_r‖                        # how much its model moved
    asleep_k ← 0 if k active, else asleep_k + 1              # sleeping clients keep old score
```
Hypothesis: fewer bytes per round for a small accuracy cost. With min_active = 2 of 4,
the saving per round is capped near 50%. Risk: quiet roads may be skipped more often,
so per-client (equity) results are always reported next to the overall MAE.
