"""Checks the FedAvg aggregation rule and the client partitions."""
import numpy as np
import pytest

from src.federated import fedavg_aggregate, run_fedavg
from src.partition import heterogeneity, iid_partition, noniid_partition


def test_weighted_average_uses_sample_counts():
    a = {"W0": np.zeros((2, 2), np.float32)}
    b = {"W0": np.ones((2, 2), np.float32)}
    out = fedavg_aggregate([a, b], [1, 3])          # weights 1/4 and 3/4
    assert np.allclose(out["W0"], 0.75)


def test_equal_clients_give_plain_mean():
    ps = [{"b0": np.array([v], np.float32)} for v in (1.0, 2.0, 6.0)]
    assert np.isclose(fedavg_aggregate(ps, [10, 10, 10])["b0"][0], 3.0)


def test_aggregate_rejects_bad_input():
    with pytest.raises(ValueError):
        fedavg_aggregate([], [])
    with pytest.raises(ValueError):
        fedavg_aggregate([{"w": np.ones(1)}], [0])


def test_partitions_cover_every_row_exactly_once():
    rng = np.random.default_rng(0)
    sensor = np.tile(np.arange(36), 50)
    y = rng.random(len(sensor)) + sensor * 0.1
    for parts in (iid_partition(len(y), 5, rng), noniid_partition(sensor, y, 5)[0]):
        allrows = np.concatenate(parts)
        assert len(allrows) == len(y) and len(np.unique(allrows)) == len(y)


def test_noniid_assigns_each_sensor_to_one_client_and_is_more_heterogeneous():
    rng = np.random.default_rng(0)
    sensor = np.tile(np.arange(36), 50)
    y = rng.random(len(sensor)) + sensor * 0.1
    parts, groups = noniid_partition(sensor, y, 4)
    assert sorted(s for g in groups for s in g) == list(range(36))
    het_non = heterogeneity(y, parts)["mean_distance"]
    het_iid = heterogeneity(y, iid_partition(len(y), 4, rng))["mean_distance"]
    assert het_non > het_iid


def test_fedavg_runs_and_counts_bytes():
    rng = np.random.default_rng(0)
    data = [(rng.normal(size=(100, 5)).astype(np.float32),
             rng.normal(size=100).astype(np.float32)) for _ in range(3)]
    hist, _ = run_fedavg(data, 5, seed=0, num_rounds=2, local_epochs=1)
    assert len(hist) == 2
    assert hist[1]["cumulative_bytes"] == sum(h["round_bytes"] for h in hist)
    assert all(h["downlink_bytes"] > 0 and h["uplink_bytes"] > 0 for h in hist)


# ---------------------------------------------------------------- PulseFed
from src.federated import select_active, update_score


def test_select_active_respects_minimum():
    active, forced = select_active([0.9, 0.1, 0.1, 0.1], [0, 0, 0, 0], min_active=2, max_sleep=2)
    assert len(active) >= 2 and 0 in active and forced == []


def test_select_active_forces_long_sleepers_back():
    active, forced = select_active([0.9, 0.8, 0.0, 0.0], [0, 0, 2, 1], min_active=2, max_sleep=2)
    assert 2 in active and forced == [2] and 3 not in active


def test_update_score_is_relative_change():
    g = {"w": np.array([3.0, 4.0])}                       # norm 5
    assert np.isclose(update_score({"w": np.array([3.0, 4.0])}, g), 0.0)
    assert np.isclose(update_score({"w": np.array([3.0, 9.0])}, g), 1.0)   # diff norm 5


def test_pulsefed_rules_hold_and_save_bytes():
    rng = np.random.default_rng(0)
    data = [(rng.normal(size=(80, 5)).astype(np.float32),
             (rng.normal(size=80) * (k + 1)).astype(np.float32)) for k in range(4)]
    full, _ = run_fedavg(data, 5, seed=0, num_rounds=8, local_epochs=1)
    pulse, _ = run_fedavg(data, 5, seed=0, num_rounds=8, local_epochs=1,
                          participation="pulse", min_active=2, max_sleep=2)
    assert len(pulse[0]["active_clients"]) == 4                     # everyone in round 1
    assert all(len(h["active_clients"]) >= 2 for h in pulse)         # minimum respected
    for k in range(4):                                               # never asleep > 2 rounds
        asleep = 0
        for h in pulse:
            asleep = 0 if k in h["active_clients"] else asleep + 1
            assert asleep <= 2
    assert pulse[-1]["cumulative_bytes"] <= full[-1]["cumulative_bytes"]
    assert all(h["control_bytes"] > 0 for h in pulse)                # control messages counted
