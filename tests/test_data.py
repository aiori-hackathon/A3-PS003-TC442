"""End-to-end data checks on a SYNTHETIC file with the real dataset's layout."""
import numpy as np
import pytest

from src import data
from tests.make_fake_dataset import make


@pytest.fixture(scope="module", params=[True, False], ids=["newest_last", "newest_first"])
def prepared(tmp_path_factory, request):
    path = tmp_path_factory.mktemp("d") / "fake.mat"
    make(path, n_train=300, n_test=200, newest_last=request.param)
    return data.prepare(path)


def test_split_is_chronological(prepared):
    assert prepared["train"]["t"].max() < prepared["test"]["t"].min()


def test_time_order_and_target_offset_are_detected(prepared):
    m = prepared["meta"]
    assert m["temporal_order_match_train"] == 1.0
    assert m["temporal_order_match_test"] == 1.0
    assert m["train_test_boundary_match"] == 1.0
    assert m["dataset_target_steps_ahead"] == 2          # like the real file
    assert m["target_steps_after_newest_lag"] == 1        # we rebuild the 15-minute target


def test_lag_1_is_the_most_recent_value(prepared):
    """Whatever the raw column order, feature 0 must be the latest observation."""
    tr = prepared["train"]
    S = len(np.unique(tr["sensor"]))
    # lag_1 at step t+1 equals the target at step t
    assert np.allclose(tr["x"][S:, 0], tr["y"][:-S], atol=1e-5)


def test_dataset_mode_keeps_provided_target(tmp_path):
    path = tmp_path / "fake.mat"
    make(path, n_train=200, n_test=100)
    ds = data.prepare(path, target_mode="dataset")
    tr = ds["train"]
    S = len(np.unique(tr["sensor"]))
    assert np.allclose(tr["x"][2 * S:, 0], tr["y"][:-2 * S], atol=1e-5)   # 2 steps ahead


def test_last_test_row_target_is_rebuilt_correctly(prepared):
    te = prepared["test"]
    S = len(np.unique(te["sensor"]))
    assert np.allclose(te["x"][S:, 0], te["y"][:-S], atol=1e-5)          # holds up to the very end


def test_wrong_layout_is_rejected(tmp_path, monkeypatch):
    from src import config
    path = tmp_path / "fake.mat"
    make(path, n_train=50, n_test=20)
    monkeypatch.setattr(config, "COLS_HOUR", (17, 41))                     # the old, wrong layout
    with pytest.raises(ValueError):
        data.prepare(path)


def test_no_missing_values_after_preparation(prepared):
    for split in ("train", "test"):
        assert np.isfinite(prepared[split]["x"]).all()
        assert np.isfinite(prepared[split]["y"]).all()


def test_missing_values_are_filled_from_training_data_only():
    rng = np.random.default_rng(0)
    X_tr, X_te = rng.random((5, 2, 47)), rng.random((3, 2, 47))
    Y_tr, Y_te = rng.random((5, 2)), rng.random((3, 2))
    Y_tr[2, 0] = np.nan
    Y_te[0, 1] = np.nan
    rep = data.fill_missing(X_tr, Y_tr, X_te, Y_te)
    assert rep["missing_Y_tr"] == 1 and rep["missing_Y_te"] == 1
    assert Y_tr[2, 0] == Y_tr[1, 0]           # carried forward
    assert np.isfinite(Y_te).all()
