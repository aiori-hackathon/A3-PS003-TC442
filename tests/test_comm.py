"""Checks serialization, byte counting and update compression."""
import numpy as np

from src import comm
from src.model import TrafficMLP


def _pair():
    rng = np.random.default_rng(0)
    g = TrafficMLP(10, rng=rng).get_params()
    local = {k: v + rng.normal(0, 0.01, v.shape).astype(np.float32) for k, v in g.items()}
    return g, local


def test_bytes_are_real_payload_length():
    g, _ = _pair()
    payload = comm.encode_global(g)
    assert isinstance(payload, bytes)
    back = comm.decode_global(payload)
    assert all(np.array_equal(back[k], g[k]) for k in g)
    # payload must hold at least all float32 values
    assert len(payload) >= sum(v.size for v in g.values()) * 4


def test_uncompressed_update_is_exact():
    g, local = _pair()
    out = comm.decode_update(comm.encode_update(local, g, "none"), g, "none")
    assert all(np.array_equal(out[k], local[k]) for k in g)


def test_compression_is_smaller_and_close():
    g, local = _pair()
    sizes = {m: len(comm.encode_update(local, g, m)) for m in comm.METHODS}
    assert sizes["int8"] < sizes["fp16"] < sizes["none"]
    for m in ("fp16", "int8"):
        out = comm.decode_update(comm.encode_update(local, g, m), g, m)
        err = max(np.abs(out[k] - local[k]).max() for k in g)
        assert err < 1e-3, (m, err)


def test_int8_roundtrip_error_is_bounded_by_half_a_step():
    t = np.linspace(-1, 1, 1000).astype(np.float32)
    q, lo, scale = comm.quantize_int8(t)
    assert np.abs(comm.dequantize_int8(q, lo, scale) - t).max() <= scale / 2 + 1e-6
