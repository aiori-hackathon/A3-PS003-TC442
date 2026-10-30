"""
comm.py - everything that "travels over the network" in the simulation.

Model parameters are packed into real bytes with numpy.savez (a standard,
portable binary format you could send over HTTP or MQTT) and we count
len(bytes). So communication cost is MEASURED, not estimated.

Uplink compression (extra feature):
    "none" : client sends its full updated float32 weights (plain FedAvg)
    "fp16" : client sends the update (local - global) as 16-bit floats
    "int8" : client sends the update quantised to 8-bit integers per tensor
Compressing the *difference* to the global model, rather than the weights,
loses much less accuracy because differences are small and similar in size.
"""
import io

import numpy as np

METHODS = ("none", "fp16", "int8")


def to_bytes(arrays: dict) -> bytes:
    """Pack a {name: array} dictionary into bytes."""
    buf = io.BytesIO()
    np.savez(buf, **{k: np.asarray(v) for k, v in arrays.items()})
    return buf.getvalue()


def from_bytes(payload: bytes) -> dict:
    with np.load(io.BytesIO(payload), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# Downlink: server -> client (always full float32 weights)
# ---------------------------------------------------------------------------
def encode_global(params: dict) -> bytes:
    return to_bytes({k: v.astype(np.float32) for k, v in params.items()})


def decode_global(payload: bytes) -> dict:
    return from_bytes(payload)


# ---------------------------------------------------------------------------
# Uplink: client -> server
# ---------------------------------------------------------------------------
def quantize_int8(t: np.ndarray):
    lo, hi = float(t.min()), float(t.max())
    scale = (hi - lo) / 255.0 or 1.0
    q = np.clip(np.round((t - lo) / scale), 0, 255).astype(np.uint8)
    return q, np.float32(lo), np.float32(scale)


def dequantize_int8(q, lo, scale) -> np.ndarray:
    return q.astype(np.float32) * scale + lo


def _flatten(params: dict):
    """All tensors -> one 1-D vector (fewer arrays = less container overhead)."""
    keys = sorted(params)
    return keys, np.concatenate([params[k].ravel() for k in keys])


def _unflatten(vec, keys, like: dict) -> dict:
    out, i = {}, 0
    for k in keys:
        n = like[k].size
        out[k] = vec[i:i + n].reshape(like[k].shape).astype(np.float32)
        i += n
    return out


def encode_update(local: dict, global_: dict, method: str = "none") -> bytes:
    if method not in METHODS:
        raise ValueError(f"Unknown compression '{method}'. Choose from {METHODS}.")
    if method == "none":
        return to_bytes({k: v.astype(np.float32) for k, v in local.items()})
    delta = {k: (local[k] - global_[k]).astype(np.float32) for k in local}
    keys, flat = _flatten(delta)
    if method == "fp16":
        return to_bytes({"delta": flat.astype(np.float16)})
    # int8: one (lo, scale) pair per tensor so small and large layers keep precision
    q_parts, los, scales = [], [], []
    for k in keys:
        q, lo, scale = quantize_int8(delta[k])
        q_parts.append(q.ravel())
        los.append(lo)
        scales.append(scale)
    return to_bytes({"q": np.concatenate(q_parts), "lo": np.array(los, np.float32),
                     "scale": np.array(scales, np.float32)})


def decode_update(payload: bytes, global_: dict, method: str = "none") -> dict:
    """Rebuild the client's float32 weights on the server side."""
    obj = from_bytes(payload)
    if method == "none":
        return obj
    keys = sorted(global_)
    if method == "fp16":
        delta = _unflatten(obj["delta"].astype(np.float32), keys, global_)
    else:
        sizes = [global_[k].size for k in keys]
        lo = np.repeat(obj["lo"], sizes)
        scale = np.repeat(obj["scale"], sizes)
        delta = _unflatten(obj["q"].astype(np.float32) * scale + lo, keys, global_)
    return {k: global_[k] + delta[k] for k in keys}


def raw_data_bytes(num_values: int, bytes_per_value: int = 4) -> int:
    """Bytes needed to send raw float32 measurements to a central server."""
    return int(num_values * bytes_per_value)


def numpy_payload_bytes(**arrays) -> int:
    return len(to_bytes(arrays))
