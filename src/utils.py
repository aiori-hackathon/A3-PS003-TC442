"""utils.py - small helpers used everywhere (seeding, saving JSON)."""
import json
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """Fix global randomness. Our code also passes explicit seeded generators."""
    random.seed(seed)
    np.random.seed(seed)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_to_builtin)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_builtin(o):
    """Let json.dump handle NumPy numbers, arrays and tuples-of-NumPy."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Cannot serialise {type(o)}")
