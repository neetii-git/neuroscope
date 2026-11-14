"""Patient-level, seeded train/val/test split. Slices from one patient never cross sets."""
from __future__ import annotations

import json
from typing import Dict, List

import numpy as np


def make_split(
    case_ids: List[str],
    seed: int = 42,
    val_frac: float = 0.10,
    test_frac: float = 0.20,
) -> Dict[str, List[str]]:
    ids = sorted(case_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test = sorted(ids[:n_test])
    val = sorted(ids[n_test : n_test + n_val])
    train = sorted(ids[n_test + n_val :])
    assert not (set(train) & set(val)) and not (set(train) & set(test))
    assert not (set(val) & set(test))
    return {"train": train, "val": val, "test": test, "seed": seed}


def save_split(split: Dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(split, f, indent=2)


def load_split(path: str) -> Dict[str, List[str]]:
    with open(path) as f:
        return json.load(f)
