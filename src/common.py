"""Shared loading and scoring for all models (same split, same metrics)."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
N = 1200
FEATS = ["elev", "slope", "n_north", "n_east", "lat", "sin_lon", "cos_lon"]
COLD_K = 110.0  # cold-trap group: true temp_max below this


def load_feat():
    return pd.read_parquet(ROOT / "data" / "features.parquet")


def grid_to_tri(a):
    """(2, N, N) array indexed [k, gi, gj] -> triangle order (gi, gj, k)."""
    return a.transpose(1, 2, 0).reshape(-1)


def score(y, p):
    e = p - y
    cold = y < COLD_K
    return {"MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()),
            "MAE_cold": np.abs(e[cold]).mean(), "RMSE_cold": np.sqrt((e[cold] ** 2).mean()),
            "MAE_warm": np.abs(e[~cold]).mean()}


def report(name, y, p, split):
    """Score one model's triangle-level predictions on val (1) and test (2)."""
    rows = [{"model": name, "split": s_name, **score(y[split == s], p[split == s])}
            for s, s_name in [(1, "val"), (2, "test")]]
    return pd.DataFrame(rows)
