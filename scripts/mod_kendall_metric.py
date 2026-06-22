#!/usr/bin/env python3
"""Shared modified-Kendall association metric for the lag_may skill plots.

Wraps modified_kendall.modified_kendall_from_scores into a drop-in scalar
association score that mirrors how scipy's kendalltau is used elsewhere: given a
predicted and observed series over the same objects (here, the 37 JJA years),
return a single number where higher = stronger top-weighted rank agreement.

Unlike Kendall τ (bounded [-1, 1]), the modified-Kendall statistic is a
standardized z-score that emphasizes repeatability among the top-ranked (most
extreme HHE-frequency) years, truncated at `k`. Sign is informative (positive z
= positive association at the top), but the scale is unbounded.

Caveat: the paper's analytic null assumes each ranking is a permutation (no
ties). Raw integer frequency counts are heavily tied, which biases z high;
continuous cluster-aggregated series are effectively tie-free.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

_MK_DIR = Path(__file__).resolve().parents[1] / "modified_kendall"
if str(_MK_DIR) not in sys.path:
    sys.path.insert(0, str(_MK_DIR))

from modified_kendall import modified_kendall_from_scores  # noqa: E402

DEFAULT_K = 10


def mk_z(pred, obs, k: int = DEFAULT_K) -> float:
    """Modified-Kendall z for one (pred, obs) series pair.

    Returns NaN for degenerate input (fewer than 4 finite paired values, or a
    constant series), matching the NaN handling around scipy's kendalltau in
    the existing skill code. `descending=True` treats larger values as the
    top-ranked objects.
    """
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)
    ok = np.isfinite(pred) & np.isfinite(obs)
    n = int(ok.sum())
    if n < 4:
        return float("nan")
    p, o = pred[ok], obs[ok]
    if p.std() < 1e-12 or o.std() < 1e-12:
        return float("nan")
    kk = int(max(2, min(int(k), n)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = modified_kendall_from_scores(
                p, o, kk, descending=True, validate_ranks=False
            )
        except Exception:
            return float("nan")
    return float(res.z_statistic)
