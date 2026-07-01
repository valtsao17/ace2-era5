#!/usr/bin/env python3
"""Shared modified-Kendall association metric for the lag_may skill plots.

Drop-in scalar association score that mirrors how scipy's kendalltau is used
elsewhere: given a predicted and observed series over the same objects (here,
the 37 JJA years), return a single number where higher = stronger top-weighted
rank agreement.

Unlike Kendall tau (bounded [-1, 1]), the modified-Kendall statistic is a
standardized z-score that emphasizes repeatability among the top-ranked years,
truncated at `k`. Sign is informative (positive z = positive association at the
top), but the scale is unbounded.

Caveat: the paper's analytic null assumes each ranking is tie-free. The default
score-to-rank conversion uses average ranks, matching R's rank() default, so raw
integer frequency counts with many ties should be interpreted cautiously.
Continuous cluster-aggregated series are effectively tie-free.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb, sqrt
from typing import Literal, Optional

import numpy as np

DEFAULT_K = 10
PLOT_NORM_PERCENTILE = 98.0
TieMethod = Literal["average", "ordinal"]


@dataclass(frozen=True)
class ModifiedKendallResult:
    """Detailed result for one modified-Kendall calculation.

    The `k` convention follows Zheng and Lo's truncated rank definition: ranks
    are replaced by min(rank, k). Thus ranks 1, ..., k - 1 remain individually
    ordered, while ranks k, ..., n are tied together.
    """

    k: int
    n: int
    agreement_count: float
    agreement_rate: float
    null_mean: float
    null_variance: float
    z_statistic: float


def _choose(n: int, p: int) -> float:
    return float(comb(n, p)) if n >= p else 0.0


def modified_kendall_null(n: int, k: int) -> tuple[float, float]:
    """Null mean/variance for the normalized agreement count.

    This is a direct Python translation of Zheng and Lo's R helper `f.theory`.
    The statistic is the number of agreements divided by n * (n - 1), not by
    choose(n, 2).
    """
    if n < 2:
        return float("nan"), float("nan")
    k = int(max(2, min(k, n)))
    nn = float(n)
    pp = float(k)
    top_factor = 1.0 - (nn - pp + 1.0) * (nn - pp) / nn / (nn - 1.0)
    mean = 0.25 * top_factor**2
    if n < 4:
        return float(mean), float("nan")
    var = (
        1.0
        / nn
        / (nn - 1.0)
        * (
            0.25 * top_factor**2
            + (nn - 2.0)
            * (nn - 3.0)
            * (
                0.25 * _choose(k - 1, 4) / _choose(n, 4)
                + 0.25
                * _choose(k - 1, 3)
                * _choose(n - k + 1, 1)
                / _choose(n, 4)
                + (1.0 / 6.0)
                * _choose(k - 1, 2)
                * _choose(n - k + 1, 2)
                / _choose(n, 4)
            )
            ** 2
            + (nn - 2.0)
            * (1.0 / 6.0)
            * (
                _choose(k - 1, 3) / _choose(n, 3)
                + _choose(k - 1, 2) * (nn - pp + 1.0) / _choose(n, 3)
            )
            ** 2
            + (nn - 2.0)
            * (1.0 / 9.0)
            * (1.0 - _choose(n - k + 1, 3) / _choose(n, 3)) ** 2
            - (nn**2 - nn) * (1.0 / 16.0) * top_factor**4
        )
    )
    return float(mean), float(var)


def _rank_average(values: np.ndarray, descending: bool) -> np.ndarray:
    """Average ranks without requiring scipy.stats.rankdata."""
    ranks = np.empty(values.size, dtype=float)
    order = np.argsort(-values if descending else values, kind="mergesort")
    sorted_vals = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_vals[end] == sorted_vals[start]:
            end += 1
        rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def rank_scores(
    scores,
    *,
    descending: bool = True,
    tie_method: TieMethod = "average",
) -> np.ndarray:
    """Convert scores to rank 1 = largest/smallest, depending on `descending`."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1:
        values = values.ravel()
    if not np.all(np.isfinite(values)):
        raise ValueError("rank_scores expects finite values only")

    if tie_method == "average":
        return _rank_average(values, descending=descending)
    if tie_method == "ordinal":
        ranks = np.empty(values.size, dtype=float)
        order = np.argsort(-values if descending else values, kind="mergesort")
        ranks[order] = np.arange(1, values.size + 1, dtype=float)
        return ranks
    raise ValueError(f"unknown tie_method={tie_method!r}")


def modified_kendall_from_ranks(rank_x, rank_y, k: int) -> ModifiedKendallResult:
    """Compute Zheng-Lo modified-Kendall z from two rank vectors."""
    rx = np.asarray(rank_x, dtype=float)
    ry = np.asarray(rank_y, dtype=float)
    if rx.shape != ry.shape:
        raise ValueError("rank_x and rank_y must have the same shape")
    if rx.ndim != 1:
        rx = rx.ravel()
        ry = ry.ravel()
    if not (np.all(np.isfinite(rx)) and np.all(np.isfinite(ry))):
        raise ValueError("rank vectors must be finite")

    n = int(rx.size)
    if n < 2:
        return ModifiedKendallResult(
            k=int(k),
            n=n,
            agreement_count=0.0,
            agreement_rate=float("nan"),
            null_mean=float("nan"),
            null_variance=float("nan"),
            z_statistic=float("nan"),
        )
    kk = int(max(2, min(int(k), n)))
    rx_c = np.minimum(rx, kk)
    ry_c = np.minimum(ry, kk)
    agreements = float(
        np.count_nonzero((rx_c[:, None] < rx_c[None, :]) & (ry_c[:, None] < ry_c[None, :]))
    )
    rate = agreements / n / (n - 1)
    mean, var = modified_kendall_null(n, kk)
    z = (rate - mean) / sqrt(var) if var > 0 else float("nan")
    return ModifiedKendallResult(kk, n, agreements, float(rate), mean, var, float(z))


def modified_kendall_from_scores(
    pred,
    obs,
    k: int,
    *,
    descending: bool = True,
    tie_method: TieMethod = "average",
) -> ModifiedKendallResult:
    """Compute modified-Kendall from paired score vectors."""
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)
    ok = np.isfinite(pred) & np.isfinite(obs)
    p = pred[ok]
    o = obs[ok]
    if p.size < 2:
        return ModifiedKendallResult(
            k=int(k),
            n=int(p.size),
            agreement_count=0.0,
            agreement_rate=float("nan"),
            null_mean=float("nan"),
            null_variance=float("nan"),
            z_statistic=float("nan"),
        )
    rank_p = rank_scores(p, descending=descending, tie_method=tie_method)
    rank_o = rank_scores(o, descending=descending, tie_method=tie_method)
    return modified_kendall_from_ranks(rank_p, rank_o, k)


def mk_z(pred, obs, k: int = DEFAULT_K, tie_method: TieMethod = "average") -> float:
    """Modified-Kendall z for one (pred, obs) series pair.

    Returns NaN for degenerate input (fewer than 4 finite paired values, or a
    constant series), matching the NaN handling around scipy's kendalltau in the
    existing skill code. `descending=True` treats larger values as the top-ranked
    objects.
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
    try:
        res = modified_kendall_from_scores(
            p, o, kk, descending=True, tie_method=tie_method
        )
    except Exception:
        return float("nan")
    return float(res.z_statistic)


def normalized_z_for_plot(
    z,
    *,
    scale: Optional[float] = None,
    percentile: float = PLOT_NORM_PERCENTILE,
) -> tuple[np.ndarray, float]:
    """Return a bounded [-1, 1] plotting field for modified-Kendall z.

    The modified-Kendall statistic is an unbounded z-score, unlike Kendall tau.
    For visual comparison with tau maps, we plot z / scale clipped to [-1, 1].
    When `scale` is omitted, use a robust symmetric scale from the requested
    percentile of |z|. The raw z values should still be saved for analysis.
    """
    arr = np.asarray(z, dtype=np.float32)
    if scale is None:
        finite = np.abs(arr[np.isfinite(arr)])
        scale = float(np.nanpercentile(finite, percentile)) if finite.size else 1.0
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    np.divide(arr, scale, out=out, where=np.isfinite(arr))
    return np.clip(out, -1.0, 1.0).astype(np.float32), scale
