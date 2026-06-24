#!/usr/bin/env python3
"""Johnson (2013) "maximum number of distinguishable clusters" criterion.

Implements the optimal-K method of Johnson (2013), "How many ENSO flavors can we
distinguish?" (J. Climate, doi:10.1175/JCLI-D-12-00649.1, Section 2b) — as
opposed to the within/between pattern-correlation *intersection* method
(Frontiers 2022) used by cluster_similarity / select_optimal_intersection.

Johnson's criterion
-------------------
For each cluster count K, every pair of cluster patterns (i, j) is tested for
*statistical distinguishability*:

  1. At each point of the "field" dimension, a two-sample (Welch) Student's t
     difference-of-means test compares the members of cluster i against the
     members of cluster j, yielding M p-values (M = size of the field).
  2. A False Discovery Rate (FDR) field-significance test (Benjamini & Hochberg
     1995; Wilks 2006) is applied to those M p-values at level q (default 0.05):
     the largest p-value satisfying p(m) <= q * m / M defines the local
     threshold. If *any* local test survives, the two patterns are statistically
     **distinguishable** at the field-significance level q; otherwise they are
     **indistinguishable**.
  3. Count the number of indistinguishable pairs for each K.

  K* = the largest K with ZERO indistinguishable pairs (Johnson found K*=9 for
  ENSO SST patterns). K*+1 would introduce at least one indistinguishable pair.

Field/member mapping for this project
-------------------------------------
Johnson clusters seasonal SST *fields*: samples = years, the t-test runs over the
M spatial grid points, and the members within a cluster = the years assigned to
it (approximately independent samples, as required by the t-test). We follow that
faithful setup here: cluster the 37 yearly HHE-frequency fields, so the t-test
runs over the grid CELLS (the field) and a cluster's members are the YEARS
assigned to it.

Do NOT run this test on a transposed cell-regionalization (samples = grid cells,
members of a cluster = cells): grid cells are strongly spatially autocorrelated,
so the effective sample size is far below the cell count, the two-sample t-test
becomes wildly overconfident, and every pair is flagged distinguishable up to the
largest K (a degenerate K*). The independent-sample (year) members are essential.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fdr_field_significant(pvals, q=0.05):
    """Benjamini-Hochberg field-significance test (Wilks 2006).

    Returns True if at least one of the M local p-values satisfies the FDR
    threshold p(m) <= q * m / M, i.e. the field is significant at level q.
    """
    p = np.asarray(pvals, float)
    p = p[np.isfinite(p)]
    M = p.size
    if M == 0:
        return False
    ps = np.sort(p)
    thresh = q * (np.arange(1, M + 1) / M)
    return bool(np.any(ps <= thresh))


def pair_distinguishable(Xi, Xj, q=0.05):
    """Two cluster member sets distinguishable by Johnson's t-test + FDR?

    Xi : (n_i, F) members of cluster i (rows = members, cols = field points)
    Xj : (n_j, F) members of cluster j
    Returns True (distinguishable), False (indistinguishable), or None when the
    test cannot be run (a cluster with < 2 members has no within-cluster
    variance estimate).
    """
    if Xi.shape[0] < 2 or Xj.shape[0] < 2:
        return None
    # Welch two-sample t per field point: t = (xbar_i - xbar_j)/sqrt(Si^2/ni + Sj^2/nj)
    with warnings.catch_warnings():
        # constant (zero-variance) field points -> NaN p; harmless, dropped by FDR
        warnings.simplefilter("ignore", RuntimeWarning)
        _, p = stats.ttest_ind(Xi, Xj, axis=0, equal_var=False)
    return fdr_field_significant(p, q=q)


def count_indistinguishable_pairs(features, labels, q=0.05):
    """Number of statistically indistinguishable cluster pairs at this K.

    Untestable pairs (involving a singleton cluster) are counted as
    indistinguishable — they cannot be shown to differ.
    Returns (n_indistinguishable, n_pairs, n_untestable).
    """
    ids = np.unique(labels)
    n_indist = n_pairs = n_untest = 0
    for a in range(len(ids)):
        Xi = features[labels == ids[a]]
        for b in range(a + 1, len(ids)):
            Xj = features[labels == ids[b]]
            n_pairs += 1
            res = pair_distinguishable(Xi, Xj, q=q)
            if res is None:
                n_untest += 1
                n_indist += 1
            elif res is False:
                n_indist += 1
    return n_indist, n_pairs, n_untest


def select_max_distinguishable(k_vals, counts):
    """K* = largest K for which the indistinguishable-pair count is zero for all
    k' <= K, i.e. (first K with count > 0) - 1. If no K has any indistinguishable
    pair, K* is the largest K swept.
    """
    k = list(k_vals)
    bad = [kk for kk, cc in zip(k, counts) if cc is not None and cc > 0]
    if not bad:
        return int(k[-1])
    kstar = min(bad) - 1
    return int(max(kstar, k[0]))


def plot_distinguishability(k_vals, counts, kstar, xlabel, out_png,
                            spread=None, title=None):
    """Johnson (2013) Fig. 1-style plot: number of statistically
    indistinguishable cluster pairs vs K, with K* marked."""
    k_vals = list(k_vals)
    counts = np.asarray(counts, float)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    BLUE, RED = "#1f5fd0", "#e8202a"
    if spread is not None:
        lo, hi = spread
        ax.fill_between(k_vals, lo, hi, color=BLUE, alpha=0.18, lw=0,
                        label="ensemble spread")
    ax.plot(k_vals, counts, "-o", color=BLUE, lw=2.2, ms=4,
            label="# indistinguishable pairs")
    ax.axhline(0, color="0.6", lw=1.0)
    ax.axvline(kstar, color=RED, lw=1.6, ls="--", label=f"K* = {kstar}")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("# statistically indistinguishable pairs", fontsize=12)
    ax.set_xticks(k_vals)
    ax.set_title(title or "Maximum number of distinguishable clusters "
                 "(Johnson 2013: t-test + FDR, q=0.05)",
                 fontsize=12, weight="bold", loc="left")
    ax.legend(fontsize=10, loc="upper left", frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
