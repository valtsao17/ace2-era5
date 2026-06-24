"""Modified Kendall rank-order association test (Zheng & Lo, 2006).

This module is a Python implementation of the method in:

    Tian Zheng and Shaw-Hwa Lo,
    "A Modified Kendall Rank-Order Association Test for Evaluating the
    Repeatability of Two Studies with a Large Number of Objects".

It also translates the supplied R helpers ``functions.R`` and
``seqn_agg_func.R``.  The public API contains both descriptive Python names
and R-compatible aliases.

Rank convention
---------------
Ranks are 1-based and smaller ranks are better: rank 1 is the highest-merit
object.  The paper truncates ranks as ``min(rank, k)``.  Consequently,
ranks 1, ..., k-1 remain distinct and ranks k, ..., n are tied at k.  This is
exactly the convention used by the original R code.

The theoretical null moments assume that each original ranking is a
permutation of 1, ..., n (no ties before truncation).  The observed agreement
counts still work with ties, but the analytic mean/variance are then not exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, erfc, floor, isclose, sqrt
from typing import Iterable, Literal, Optional, Sequence, Union
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

Alternative = Literal["greater", "less", "two-sided"]
TieMethod = Literal["average", "ordinal"]


@dataclass(frozen=True)
class NullMoments:
    """Null mean and variance for the truncated-rank agreement statistic.

    ``mean`` and ``variance`` refer to the normalized statistic
    ``agreements / (n * (n - 1))``, matching ``f.theory`` in the R code.
    Count-scale moments are included for convenience.
    """

    n: int
    k: int
    mean: float
    variance: float
    expected_agreements: float
    variance_agreements: float


@dataclass(frozen=True)
class ModifiedKendallResult:
    """Result from :func:`modified_kendall_test`."""

    n: int
    k: int
    agreements: int
    normalized_agreements: float
    null_mean: float
    null_variance: float
    expected_agreements: float
    variance_agreements: float
    z_statistic: float
    normal_pvalue: float
    permutation_pvalue: Optional[float]
    pvalue: float
    pvalue_method: str
    alternative: Alternative


@dataclass(frozen=True)
class SequentialAgreementResult:
    """Agreement statistic and null moments for every truncation value.

    Array position ``k - 1`` corresponds to the paper's truncation value ``k``.
    For ``k=1`` all ranks are tied, so the statistic, mean, and variance are 0;
    the z-score is undefined and returned as ``nan``.
    """

    k: NDArray[np.int64]
    agreement: NDArray[np.float64]
    null_mean: NDArray[np.float64]
    null_variance: NDArray[np.float64]
    z_statistic: NDArray[np.float64]


@dataclass(frozen=True)
class TopOverlapResult:
    """Result for the top-rank overlap test from Section 2.4 of the paper."""

    n: int
    k: int
    overlap: int
    null_mean: float
    null_variance: float
    z_statistic: float
    pvalue: float
    alternative: Alternative


class _FenwickTree:
    """Fenwick tree storing integer frequencies."""

    __slots__ = ("tree",)

    def __init__(self, size: int) -> None:
        self.tree = np.zeros(size + 1, dtype=np.int64)

    def add(self, index: int, value: int = 1) -> None:
        tree = self.tree
        size = len(tree)
        while index < size:
            tree[index] += value
            index += index & -index

    def prefix_sum(self, index: int) -> int:
        total = 0
        tree = self.tree
        while index > 0:
            total += int(tree[index])
            index -= index & -index
        return total


def _as_finite_1d(values: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def _validate_pair(rank_x: ArrayLike, rank_y: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x = _as_finite_1d(rank_x, "rank_x")
    y = _as_finite_1d(rank_y, "rank_y")
    if x.size != y.size:
        raise ValueError("rank_x and rank_y must have the same length")
    return x, y


def _is_rank_permutation(rank: NDArray[np.float64]) -> bool:
    n = rank.size
    return bool(np.array_equal(np.sort(rank), np.arange(1.0, n + 1.0)))


def _warn_if_nonpermutation_ranks(x: NDArray[np.float64], y: NDArray[np.float64]) -> None:
    if not (_is_rank_permutation(x) and _is_rank_permutation(y)):
        warnings.warn(
            "The paper's analytic null mean and variance assume each original "
            "ranking is a permutation of 1,...,n with no ties. The observed "
            "statistic is valid, but the analytic p-value may not be exact; "
            "consider a permutation p-value.",
            RuntimeWarning,
            stacklevel=3,
        )


def rank_scores(
    scores: ArrayLike,
    *,
    descending: bool = True,
    ties: TieMethod = "average",
) -> NDArray[np.float64]:
    """Convert scores to 1-based ranks.

    Parameters
    ----------
    scores:
        Numeric scores for the same objects.
    descending:
        When true (the paper's convention), larger scores receive smaller ranks.
    ties:
        ``"average"`` matches R's default ``rank()`` behavior. ``"ordinal"``
        breaks ties by stable input order and always returns a permutation.
    """

    values = _as_finite_1d(scores, "scores")
    n = values.size
    key = -values if descending else values
    order = np.argsort(key, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)

    if ties == "ordinal":
        ranks[order] = np.arange(1.0, n + 1.0)
        return ranks
    if ties != "average":
        raise ValueError("ties must be 'average' or 'ordinal'")

    sorted_values = values[order]
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_values[end] == sorted_values[start]:
            end += 1
        # Positions are start+1, ..., end in 1-based indexing.
        ranks[order[start:end]] = ((start + 1) + end) / 2.0
        start = end
    return ranks


def safe_choose(n: int, p: int) -> int:
    """Return ``n choose p`` when ``n >= p >= 0``, otherwise 0.

    This is the direct Python counterpart of ``f.choose``.
    """

    n = int(n)
    p = int(p)
    if p < 0 or n < p:
        return 0
    return comb(n, p)


def _strict_greater_dominance_counts(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.int64]:
    """For each i, count j such that x[j] > x[i] and y[j] > y[i].

    Equal x or y values do not count.  The implementation is O(n log n) and
    uses grouped updates so that equality in x is handled strictly.
    """

    n = x.size
    _, y_inverse = np.unique(y, return_inverse=True)
    # Descending x: previously inserted points have strictly larger x. Equal-x
    # groups are queried before any member of the group is inserted.
    order = np.argsort(x, kind="mergesort")[::-1]
    tree = _FenwickTree(int(y_inverse.max()) + 1)
    counts = np.zeros(n, dtype=np.int64)
    seen = 0
    pos = 0

    while pos < n:
        end = pos + 1
        x_value = x[order[pos]]
        while end < n and x[order[end]] == x_value:
            end += 1

        for ordered_pos in range(pos, end):
            idx = int(order[ordered_pos])
            y_index = int(y_inverse[idx]) + 1
            # Number already seen with y strictly greater than y[i].
            counts[idx] = seen - tree.prefix_sum(y_index)

        for ordered_pos in range(pos, end):
            idx = int(order[ordered_pos])
            tree.add(int(y_inverse[idx]) + 1)
            seen += 1

        pos = end

    return counts


def count_agreements(rank_x: ArrayLike, rank_y: ArrayLike) -> int:
    """Count Kendall agreements, matching ``f.agg``.

    Counts ordered pairs ``(i, j)`` satisfying
    ``rank_x[i] < rank_x[j]`` and ``rank_y[i] < rank_y[j]``.
    Each concordant unordered pair is therefore counted once when there are no
    ties.  Runtime is O(n log n), faster than the O(n^2) R loop.
    """

    x, y = _validate_pair(rank_x, rank_y)
    return int(_strict_greater_dominance_counts(x, y).sum(dtype=np.int64))


def count_disagreements(rank_x: ArrayLike, rank_y: ArrayLike) -> int:
    """Count Kendall disagreements, matching ``f.disagg``."""

    x, y = _validate_pair(rank_x, rank_y)
    # y_i > y_j is equivalent to -y_i < -y_j.
    return int(_strict_greater_dominance_counts(x, -y).sum(dtype=np.int64))


def tie_count(rank: ArrayLike, *, round_values: bool = True) -> int:
    """Return the oriented tie-pair count, matching ``f.tie``.

    For a tied group of size m, the contribution is m(m-1), not C(m, 2).
    By default values are rounded first, exactly as in the supplied R helper.
    """

    values = _as_finite_1d(rank, "rank")
    if round_values:
        values = np.rint(values)
    _, counts = np.unique(values, return_counts=True)
    return int(np.sum(counts * (counts - 1), dtype=np.int64))


def _choose_small(n: int, p: int) -> float:
    """Fast C(n,p) for p <= 4, returning 0 when n < p."""

    if n < p or p < 0:
        return 0.0
    if p == 0:
        return 1.0
    if p == 1:
        return float(n)
    if p == 2:
        return float(n * (n - 1) // 2)
    if p == 3:
        return float(n * (n - 1) * (n - 2) // 6)
    if p == 4:
        return float(n * (n - 1) * (n - 2) * (n - 3) // 24)
    return float(comb(n, p))


def censored_rank_null_moments(n: int, k: int) -> NullMoments:
    """Analytic null moments for the paper's truncated agreement statistic.

    This is the direct translation of ``f.theory(nn, pp)``.  The statistic is

        A_k / (n * (n - 1)),

    where ``A_k`` is the agreement count after replacing each rank r by
    ``min(r, k)``.

    Parameters
    ----------
    n:
        Number of objects. The closed-form variance requires n >= 4.
    k:
        Truncation value in 2, ..., n.
    """

    n = int(n)
    k = int(k)
    if n < 4:
        raise ValueError("the closed-form variance requires n >= 4")
    if not 2 <= k <= n:
        raise ValueError("k must satisfy 2 <= k <= n")

    n_float = float(n)
    denominator = n_float * (n_float - 1.0)
    q = 1.0 - ((n - k + 1) * (n - k)) / denominator
    mean = 0.25 * q * q

    c_n4 = _choose_small(n, 4)
    c_n3 = _choose_small(n, 3)

    b = (
        0.25 * _choose_small(k - 1, 4) / c_n4
        + 0.25
        * _choose_small(k - 1, 3)
        * _choose_small(n - k + 1, 1)
        / c_n4
        + (1.0 / 6.0)
        * _choose_small(k - 1, 2)
        * _choose_small(n - k + 1, 2)
        / c_n4
    )

    # Algebraically equivalent to the paper's
    # [C(k,3) + C(k-1,2) C(n-k,1)] / C(n,3).
    c = (
        _choose_small(k - 1, 3)
        + _choose_small(k - 1, 2) * (n - k + 1)
    ) / c_n3

    d = 1.0 - _choose_small(n - k + 1, 3) / c_n3

    variance = (
        0.25 * q * q
        + (n - 2) * (n - 3) * b * b
        + (n - 2) * (1.0 / 6.0) * c * c
        + (n - 2) * (1.0 / 9.0) * d * d
        - (n * n - n) * (1.0 / 16.0) * q**4
    ) / denominator

    # Guard against tiny negative values from floating-point cancellation.
    if variance < 0.0 and isclose(variance, 0.0, abs_tol=1e-15):
        variance = 0.0
    if variance <= 0.0:
        raise ArithmeticError(
            f"computed a non-positive null variance ({variance}) for n={n}, k={k}"
        )

    expected_count = mean * denominator
    variance_count = variance * denominator * denominator
    return NullMoments(
        n=n,
        k=k,
        mean=mean,
        variance=variance,
        expected_agreements=expected_count,
        variance_agreements=variance_count,
    )


def truncate_ranks(rank: ArrayLike, k: int) -> NDArray[np.float64]:
    """Apply the paper's truncation ``min(rank, k)``."""

    values = _as_finite_1d(rank, "rank")
    k = int(k)
    if k < 1:
        raise ValueError("k must be at least 1")
    return np.minimum(values, float(k))


def _normal_pvalue(z: float, alternative: Alternative) -> float:
    if alternative == "greater":
        return 0.5 * erfc(z / sqrt(2.0))
    if alternative == "less":
        return 0.5 * erfc(-z / sqrt(2.0))
    if alternative == "two-sided":
        return min(1.0, erfc(abs(z) / sqrt(2.0)))
    raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")


def _is_extreme(value: int, observed: int, alternative: Alternative) -> bool:
    if alternative == "greater":
        return value >= observed
    if alternative == "less":
        return value <= observed
    if alternative == "two-sided":
        # For the permutation distribution, distance from its null mean is the
        # appropriate two-sided ordering. This branch is handled separately.
        raise RuntimeError("two-sided comparison needs the permutation center")
    raise ValueError("invalid alternative")


def modified_kendall_test(
    rank_x: ArrayLike,
    rank_y: ArrayLike,
    k: int,
    *,
    alternative: Alternative = "greater",
    permutations: int = 0,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    validate_ranks: bool = True,
) -> ModifiedKendallResult:
    """Run the modified Kendall rank-order association test.

    The default alternative is positive association among the top-ranked
    objects, matching the one-sided test used in the paper.  Set
    ``permutations`` to a positive integer to obtain a Monte Carlo permutation
    p-value; that value becomes the returned primary ``pvalue``.
    """

    x, y = _validate_pair(rank_x, rank_y)
    n = x.size
    k = int(k)
    if n < 4:
        raise ValueError("at least 4 objects are required for the analytic test")
    if not 2 <= k <= n:
        raise ValueError("k must satisfy 2 <= k <= n")
    if permutations < 0:
        raise ValueError("permutations must be non-negative")
    if alternative not in ("greater", "less", "two-sided"):
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")

    if validate_ranks:
        _warn_if_nonpermutation_ranks(x, y)

    if k < 10 or n <= 30:
        warnings.warn(
            "The paper notes that the normal approximation can be discrete "
            "when n and k are small; a permutation p-value is preferable in "
            "that setting.",
            RuntimeWarning,
            stacklevel=2,
        )

    x_c = np.minimum(x, float(k))
    y_c = np.minimum(y, float(k))
    agreements = count_agreements(x_c, y_c)
    scale = float(n * (n - 1))
    normalized = agreements / scale
    moments = censored_rank_null_moments(n, k)
    z = (normalized - moments.mean) / sqrt(moments.variance)
    normal_p = _normal_pvalue(z, alternative)

    permutation_p: Optional[float] = None
    if permutations:
        if isinstance(random_state, np.random.Generator):
            rng = random_state
        else:
            rng = np.random.default_rng(random_state)

        extreme = 0
        if alternative == "two-sided":
            center = moments.expected_agreements
            observed_distance = abs(agreements - center)
            for _ in range(int(permutations)):
                permuted_y = rng.permutation(y)
                permuted_count = count_agreements(
                    x_c, np.minimum(permuted_y, float(k))
                )
                if abs(permuted_count - center) >= observed_distance:
                    extreme += 1
        else:
            for _ in range(int(permutations)):
                permuted_y = rng.permutation(y)
                permuted_count = count_agreements(
                    x_c, np.minimum(permuted_y, float(k))
                )
                if _is_extreme(permuted_count, agreements, alternative):
                    extreme += 1

        # Phipson-Smyth style +1 correction for a Monte Carlo permutation test.
        permutation_p = (extreme + 1.0) / (int(permutations) + 1.0)

    if permutation_p is None:
        pvalue = normal_p
        method = "normal approximation"
    else:
        pvalue = permutation_p
        method = f"Monte Carlo permutation ({int(permutations)} permutations)"

    return ModifiedKendallResult(
        n=n,
        k=k,
        agreements=agreements,
        normalized_agreements=normalized,
        null_mean=moments.mean,
        null_variance=moments.variance,
        expected_agreements=moments.expected_agreements,
        variance_agreements=moments.variance_agreements,
        z_statistic=z,
        normal_pvalue=normal_p,
        permutation_pvalue=permutation_p,
        pvalue=pvalue,
        pvalue_method=method,
        alternative=alternative,
    )


def modified_kendall_from_scores(
    scores_x: ArrayLike,
    scores_y: ArrayLike,
    k: int,
    *,
    descending: bool = True,
    ties: TieMethod = "average",
    **test_kwargs: object,
) -> ModifiedKendallResult:
    """Rank two score vectors, then run :func:`modified_kendall_test`."""

    rank_x = rank_scores(scores_x, descending=descending, ties=ties)
    rank_y = rank_scores(scores_y, descending=descending, ties=ties)
    return modified_kendall_test(rank_x, rank_y, k, **test_kwargs)


def sequential_agreement(
    rank_x: ArrayLike,
    rank_y: ArrayLike,
    *,
    validate_ranks: bool = True,
) -> SequentialAgreementResult:
    """Compute the modified statistic for every k=1,...,n.

    This reproduces ``f.seqn.agg`` but uses an O(n log n) identity rather than
    the original O(n^2) incremental loops:

        A_k = sum_i 1(rank_x[i] < k, rank_y[i] < k)
                    * #{j: rank_x[j] > rank_x[i], rank_y[j] > rank_y[i]}.

    Thus each object contributes its full dominance count once it lies above
    the truncation boundary in both rankings.
    """

    x, y = _validate_pair(rank_x, rank_y)
    n = x.size
    if n < 4:
        raise ValueError("at least 4 objects are required for null variances")
    if validate_ranks:
        _warn_if_nonpermutation_ranks(x, y)
    if np.any(x < 1.0) or np.any(y < 1.0) or np.any(x > n) or np.any(y > n):
        raise ValueError("rank values must lie between 1 and n")

    dominance = _strict_greater_dominance_counts(x, y)

    # Smallest integer k satisfying x_i < k and y_i < k.
    activation = np.floor(np.maximum(x, y)).astype(np.int64) + 1
    increments = np.zeros(n + 2, dtype=np.int64)
    active_mask = (activation >= 1) & (activation <= n)
    np.add.at(increments, activation[active_mask], dominance[active_mask])
    cumulative_counts = np.cumsum(increments, dtype=np.int64)

    scale = float(n * (n - 1))
    agreement = cumulative_counts[1 : n + 1].astype(np.float64) / scale

    null_mean = np.zeros(n, dtype=np.float64)
    null_variance = np.zeros(n, dtype=np.float64)
    for k in range(2, n + 1):
        moments = censored_rank_null_moments(n, k)
        null_mean[k - 1] = moments.mean
        null_variance[k - 1] = moments.variance

    z = np.full(n, np.nan, dtype=np.float64)
    positive_variance = null_variance > 0.0
    z[positive_variance] = (
        agreement[positive_variance] - null_mean[positive_variance]
    ) / np.sqrt(null_variance[positive_variance])

    return SequentialAgreementResult(
        k=np.arange(1, n + 1, dtype=np.int64),
        agreement=agreement,
        null_mean=null_mean,
        null_variance=null_variance,
        z_statistic=z,
    )


def top_overlap_test(
    rank_x: ArrayLike,
    rank_y: ArrayLike,
    k: int,
    *,
    alternative: Alternative = "greater",
) -> TopOverlapResult:
    """Top-k overlap z-test from Section 2.4 of the paper."""

    x, y = _validate_pair(rank_x, rank_y)
    n = x.size
    k = int(k)
    if n < 2:
        raise ValueError("at least 2 objects are required")
    if not 1 <= k <= n:
        raise ValueError("k must satisfy 1 <= k <= n")

    overlap = int(np.count_nonzero((x <= k) & (y <= k)))
    mean = (k * k) / n
    variance = (
        (k * k) / n
        + (k * k * (k - 1) * (k - 1)) / (n * (n - 1))
        - (k**4) / (n * n)
    )
    if variance <= 0.0:
        raise ValueError("the overlap variance is zero at this k")
    z = (overlap - mean) / sqrt(variance)
    pvalue = _normal_pvalue(z, alternative)
    return TopOverlapResult(
        n=n,
        k=k,
        overlap=overlap,
        null_mean=mean,
        null_variance=variance,
        z_statistic=z,
        pvalue=pvalue,
        alternative=alternative,
    )


def kendall_tau_a(rank_x: ArrayLike, rank_y: ArrayLike) -> float:
    """Classic Kendall tau-a from agreement and disagreement counts."""

    x, y = _validate_pair(rank_x, rank_y)
    n = x.size
    if n < 2:
        raise ValueError("at least 2 objects are required")
    agreements = count_agreements(x, y)
    disagreements = count_disagreements(x, y)
    return (agreements - disagreements) / (n * (n - 1) / 2.0)


# ---------------------------------------------------------------------------
# R-compatible aliases
# ---------------------------------------------------------------------------

def f_agg(rank_x: ArrayLike, rank_y: ArrayLike) -> int:
    return count_agreements(rank_x, rank_y)


def f_choose(n: int, p: int) -> int:
    return safe_choose(n, p)


def f_theory(nn: int, pp: int) -> dict[str, float]:
    moments = censored_rank_null_moments(nn, pp)
    return {"mean": moments.mean, "var": moments.variance}


def f_disagg(rank_x: ArrayLike, rank_y: ArrayLike) -> int:
    return count_disagreements(rank_x, rank_y)


def f_tie(rank_x: ArrayLike) -> int:
    return tie_count(rank_x, round_values=True)


def f_seqn_agg(
    rank_x: ArrayLike, rank_y: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    result = sequential_agreement(rank_x, rank_y)
    return result.agreement, result.null_mean, result.null_variance


__all__ = [
    "NullMoments",
    "ModifiedKendallResult",
    "SequentialAgreementResult",
    "TopOverlapResult",
    "rank_scores",
    "safe_choose",
    "count_agreements",
    "count_disagreements",
    "tie_count",
    "censored_rank_null_moments",
    "truncate_ranks",
    "modified_kendall_test",
    "modified_kendall_from_scores",
    "sequential_agreement",
    "top_overlap_test",
    "kendall_tau_a",
    "f_agg",
    "f_choose",
    "f_theory",
    "f_disagg",
    "f_tie",
    "f_seqn_agg",
]
