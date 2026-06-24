"""Correctness tests for modified_kendall.py.

These tests compare the optimized implementation with direct O(n^2)
formulas and verify the theoretical moments by exact permutation enumeration
for small n.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np

from modified_kendall import (
    censored_rank_null_moments,
    count_agreements,
    count_disagreements,
    f_seqn_agg,
    rank_scores,
    sequential_agreement,
)


def brute_agreements(x: np.ndarray, y: np.ndarray) -> int:
    return int(
        sum(
            x[i] < x[j] and y[i] < y[j]
            for i in range(len(x))
            for j in range(len(x))
            if i != j
        )
    )


def brute_disagreements(x: np.ndarray, y: np.ndarray) -> int:
    return int(
        sum(
            x[i] < x[j] and y[i] > y[j]
            for i in range(len(x))
            for j in range(len(x))
            if i != j
        )
    )


def test_fast_counts_against_brute_force() -> None:
    rng = np.random.default_rng(42)
    for n in range(2, 20):
        for _ in range(20):
            # Include ties and fractional ranks.
            x = rng.choice([1.0, 1.5, 2.0, 3.0, 4.0, 5.0], size=n)
            y = rng.choice([1.0, 1.5, 2.0, 3.0, 4.0, 5.0], size=n)
            assert count_agreements(x, y) == brute_agreements(x, y)
            assert count_disagreements(x, y) == brute_disagreements(x, y)


def test_sequential_against_direct_truncation() -> None:
    rng = np.random.default_rng(123)
    for n in range(4, 20):
        for _ in range(10):
            x = rng.permutation(np.arange(1, n + 1)).astype(float)
            y = rng.permutation(np.arange(1, n + 1)).astype(float)
            result = sequential_agreement(x, y)
            expected = np.array(
                [
                    brute_agreements(np.minimum(x, k), np.minimum(y, k))
                    / (n * (n - 1))
                    for k in range(1, n + 1)
                ]
            )
            np.testing.assert_allclose(result.agreement, expected, atol=0.0, rtol=0.0)


def test_null_moments_by_exact_enumeration() -> None:
    # Fix X=1,...,n and enumerate all relative permutations of Y.
    for n in range(4, 8):
        x = np.arange(1, n + 1, dtype=float)
        scale = n * (n - 1)
        all_y = list(permutations(range(1, n + 1)))
        for k in range(2, n + 1):
            values = np.array(
                [
                    brute_agreements(
                        np.minimum(x, k), np.minimum(np.asarray(y, dtype=float), k)
                    )
                    / scale
                    for y in all_y
                ]
            )
            moments = censored_rank_null_moments(n, k)
            np.testing.assert_allclose(values.mean(), moments.mean, atol=1e-14)
            np.testing.assert_allclose(values.var(), moments.variance, atol=1e-14)


def test_rank_scores_matches_r_style_average_ranks() -> None:
    scores = np.array([4.0, 2.0, 2.0, 1.0])
    np.testing.assert_allclose(
        rank_scores(scores, descending=True, ties="average"),
        np.array([1.0, 2.5, 2.5, 4.0]),
    )


def test_r_compatible_sequence_alias() -> None:
    x = np.array([1.0, 3.0, 2.0, 4.0])
    y = np.array([1.0, 2.0, 4.0, 3.0])
    agreement, mean, variance = f_seqn_agg(x, y)
    result = sequential_agreement(x, y)
    np.testing.assert_array_equal(agreement, result.agreement)
    np.testing.assert_array_equal(mean, result.null_mean)
    np.testing.assert_array_equal(variance, result.null_variance)


if __name__ == "__main__":
    test_fast_counts_against_brute_force()
    test_sequential_against_direct_truncation()
    test_null_moments_by_exact_enumeration()
    test_rank_scores_matches_r_style_average_ranks()
    test_r_compatible_sequence_alias()
    print("All tests passed.")
