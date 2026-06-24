"""Reproduce the supplied R simulation in Python.

Run:
    python example_modified_kendall.py

The script writes ``modified_kendall_simulation.png`` in the current folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from modified_kendall import rank_scores, sequential_agreement


def run_simulation(
    n: int = 1000,
    repetitions: int = 20,
    k0: int = 30,
    delta: float = 1.0,
    sigma: float = 1.0,
    seed: int = 20260619,
) -> np.ndarray:
    """Return a matrix whose columns are sequential z-statistic curves."""

    if not 1 <= k0 <= n:
        raise ValueError("k0 must satisfy 1 <= k0 <= n")

    rng = np.random.default_rng(seed)
    alpha = np.concatenate(
        [delta * np.arange(k0, 0, -1, dtype=float) / k0, np.zeros(n - k0)]
    )

    z_curves = np.empty((n, repetitions), dtype=float)
    for b in range(repetitions):
        score_x = alpha + rng.normal(0.0, sigma, n)
        score_y = alpha + rng.normal(0.0, sigma, n)
        rank_x = rank_scores(score_x, descending=True, ties="average")
        rank_y = rank_scores(score_y, descending=True, ties="average")
        result = sequential_agreement(rank_x, rank_y)
        z_curves[:, b] = result.z_statistic

    # The R script explicitly sets the first row to zero because k=1 has
    # zero variance and hence an undefined z-statistic.
    z_curves[0, :] = 0.0
    return z_curves


def plot_simulation(z_curves: np.ndarray, k0: int, output: Path) -> None:
    n = z_curves.shape[0]
    k = np.arange(1, n + 1)
    q025, q25, median, q75, q975 = np.nanquantile(
        z_curves, [0.025, 0.25, 0.5, 0.75, 0.975], axis=1
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.vlines(k, q025, q975, linewidth=1, alpha=0.45, label="2.5%-97.5%")
    ax.vlines(k, q25, q75, linewidth=2, alpha=0.8, label="25%-75%")
    ax.scatter(k, median, s=7, label="median")
    ax.axhline(0.0, linewidth=1)
    ax.axvline(k0, linewidth=1, linestyle="--", label=f"k0={k0}")
    ax.set_xlabel("truncation value k")
    ax.set_ylabel("standardized statistic z")
    ax.set_title("Modified Kendall sequential statistic")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--k0", type=int, default=30)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument(
        "--output", type=Path, default=Path("modified_kendall_simulation.png")
    )
    args = parser.parse_args()

    z_curves = run_simulation(
        n=args.n,
        repetitions=args.repetitions,
        k0=args.k0,
        delta=args.delta,
        sigma=args.sigma,
        seed=args.seed,
    )
    plot_simulation(z_curves, args.k0, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
