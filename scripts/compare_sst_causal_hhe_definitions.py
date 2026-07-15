#!/usr/bin/env python3
"""Compare absolute-HHE and frozen-relative-HHE SST experiment responses."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


EXPERIMENTS = (
    ("persistent_mar_global", "Persistent March"),
    ("persistent_apr_global", "Persistent April"),
    ("persistent_may_global", "Persistent May"),
    ("evolving_may_aug_global", "Evolving May–August"),
    ("evolving_may_aug_tropical_pacific", "Tropical Pacific"),
    ("evolving_may_aug_north_pacific", "North Pacific"),
    ("evolving_may_aug_tropical_atlantic", "Tropical Atlantic"),
    ("evolving_may_aug_north_atlantic", "North Atlantic"),
)

NAVY = "#172936"
RED = "#BE3E4A"
TEAL = "#218C8D"
GRID = "#D8DDE1"
MUTED = "#68747E"


def member_statistics(path: Path) -> tuple[float, float, int, int, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with xr.open_dataset(path) as dataset:
        effects = 100.0 * np.asarray(dataset["regional_effect_by_case"], dtype=float)
    effects = effects[np.isfinite(effects)]
    if effects.size == 0:
        raise ValueError(f"No finite regional_effect_by_case values in {path}")
    return (
        float(np.mean(effects)),
        float(np.median(effects)),
        int(np.count_nonzero(effects > 0.0)),
        int(effects.size),
        effects,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--absolute-dir", type=Path, required=True)
    parser.add_argument("--relative-dir", type=Path, required=True)
    parser.add_argument("--base-year", type=int, default=2000)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for key, label in EXPERIMENTS:
        absolute_path = args.absolute_dir / f"{key}_{args.base_year}.nc"
        relative_path = args.relative_dir / f"{key}_{args.base_year}.nc"
        absolute_mean, absolute_median, absolute_positive, n_absolute, absolute_cases = member_statistics(
            absolute_path
        )
        relative_mean, relative_median, relative_positive, n_relative, relative_cases = member_statistics(
            relative_path
        )
        if n_absolute != n_relative:
            raise ValueError(
                f"Paired-case count differs for {key}: absolute={n_absolute}, relative={n_relative}"
            )
        rows.append(
            {
                "key": key,
                "label": label,
                "absolute_mean_pp": absolute_mean,
                "relative_mean_pp": relative_mean,
                "relative_minus_absolute_mean_pp": relative_mean - absolute_mean,
                "absolute_median_pp": absolute_median,
                "relative_median_pp": relative_median,
                "absolute_positive_members": absolute_positive,
                "relative_positive_members": relative_positive,
                "n_members": n_absolute,
                "member_effect_correlation": float(
                    np.corrcoef(absolute_cases, relative_cases)[0, 1]
                ),
            }
        )

    json_path = args.out_dir / "relative_vs_absolute_hhe_sensitivity.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    csv_path = args.out_dir / "relative_vs_absolute_hhe_sensitivity.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = [row["label"] for row in rows]
    absolute = np.asarray([row["absolute_mean_pp"] for row in rows])
    relative = np.asarray([row["relative_mean_pp"] for row in rows])
    y = np.arange(len(rows))
    height = 0.34
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": NAVY,
            "axes.labelcolor": NAVY,
            "xtick.color": NAVY,
            "ytick.color": NAVY,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(11.8, 6.6), constrained_layout=True)
    absolute_bars = ax.barh(
        y - height / 2,
        absolute,
        height=height,
        color=RED,
        edgecolor="none",
        label="Absolute HHE: HI ≥ 105°F",
        zorder=3,
    )
    relative_bars = ax.barh(
        y + height / 2,
        relative,
        height=height,
        color=TEAL,
        edgecolor="none",
        label="Relative HHE: HI > fixed ACE2 p90",
        zorder=3,
    )
    ax.axvline(0.0, color=NAVY, linewidth=0.9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Regional perturbation minus control (percentage points)")
    ax.set_title(
        "SST treatment effect under absolute and relative HHE definitions",
        loc="left",
        fontsize=15,
        fontweight="semibold",
    )
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    combined = np.r_[absolute, relative]
    span = max(float(np.ptp(combined)), 1.0)
    padding = 0.08 * span
    ax.set_xlim(float(np.min(combined)) - 2.0 * padding, float(np.max(combined)) + 2.8 * padding)

    for bars, values in ((absolute_bars, absolute), (relative_bars, relative)):
        for bar, value in zip(bars, values):
            offset = padding * 0.22 if value >= 0 else -padding * 0.22
            ax.text(
                value + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{value:+.3f}",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=9.2,
            )
    ax.legend(frameon=False, loc="lower right")
    ax.text(
        0.0,
        -0.10,
        "One atmospheric base year; 25 lagged initializations are descriptive, not independent years.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=MUTED,
        fontsize=9.5,
    )
    figure_path = args.out_dir / "relative_vs_absolute_hhe_sensitivity.png"
    fig.savefig(figure_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {figure_path}")


if __name__ == "__main__":
    main()
