#!/usr/bin/env python3
"""Robustness of the HHE seasonal-frequency skill to *non-consecutive* day sampling.

Motivation: the JJA HHE definition uses a ±7-day LOO sliding-window
threshold, so neighbouring days are strongly autocorrelated — a hot spell paints
a run of consecutive "extreme" days that are not independent samples. If the
ACE2-vs-ERA5 rank-correlation skill (Kendall τ on the 37-year seasonal
frequency) is real, it should survive when we estimate the seasonal frequency
from only a *sparse, non-consecutive* subset of JJA days instead of all 92.

Method
------
1. Load the full day-level arrays and the ±7d LOO thresholds exactly as
   seasonal_jja_skill.py does, and reduce to per-year per-day exceedance:
       era5_exc[i]     = (ERA5_day > thr)                         (92,lat,lon) bool
       ace2_dayfrac[i] = mean_members(ACE2_day > thr)             (92,lat,lon) float
   (Heavy float arrays are freed immediately after this reduction.)
2. For each sampling scheme S (a subset of the 92 day positions):
       era5_freq[i] = mean over S of era5_exc[i]
       ace2_freq[i] = mean over S of ace2_dayfrac[i]
   then Kendall τ across the 37 years at every grid cell.
3. Schemes: the full 92 days (reference), fixed strides s∈{2,3,5,8} (every s-th
   day → guaranteed non-consecutive, min gap = s), and a random non-consecutive
   sample (min gap ≥ 2) at several seeds for each density. Stride 8 exceeds the
   ±7-day window, so those days are effectively independent.

Outputs → outputs/lag_may/sparse_sampling_robustness/
  tau_vs_sampling.png         domain-mean τ vs #days, full as reference line
  tau_maps_full_vs_sparse.png [full τ | stride-8 τ | difference] CONUS
  sparse_sampling_summary.json
"""
from __future__ import annotations

import sys
import gc
import json
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from seasonal_jja_skill import (
    YEARS, COMBINED_DIR,
    _load_all_era5_jja, _load_all_ace2_jja, compute_daywise_thresholds_loo,
    kendall_tau_map, cos_lat_mean,
)
from cluster_skill_analysis_sliding7d import (
    _plain_map_axes, _TAU_CMAP, CONUS_LAT_SLICE, CONUS_LON_SLICE,
)

OUT_DIR = PROJECT_ROOT / "outputs/lag_may/sparse_sampling_robustness"
N_DAYS = 92


def random_noncons(n_keep, seed, n=N_DAYS, min_gap=2):
    """A length-n_keep sorted subset of range(n) with consecutive gaps ≥ min_gap."""
    rng = np.random.default_rng(seed)
    for _ in range(200):
        pick = np.sort(rng.choice(n, size=n_keep, replace=False))
        if np.all(np.diff(pick) >= min_gap):
            return pick
    # fall back: greedy thinning of a random permutation
    chosen = []
    for d in rng.permutation(n):
        if all(abs(d - c) >= min_gap for c in chosen):
            chosen.append(d)
        if len(chosen) == n_keep:
            break
    return np.sort(chosen)


def conus_mean(field, lat):
    return cos_lat_mean(field[CONUS_LAT_SLICE, CONUS_LON_SLICE], lat[CONUS_LAT_SLICE])


def tau_for_days(era5_exc, ace2_dayfrac, days):
    era5_freq = era5_exc[:, days].mean(axis=1)          # (n_years,lat,lon)
    ace2_freq = ace2_dayfrac[:, days].mean(axis=1)
    tau_map, _ = kendall_tau_map(ace2_freq.astype(np.float32),
                                 era5_freq.astype(np.float32))
    return tau_map


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(COMBINED_DIR / f"tmax_jja_{YEARS[0]}.nc") as ds:
        lat = ds["lat"].values
        lon = ds["lon"].values
    nlat, nlon = len(lat), len(lon)
    print(f"Grid {nlat}x{nlon}", flush=True)

    print("Loading ERA5 JJA ...", flush=True)
    era5_all = _load_all_era5_jja(nlat, nlon)
    print("Loading ACE2 JJA ...", flush=True)
    ace2_all = _load_all_ace2_jja(nlat, nlon)
    print("Computing ±7d LOO thresholds ...", flush=True)
    era5_thr = compute_daywise_thresholds_loo(era5_all)
    ace2_thr = compute_daywise_thresholds_loo(ace2_all)

    # reduce to per-day exceedance, then free the heavy float arrays
    print("Reducing to per-day exceedance ...", flush=True)
    era5_exc = np.empty((len(YEARS), N_DAYS, nlat, nlon), dtype=np.float32)
    ace2_dayfrac = np.empty((len(YEARS), N_DAYS, nlat, nlon), dtype=np.float32)
    for i in range(len(YEARS)):
        era5_exc[i] = (era5_all[i] > era5_thr[i]).astype(np.float32)
        ace2_dayfrac[i] = (ace2_all[i] > ace2_thr[i][np.newaxis]).mean(axis=0).astype(np.float32)
    del era5_all, ace2_all, era5_thr, ace2_thr
    gc.collect()
    print(f"  era5_exc {era5_exc.shape}  ace2_dayfrac {ace2_dayfrac.shape}", flush=True)

    all_days = np.arange(N_DAYS)

    # ── reference: all 92 days ──
    print("τ — full 92 days ...", flush=True)
    tau_full = tau_for_days(era5_exc, ace2_dayfrac, all_days)
    glob_full  = cos_lat_mean(tau_full, lat)
    conus_full = conus_mean(tau_full, lat)
    print(f"  full: global τ={glob_full:.4f}  CONUS τ={conus_full:.4f}", flush=True)

    schemes = []  # (label, n_days, global_tau, conus_tau, kind)
    schemes.append(dict(label="full (92)", n_days=N_DAYS,
                        global_tau=glob_full, conus_tau=conus_full, kind="full"))

    # ── fixed strides (deterministic, non-consecutive) ──
    tau_stride8 = None
    for s in (2, 3, 5, 8):
        days = all_days[::s]
        tau = tau_for_days(era5_exc, ace2_dayfrac, days)
        g, c = cos_lat_mean(tau, lat), conus_mean(tau, lat)
        schemes.append(dict(label=f"stride {s}", n_days=len(days),
                            global_tau=g, conus_tau=c, kind="stride", stride=s))
        print(f"  stride {s} ({len(days)} d): global τ={g:.4f}  CONUS τ={c:.4f}", flush=True)
        if s == 8:
            tau_stride8 = tau

    # ── random non-consecutive, several seeds per density ──
    for n_keep in (46, 30, 18, 11):   # ≈ stride 2,3,5,8 densities
        gs, cs = [], []
        for seed in range(8):
            days = random_noncons(n_keep, seed)
            tau = tau_for_days(era5_exc, ace2_dayfrac, days)
            gs.append(cos_lat_mean(tau, lat)); cs.append(conus_mean(tau, lat))
        schemes.append(dict(label=f"random ×8 ({n_keep})", n_days=n_keep,
                            global_tau=float(np.mean(gs)), global_std=float(np.std(gs)),
                            conus_tau=float(np.mean(cs)), conus_std=float(np.std(cs)),
                            kind="random"))
        print(f"  random {n_keep}d (8 seeds): global τ={np.mean(gs):.4f}±{np.std(gs):.4f}  "
              f"CONUS τ={np.mean(cs):.4f}±{np.std(cs):.4f}", flush=True)

    # ── curve: τ vs #days ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, std_key, mfull, ttl in (
        (axes[0], "global_tau", "global_std", glob_full, "Global"),
        (axes[1], "conus_tau", "conus_std", conus_full, "CONUS"),
    ):
        strd = [s for s in schemes if s["kind"] == "stride"]
        rnd  = [s for s in schemes if s["kind"] == "random"]
        ax.plot([s["n_days"] for s in strd], [s[key] for s in strd],
                "o-", color="#1f77b4", label="fixed stride (non-consec.)")
        ax.errorbar([s["n_days"] for s in rnd], [s[key] for s in rnd],
                    yerr=[s.get(std_key, 0) for s in rnd], fmt="s--", color="#ff7f0e",
                    capsize=3, label="random non-consec. (8 seeds)")
        ax.axhline(mfull, color="0.4", ls=":", lw=1.4, label=f"full 92 days = {mfull:.3f}")
        ax.set_xlabel("number of JJA days sampled")
        ax.set_ylabel("domain-mean Kendall τ")
        ax.set_title(f"{ttl}: HHE skill vs sampling density")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Robustness of ACE2 HHE rank-correlation skill to non-consecutive day sampling  |  JJA 1980–2016",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    out_curve = OUT_DIR / "tau_vs_sampling.png"
    fig.savefig(out_curve, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_curve}", flush=True)

    # ── maps: full vs stride-8 (independent days) vs diff, CONUS ──
    clat = lat[CONUS_LAT_SLICE]; clon = lon[CONUS_LON_SLICE]
    tf = tau_full[CONUS_LAT_SLICE, CONUS_LON_SLICE]
    ts = tau_stride8[CONUS_LAT_SLICE, CONUS_LON_SLICE]
    td = ts - tf
    tlim = float(np.nanpercentile(np.abs(np.r_[tf[np.isfinite(tf)], ts[np.isfinite(ts)]]), 98))
    dlim = max(float(np.nanpercentile(np.abs(td[np.isfinite(td)]), 98)), 1e-3)
    lon_plot = clon - 360.0 if float(clon.mean()) > 180 else clon
    ext = [float(lon_plot[0]) - 0.5, float(lon_plot[-1]) + 0.5,
           float(clat[0]) - 0.5, float(clat[-1]) + 0.5]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    for ax, fld, ttl, vlim, cb in (
        (axes[0], tf, f"Full 92 days  (τ̄={conus_full:.3f})", tlim, "τ"),
        (axes[1], ts, f"Stride 8, independent  (τ̄={conus_mean(tau_stride8, lat):.3f})", tlim, "τ"),
        (axes[2], td, "stride-8 − full", dlim, "Δτ"),
    ):
        im = ax.imshow(fld, origin="lower", extent=ext, aspect="equal",
                       vmin=-vlim, vmax=vlim, cmap=_TAU_CMAP, interpolation="nearest", zorder=1)
        _plain_map_axes(ax, clon, clat, pad=0.0)
        ax.set_title(ttl, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label=cb)
    fig.suptitle("CONUS HHE Kendall τ — full vs sparse independent sampling", fontsize=12, y=1.01)
    fig.tight_layout()
    out_maps = OUT_DIR / "tau_maps_full_vs_sparse.png"
    fig.savefig(out_maps, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_maps}", flush=True)

    summary = {
        "definition": "Kendall τ on 37-yr ACE2 vs ERA5 JJA HHE seasonal frequency, "
                      "frequency estimated from a non-consecutive subset of the 92 JJA days",
        "full_global_tau": round(glob_full, 4),
        "full_conus_tau": round(conus_full, 4),
        "schemes": [{k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in s.items()} for s in schemes],
    }
    (OUT_DIR / "sparse_sampling_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT_DIR / 'sparse_sampling_summary.json'}", flush=True)
    print("All done.", flush=True)


if __name__ == "__main__":
    main()
