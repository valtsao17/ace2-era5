from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import xarray as xr
from tqdm.auto import tqdm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from .io import coord_name, LAT_NAMES, LON_NAMES

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "0.25",
    "axes.labelcolor": "0.15",
    "font.size": 10,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
})

_PLATE = ccrs.PlateCarree()
_WHITE_RED = mcolors.LinearSegmentedColormap.from_list("white_red", ["white", "#67000d"])


def plot_map(ax, field: xr.DataArray, title: str, cmap: str, vmin=None, vmax=None):
    lat = coord_name(field, LAT_NAMES)
    lon = coord_name(field, LON_NAMES)
    if lat is None or lon is None:
        raise KeyError(f"Missing lat/lon for plot. dims={field.dims}, coords={list(field.coords)}")

    lon_vals = field[lon].values
    # pcolormesh with PlateCarree transform handles both 0-360 and -180/180 grids
    mesh = ax.pcolormesh(
        lon_vals, field[lat].values, field.values,
        shading="auto", cmap=cmap, vmin=vmin, vmax=vmax,
        transform=_PLATE,
    )
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, zorder=3)
    ax.set_title(title, fontsize=9)
    return mesh


def save_all(fig: plt.Figure, base_path: Path, formats: list[str]) -> list[Path]:
    out_paths = []
    for fmt in formats:
        out = base_path.with_suffix(f".{fmt}")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        print(f"wrote figure: {out}", flush=True)
        out_paths.append(out)
    return out_paths


def targetdate_diagnostics(
    ds: xr.Dataset,
    title_prefix: str,
    figure_dir: Path,
    stem: str,
    formats: list[str],
    max_map_members: int = 6,
    force: bool = False,
) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    images: list[Path] = []

    pred_prob = ds["predicted_hot_extreme_probability"].load()
    obs = ds["observed_hot_extreme"].load()
    uncertainty = ds["ensemble_uncertainty"].load()
    error = ds["prediction_error_era5_minus_ace2s"].load()

    base = figure_dir / f"{stem}_prob_uncertainty_error"
    expected = [base.with_suffix(f".{fmt}") for fmt in formats]
    if force or any(not p.exists() for p in expected):
        fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True,
                                 subplot_kw=dict(projection=_PLATE))
        panels = [
            (pred_prob, "ACE2S event probability", _WHITE_RED, 0.0, 1.0),
            (obs, "ERA5 event indicator", _WHITE_RED, 0.0, 1.0),
            (uncertainty, "ensemble std dev", "magma", 0.0, None),
            (error, "error: ERA5 - ACE2S prob.", "RdBu_r", -1.0, 1.0),
        ]
        for ax, (field, title, cmap, vmin, vmax) in zip(axes.ravel(), panels):
            mesh = plot_map(ax, field, title, cmap, vmin, vmax)
            fig.colorbar(mesh, ax=ax, shrink=0.78)
        fig.suptitle(title_prefix)
        images += save_all(fig, base, formats)
        plt.close(fig)
    else:
        images += expected

    obs_tmax = ds["observed_daily_tmax_C"].load()
    pred_tmax_m0 = ds["predicted_daily_tmax_C_member0"].load()
    obs_anom = ds["observed_tmax_anomaly_C"].load()
    pred_anom_m0 = ds["predicted_tmax_anomaly_C_member0"].load()

    base = figure_dir / f"{stem}_temperature_fields"
    expected = [base.with_suffix(f".{fmt}") for fmt in formats]
    if force or any(not p.exists() for p in expected):
        tmax_min = float(min(float(obs_tmax.min()), float(pred_tmax_m0.min())))
        tmax_max = float(max(float(obs_tmax.max()), float(pred_tmax_m0.max())))
        anom_abs = float(max(float(abs(obs_anom).max()), float(abs(pred_anom_m0).max())))

        fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True,
                                 subplot_kw=dict(projection=_PLATE))
        temp_panels = [
            (obs_tmax,    "ERA5 observed Tmax (°C)",              "RdYlBu_r", tmax_min,  tmax_max),
            (pred_tmax_m0,"ACE2S Tmax — member 0 (°C)",          "RdYlBu_r", tmax_min,  tmax_max),
            (obs_anom,    "ERA5 anomalies",                       "RdBu_r",   -anom_abs, anom_abs),
            (pred_anom_m0,"ACE2S anomalies — member 0 (°C)",     "RdBu_r",   -anom_abs, anom_abs),
        ]
        for ax, (field, title, cmap, vmin, vmax) in zip(axes.ravel(), temp_panels):
            mesh = plot_map(ax, field, title, cmap, vmin, vmax)
            fig.colorbar(mesh, ax=ax, shrink=0.78)
        fig.suptitle(title_prefix + " | raw temperature")
        images += save_all(fig, base, formats)
        plt.close(fig)
    else:
        images += expected

    obs_tmax_raw = ds["observed_daily_tmax_C"].load()
    base = figure_dir / f"{stem}_summary"
    expected = [base.with_suffix(f".{fmt}") for fmt in formats]
    if force or any(not p.exists() for p in expected):
        fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True,
                                 subplot_kw=dict(projection=_PLATE))
        summary_panels = [
            (obs_tmax_raw, "ERA5 observed Tmax (°C)",     "RdYlBu_r",  None,       None),
            (obs,          "ERA5 event indicator",         _WHITE_RED,   0.0,        1.0),
            (uncertainty,  "ACE2S ensemble std dev",       "magma",      0.0,        None),
            (pred_prob,    "ACE2S event probability",      _WHITE_RED,   0.0,        1.0),
        ]
        binary_legend_handles = [
            mlines.Line2D([], [], linestyle="none", marker="o", markersize=10,
                          markerfacecolor="white", markeredgecolor="0.4", markeredgewidth=0.8,
                          label="no extreme"),
            mlines.Line2D([], [], linestyle="none", marker="o", markersize=10,
                          markerfacecolor="#67000d", markeredgecolor="none",
                          label="heat extreme"),
        ]
        for i, (ax, (field, title, cmap, vmin, vmax)) in enumerate(zip(axes.ravel(), summary_panels)):
            mesh = plot_map(ax, field, title, cmap, vmin, vmax)
            if i == 1:
                ax.legend(handles=binary_legend_handles, loc="lower center", ncol=2,
                          frameon=True, fontsize=9, bbox_to_anchor=(0.5, -0.08),
                          bbox_transform=ax.transAxes)
            else:
                fig.colorbar(mesh, ax=ax, shrink=0.78)
        fig.suptitle(title_prefix)
        images += save_all(fig, base, formats)
        plt.close(fig)
    else:
        images += expected

    ensmean_binary = (ds["predicted_tmax_anomaly_C_ensmean"] > ds["hot_anomaly_threshold_C"]).load().astype("float32")
    obs_binary = ds["observed_hot_extreme"].load()

    base = figure_dir / f"{stem}_ensmean_binary"
    expected = [base.with_suffix(f".{fmt}") for fmt in formats]
    if force or any(not p.exists() for p in expected):
        fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True,
                                 subplot_kw=dict(projection=_PLATE))
        for ax, (field, title) in zip(axes, [
            (obs_binary,     "ERA5 observed extreme (binary)"),
            (ensmean_binary, "ACE2S ens. mean extreme (binary)"),
        ]):
            mesh = plot_map(ax, field, title, _WHITE_RED, 0.0, 1.0)
            fig.colorbar(mesh, ax=ax, shrink=0.78)
        fig.suptitle(title_prefix + " | ensemble mean binary extreme")
        images += save_all(fig, base, formats)
        plt.close(fig)
    else:
        images += expected

    pred_members = ds["predicted_hot_extreme"]
    base = figure_dir / f"{stem}_independent_realizations"
    expected = [base.with_suffix(f".{fmt}") for fmt in formats]
    if force or any(not p.exists() for p in expected):
        n = pred_members.sizes.get("sample", 1)
        selected = np.unique(np.linspace(0, n - 1, min(max_map_members, n), dtype=int))
        fields = [pred_members.isel(sample=int(i)).load() for i in tqdm(selected, desc="load member maps", leave=False)]

        ncols = min(3, len(fields))
        nrows = int(math.ceil(len(fields) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 3.5 * nrows), squeeze=False,
                                 subplot_kw=dict(projection=_PLATE))
        fig.subplots_adjust(bottom=0.08, hspace=0.12, wspace=0.05)

        for ax, idx, field in zip(axes.ravel(), selected, fields):
            plot_map(ax, field, f"member {idx:02d}", _WHITE_RED, 0.0, 1.0)

        for ax in axes.ravel()[len(fields):]:
            ax.set_visible(False)

        legend_handles = [
            mlines.Line2D([], [], linestyle="none", marker="o", markersize=10,
                          markerfacecolor="white", markeredgecolor="0.4", markeredgewidth=0.8,
                          label="no extreme"),
            mlines.Line2D([], [], linestyle="none", marker="o", markersize=10,
                          markerfacecolor="#67000d", markeredgecolor="none",
                          label="heat extreme"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=2,
                   frameon=True, fontsize=10, bbox_to_anchor=(0.5, 0.01),
                   bbox_transform=fig.transFigure)

        fig.suptitle(title_prefix + " | independent members")
        images += save_all(fig, base, formats)
        plt.close(fig)
    else:
        images += expected

    return images


def write_index(figure_dir: Path, images: list[Path]) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "<!doctype html><meta charset='utf-8'><title>ACE2S target-date figures</title>",
        "<style>body{font-family:sans-serif;margin:24px}img{max-width:100%;border:1px solid #ddd;margin-bottom:24px}</style>",
        "<h1>ACE2S target-date hot-extreme figures</h1>",
    ]
    for img in sorted(set(images)):
        rel = img.relative_to(figure_dir)
        if img.suffix.lower() in {".png", ".svg"}:
            lines.append(f"<h2>{rel}</h2><img src='{rel}'>")
        else:
            lines.append(f"<p><a href='{rel}'>{rel}</a></p>")
    out = figure_dir / "index.html"
    out.write_text("\n".join(lines))
    print(f"wrote figure index: {out}", flush=True)
