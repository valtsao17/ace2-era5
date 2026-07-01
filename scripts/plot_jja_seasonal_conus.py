#!/usr/bin/env python3
"""Generate CONUS JJA seasonal skill PNG maps from existing skill_jja_seasonal.nc.

Subsets the global NC file to CONUS bbox and plots with the same style
as seasonal_djf_skill.py (imshow + shapely borders, no cartopy GeoAxes).
"""

from pathlib import Path

import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.io.shapereader as shpreader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NC_PATH      = PROJECT_ROOT / "outputs/lag_may/seasonal_jja_sliding7d/skill_jja_seasonal.nc"
FORCING_DIR  = PROJECT_ROOT / "data/lag_data/forcing_data_ace2era5"
OUT_DIR      = NC_PATH.parent

CONUS_EXTENT = (-175.0, -50.0, 12.0, 77.0)   # lon_min, lon_max, lat_min, lat_max

_SKILL_CMAP = LinearSegmentedColormap.from_list(
    "skill",
    ["#ffffff", "#fff2b0", "#ffcc55", "#ff8800", "#cc2200", "#780000"],
    N=256,
)
_SKILL_CMAP.set_bad("white")

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "font.size": 10, "savefig.dpi": 180, "savefig.bbox": "tight"})

_LAND_GEOMS  = None
_STATE_GEOMS = None


def _get_borders():
    global _LAND_GEOMS, _STATE_GEOMS
    if _LAND_GEOMS is None:
        shp = shpreader.natural_earth(resolution="50m", category="physical", name="land")
        _LAND_GEOMS = list(shpreader.Reader(shp).geometries())
        shp = shpreader.natural_earth(resolution="50m", category="cultural",
                                      name="admin_1_states_provinces_lakes")
        _STATE_GEOMS = list(shpreader.Reader(shp).geometries())
    return _LAND_GEOMS, _STATE_GEOMS


def _draw_borders(ax, xlim, ylim):
    from shapely.geometry import box
    vp = box(xlim[0], ylim[0], xlim[1], ylim[1])
    land_geoms, state_geoms = _get_borders()

    def _plot(geom, color, lw):
        if geom is None or geom.is_empty:
            return
        if hasattr(geom, "geoms"):
            for g in geom.geoms:
                _plot(g, color, lw)
        elif hasattr(geom, "exterior"):
            xs, ys = geom.exterior.xy
            ax.plot(xs, ys, color=color, linewidth=lw, zorder=3)
            for ring in geom.interiors:
                xs, ys = ring.xy
                ax.plot(xs, ys, color=color, linewidth=lw, zorder=3)

    for geom in land_geoms:
        try:
            _plot(geom.intersection(vp), "black", 0.5)
        except Exception:
            continue
    for geom in state_geoms:
        try:
            _plot(geom.intersection(vp), "0.4", 0.3)
        except Exception:
            continue


def load_land_mask(lat, lon_360):
    f = next(FORCING_DIR.glob("forcing_*.nc"), None)
    if f is None:
        return None
    with xr.open_dataset(f) as ds:
        lf    = ds["land_fraction"]
        lat_n = "latitude" if "latitude" in lf.dims else "lat"
        lon_n = "longitude" if "longitude" in lf.dims else "lon"
        lf2   = lf.interp({lat_n: lat, lon_n: lon_360}, method="nearest")
    return lf2.values > 0.5


def cos_lat_mean(field, lat, mask=None):
    w     = np.cos(np.deg2rad(lat))[:, np.newaxis]
    f     = field if mask is None else np.where(mask, field, np.nan)
    valid = np.isfinite(f)
    return float(np.nansum(f * w * valid) / np.nansum(w * valid))


def plot_conus(r, pval, lat_c, lon_c, title, out_path, land_mask_c):
    lon_min, lon_max, lat_min, lat_max = CONUS_EXTENT
    signed = "Kendall" in title

    stat_data = r if signed else np.abs(r)
    plot_data = np.where(land_mask_c, stat_data, np.nan)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_facecolor("white")

    extent = [lon_c.min() - 0.5, lon_c.max() + 0.5,
              lat_c.min() - 0.5, lat_c.max() + 0.5]
    cmap = "RdBu_r" if signed else _SKILL_CMAP
    vmin, vmax = (-1.0, 1.0) if signed else (0.0, 0.6)
    im = ax.imshow(plot_data, extent=extent, origin="lower",
                   cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest", zorder=1)

    _draw_borders(ax, (lon_min - 2, lon_max + 2), (lat_min - 2, lat_max + 2))

    LON2D, LAT2D = np.meshgrid(lon_c, lat_c)
    not_sig = (pval > 0.05) & land_mask_c & np.isfinite(r)
    ax.plot(LON2D[not_sig].ravel(), LAT2D[not_sig].ravel(),
            "k.", markersize=1.5, alpha=0.55, zorder=4)

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    lon_ticks = np.arange(-160, -40, 20)
    lat_ticks = np.arange(20, 80, 10)
    ax.set_xticks(lon_ticks)
    ax.set_xticklabels([f"{abs(x)}°W" for x in lon_ticks], fontsize=8)
    ax.set_yticks(lat_ticks)
    ax.set_yticklabels([f"{y}°N" for y in lat_ticks], fontsize=8)
    ax.grid(True, linewidth=0.3, color="gray", alpha=0.4, linestyle="--")

    full_mean  = cos_lat_mean(stat_data, lat_c, mask=np.isfinite(r))
    land_mean  = cos_lat_mean(stat_data, lat_c, mask=land_mask_c & np.isfinite(r))
    ocean_mean = cos_lat_mean(stat_data, lat_c, mask=(~land_mask_c) & np.isfinite(r))
    txt = f"full={full_mean:.3f}   land={land_mean:.3f}   ocean={ocean_mean:.3f}"
    ax.text(0.98, 0.04, txt, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=3), zorder=5)

    metric_label = "Kendall τ" if "Kendall" in title else "Pearson r"
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal",
                        shrink=0.65, pad=0.07, aspect=30)
    cbar.set_label(metric_label, fontsize=9)
    ticks = np.linspace(-1.0, 1.0, 9) if signed else np.arange(0, 0.61, 0.06)
    cbar.set_ticks(ticks)
    cbar.ax.set_xticklabels([f"{t:.2f}" for t in ticks], fontsize=7)

    ax.set_title(title, fontsize=10, pad=6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote: {out_path}", flush=True)


def main():
    print(f"Loading {NC_PATH}", flush=True)
    ds = xr.open_dataset(NC_PATH)
    lat     = ds["lat"].values          # -89.24 ... 89.24
    lon_360 = ds["lon"].values          # 0.5 ... 359.5
    r_map    = ds["pearson_r"].values
    r_pval   = ds["r_p_value"].values
    tau_map  = ds["kendall_tau"].values
    tau_pval = ds["tau_p_value"].values
    ds.close()

    # Convert lon to -180/180 for CONUS subsetting
    lon_180 = np.where(lon_360 > 180.0, lon_360 - 360.0, lon_360)

    lat_sel = (lat     >= CONUS_EXTENT[2]) & (lat     <= CONUS_EXTENT[3])
    lon_sel = (lon_180 >= CONUS_EXTENT[0]) & (lon_180 <= CONUS_EXTENT[1])

    lat_c   = lat[lat_sel]
    lon_c   = lon_180[lon_sel]

    def _sub(arr):
        return arr[lat_sel, :][:, lon_sel]

    r_c    = _sub(r_map)
    rp_c   = _sub(r_pval)
    tau_c  = _sub(tau_map)
    taup_c = _sub(tau_pval)

    print(f"CONUS grid: {len(lat_c)} lat × {len(lon_c)} lon", flush=True)

    # Land mask (lon in 0-360 for interp, matching the forcing file grid)
    lon_c_360 = lon_c % 360.0
    land_mask_c = load_land_mask(lat_c, lon_c_360)
    if land_mask_c is None:
        print("WARNING: no forcing file found — using all-land mask", flush=True)
        land_mask_c = np.ones((len(lat_c), len(lon_c)), dtype=bool)

    yr_range     = "1980–2016  (n=37 seasons)"
    thresh_label = "±7-day sliding window thresholds"

    print("Loading borders ...", flush=True)
    _get_borders()
    print("  done.", flush=True)

    plot_conus(
        r_c, rp_c, lat_c, lon_c,
        f"ACE2-ERA5  |  JJA heat-extreme frequency skill  |  Pearson r\n"
        f"{yr_range}  |  {thresh_label}",
        OUT_DIR / "pearsonr_jja_seasonal_conus.png",
        land_mask_c,
    )
    plot_conus(
        tau_c, taup_c, lat_c, lon_c,
        f"ACE2-ERA5  |  JJA heat-extreme frequency skill  |  Kendall τ\n"
        f"{yr_range}  |  {thresh_label}",
        OUT_DIR / "tau_jja_seasonal_conus.png",
        land_mask_c,
    )

    print("All done.", flush=True)


if __name__ == "__main__":
    main()
