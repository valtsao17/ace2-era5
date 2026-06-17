from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import PipelineConfig
from .paths import ensure_layout
from .planning import build_cases, cases_to_frame
from .extremes import load_prediction_daily_tmax, compute_target_date_extremes
from .era5 import observed_target_date, sliding_window_baseline_threshold
from .figures import targetdate_diagnostics
from .synthetic import make_synthetic_prediction, make_synthetic_era5_target_date, make_synthetic_baseline_threshold


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("hiro-ace-pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name in ["plan", "postprocess", "sanity"]:
        sp = sub.add_parser(name)
        sp.add_argument("--config", required=True)
        sp.add_argument("--force", action="store_true")
        sp.add_argument("--force-figures", action="store_true")

    return p.parse_args()


def compute_case_metrics(ds, case) -> dict:
    prob = ds["predicted_hot_extreme_probability"].values.ravel().astype(float)
    obs  = ds["observed_hot_extreme"].values.ravel().astype(float)
    pred_tmax = ds["predicted_daily_tmax_C_ensmean"].values.ravel().astype(float)
    obs_tmax  = ds["observed_daily_tmax_C"].values.ravel().astype(float)

    mask = np.isfinite(prob) & np.isfinite(obs) & np.isfinite(pred_tmax) & np.isfinite(obs_tmax)
    prob, obs, pred_tmax, obs_tmax = prob[mask], obs[mask], pred_tmax[mask], obs_tmax[mask]

    brier = float(np.mean((prob - obs) ** 2))
    frac_obs = float(np.mean(obs))
    mean_prob = float(np.mean(prob))
    corr = float(np.corrcoef(prob, obs)[0, 1]) if obs.std() > 0 else float("nan")
    tmax_bias = float(np.mean(pred_tmax - obs_tmax))
    tmax_rmse = float(np.sqrt(np.mean((pred_tmax - obs_tmax) ** 2)))

    # Brier skill score vs. climatological forecast (frac_obs everywhere)
    brier_ref = float(np.mean((frac_obs - obs) ** 2))
    bss = 1.0 - brier / brier_ref if brier_ref > 0 else float("nan")

    return {
        "target_date": str(case.target_date.date()),
        "lead_days": case.lead_days,
        "init_date": str(case.init_time.date()),
        "frac_obs_extreme": round(frac_obs, 4),
        "mean_pred_prob": round(mean_prob, 4),
        "brier_score": round(brier, 4),
        "brier_skill_score": round(bss, 4),
        "spatial_corr_prob_obs": round(corr, 4),
        "pred_tmax_bias_C": round(tmax_bias, 3),
        "pred_tmax_rmse_C": round(tmax_rmse, 3),
    }


def write_plan(cfg: PipelineConfig, paths: dict[str, Path]):
    cases = build_cases(cfg.target_dates, cfg.lead_days, cfg.prediction_pattern)
    df = cases_to_frame(cases)
    out = paths["plans"] / "target_date_inference_plan.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print()
    print(f"wrote plan: {out}")
    return cases


def run_postprocess(cfg: PipelineConfig, paths: dict[str, Path], force: bool = False, force_figures: bool = False):
    t0 = time.perf_counter()
    cases = write_plan(cfg, paths)
    all_images = []
    metrics_rows = []

    for case in tqdm(cases, desc="target-date cases"):
        out_name = f"TMP2m_hot_extremes_target{case.target_label}_lead{case.lead_days:03d}_init{case.init_label}.nc"
        out_path = paths["processed"] / out_name

        pred_daily = load_prediction_daily_tmax(
            case.prediction_path,
            case.init_time,
            cfg.expected_members,
            case.required_6hour_steps,
            cfg.bbox,
        )

        obs_daily = observed_target_date(
            cfg.era5_zarr,
            case.target_date,
            cfg.bbox,
            pred_daily,
            paths["cache"],
            force=False,
        )

        baseline_end = cfg.baseline_end_year or (case.target_date.year - 1)
        baseline, threshold = sliding_window_baseline_threshold(
            cfg.era5_zarr,
            case.target_date,
            15,
            cfg.baseline_start_year,
            baseline_end,
            cfg.bbox,
            pred_daily,
            paths["cache"],
            paths["processed"],
            force=False,
        )

        ds = compute_target_date_extremes(
            pred_daily,
            obs_daily,
            baseline,
            threshold,
            case.target_date,
            out_path,
            force=force,
        )

        title = (
            f"ACE2S TMP2m target-date hot extremes | target {case.target_date:%Y-%m-%d}"
            f" | lead {case.lead_days}d | init {case.init_time:%Y-%m-%d}"
        )
        stem = f"TMP2m_hot_extreme_target{case.target_label}_lead{case.lead_days:03d}_init{case.init_label}"
        all_images += targetdate_diagnostics(
            ds,
            title,
            paths["figures"],
            stem,
            cfg.figure_formats,
            cfg.max_map_members,
            force=force or force_figures,
        )

        metrics_rows.append(compute_case_metrics(ds, case))

    metrics_csv = paths["metrics"] / "lead_comparison_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_csv, index=False)
    print(f"wrote metrics: {metrics_csv}", flush=True)

    summary = {"seconds": round(time.perf_counter() - t0, 3), "n_cases": len(cases)}
    (paths["metrics"] / "postprocess_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"done in {summary['seconds']} seconds")


def run_sanity(cfg: PipelineConfig, paths: dict[str, Path], force: bool = False, force_figures: bool = False):
    t0 = time.perf_counter()
    cases = build_cases(cfg.target_dates, cfg.lead_days, cfg.prediction_pattern)
    df = cases_to_frame(cases)
    plan_path = paths["plans"] / "target_date_inference_plan_synthetic.csv"
    df.to_csv(plan_path, index=False)
    print(f"wrote synthetic plan: {plan_path}")
    print(df.to_string(index=False))

    all_images = []
    for case in tqdm(cases, desc="synthetic target-date cases"):
        pred_path = paths["synthetic_predictions"] / f"synthetic_TMP2m_init{case.init_label}_horizon{case.required_horizon_days:03d}d.nc"
        if force or not pred_path.exists():
            make_synthetic_prediction(case, pred_path, members=3)

        pred_daily = load_prediction_daily_tmax(
            pred_path,
            case.init_time,
            expected_members=3,
            expected_steps=case.required_6hour_steps,
            bbox=cfg.bbox,
        )

        obs_daily = make_synthetic_era5_target_date(case.target_date, pred_daily)
        baseline, threshold = make_synthetic_baseline_threshold(pred_daily)
        out_path = paths["processed"] / f"TMP2m_hot_extremes_target{case.target_label}_lead{case.lead_days:03d}_init{case.init_label}.nc"

        ds = compute_target_date_extremes(
            pred_daily,
            obs_daily,
            baseline,
            threshold,
            case.target_date,
            out_path,
            force=True,
        )

        title = f"SYNTHETIC target-date sanity | target {case.target_label} | lead {case.lead_days}d | init {case.init_label}"
        stem = f"TMP2m_hot_extreme_target{case.target_label}_lead{case.lead_days:03d}_init{case.init_label}"
        all_images += targetdate_diagnostics(ds, title, paths["figures"], stem, ["png"], max_map_members=3, force=True)

    write_index(paths["figures"], all_images)
    print(f"sanity done in {round(time.perf_counter() - t0, 3)} seconds")


def main():
    args = parse_args()
    cfg = PipelineConfig.from_yaml(args.config)
    paths = ensure_layout(cfg.output_root)

    if args.cmd == "plan":
        write_plan(cfg, paths)
    elif args.cmd == "postprocess":
        run_postprocess(cfg, paths, force=args.force, force_figures=args.force_figures)
    elif args.cmd == "sanity":
        run_sanity(cfg, paths, force=args.force, force_figures=args.force_figures)


if __name__ == "__main__":
    main()
