# ACE2-ERA5 lag ensemble experiment

Seasonal forecast skill of winter temperature extremes using [ACE2-ERA5](https://huggingface.co/allenai/ace2-era5), an ML atmospheric emulator trained on ERA5 reanalysis.

The basic setup: initialize 25-member ensembles on November 1 of each year and run forward 90 days. We then ask how well the predicted probability of a heat or cold extreme at 30/60/90-day lead matches what ERA5 actually observed that winter — scored across years at each grid cell over North America.

## Experiment

- **Years**: 1980–2000 (21 years, run in two batches)
- **Ensemble**: 25 members per year, initialized at 6-hour offsets centered on Nov 1
- **Lead times**: 30, 60, 90 days
- **Extremes**: Tmax > 90th percentile (heat), Tmin < 10th percentile (cold), thresholds from ERA5 1940–2022 climatology
- **Skill metrics**: Spearman ρ, Kendall τ, Brier score — computed pointwise across years at each CONUS grid cell

## Pipeline

```
make_lag_ics_nov.py          # build 25-member IC files from ERA5
convert_forcing_ace2era5.py  # build annual forcing files
run_lag_inference_nov.py     # orchestrate GPU inference (calls 07_run_ace2s_inference_with_ensemble_fix.py)
combine_lag_nov.py           # extract Tmax/Tmin at lead days from raw output
rank_corr_analysis.py        # compute skill maps and generate figures
```

Only the inference step (`run_lag_inference_nov.py` → `07_run_ace2s_inference_with_ensemble_fix.py`) requires a GPU. Everything else is CPU.

## Running inference

```bash
python scripts/run_lag_inference_nov.py \
  --years 1980,1981,...,1989 \
  --skip-existing \
  --forward-steps-in-memory 40 \
  --python /path/to/python
```

`--skip-existing` checks whether each member's output is complete (validates the netCDF) before skipping, so it's safe to use for resuming interrupted runs.

## Outputs

Figures go to `outputs/lag_10yr/figures/` — 18 maps total (3 leads × 2 extremes × 3 metrics). Skill metric netCDFs are in `outputs/lag_10yr/rank_corr/`.

Raw inference output (~150MB per member) and combined Tmax/Tmin files are not tracked in git.

## Environment

```bash
pip install fme  # AI2 full model emulator package
pip install -r requirements-ace2s-smoke.txt
```

Checkpoint: `ace2_era5_ckpt.tar` (not in repo, ~3.4GB).
